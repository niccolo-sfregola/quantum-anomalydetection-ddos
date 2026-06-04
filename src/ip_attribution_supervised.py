"""Supervised malicious source-IP attribution.

The supervised model learns from candidate (window_id, src_ip) rows and uses the
audit-derived true_malicious column only as the target/evaluation label. It never
uses audit columns as input features. Scalers and imputers are fitted on training
candidates only, thresholds are selected on validation candidates only when
validation labels exist, and test candidates are evaluated once with the frozen
model and frozen threshold.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from .ip_attribution_common import (
        DEFAULT_TOP_KS,
        binary_metrics,
        clean_binary_labels,
        ensure_dir,
        get_feature_columns,
        has_labels,
        make_prediction_frame,
        metrics_by_split_and_dataset,
        parse_top_ks,
        plot_feature_importance,
        read_table,
        ranking_metrics,
        save_json,
        save_per_window_ip_lists,
        save_standard_plots,
        setup_logging,
        split_mask,
        tune_threshold_from_validation,
        write_table,
    )
except ImportError:  # pragma: no cover
    from ip_attribution_common import (
        DEFAULT_TOP_KS,
        binary_metrics,
        clean_binary_labels,
        ensure_dir,
        get_feature_columns,
        has_labels,
        make_prediction_frame,
        metrics_by_split_and_dataset,
        parse_top_ks,
        plot_feature_importance,
        read_table,
        ranking_metrics,
        save_json,
        save_per_window_ip_lists,
        save_standard_plots,
        setup_logging,
        split_mask,
        tune_threshold_from_validation,
        write_table,
    )

LOGGER = logging.getLogger("ip_attribution.supervised")


@dataclass
class SupervisedConfig:
    features_path: Path
    output_dir: Path
    feature_mode: str = "combined"
    model_type: str = "logistic"
    threshold_strategy: str = "f1"
    default_threshold: float = 0.5
    contamination: float = 0.02
    random_seed: int = 42
    max_iter: int = 1000
    logistic_c: float = 1.0
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    hgb_max_iter: int = 250
    hgb_learning_rate: float = 0.08
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS
    output_format: str = "csv"

    def __post_init__(self) -> None:
        self.features_path = Path(self.features_path)
        self.output_dir = Path(self.output_dir)
        self.top_ks = tuple(int(k) for k in self.top_ks)


class ConstantScoreModel:
    """Fallback model for degenerate one-class training data."""

    def __init__(self, score: float = 0.0):
        self.score = float(score)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConstantScoreModel":
        if len(y):
            self.score = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.score, dtype=float)
        return np.column_stack([1.0 - p, p])


def build_model(config: SupervisedConfig) -> Pipeline | ConstantScoreModel:
    model_type = config.model_type.lower()
    if model_type in {"logistic", "logistic_regression", "lr"}:
        clf = LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=config.random_seed,
            solver="lbfgs",
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
    if model_type in {"random_forest", "rf"}:
        clf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_seed,
            min_samples_leaf=2,
        )
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    if model_type in {"hist_gradient_boosting", "hgb", "histgb"}:
        clf = HistGradientBoostingClassifier(
            max_iter=config.hgb_max_iter,
            learning_rate=config.hgb_learning_rate,
            random_state=config.random_seed,
            l2_regularization=0.0,
        )
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    raise ValueError(f"Unsupported supervised model_type={config.model_type!r}")


def _score_model(model: Pipeline | ConstantScoreModel, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return np.asarray(proba).reshape(-1).astype(float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        # Logistic-like squashing for rankable scores in [0, 1].
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
    raise RuntimeError("Model does not expose predict_proba or decision_function.")


def _fit_model(model: Pipeline | ConstantScoreModel, X_train: pd.DataFrame, y_train: np.ndarray, config: SupervisedConfig) -> None:
    if isinstance(model, ConstantScoreModel):
        model.fit(X_train, y_train)
        return
    if config.model_type.lower() in {"hist_gradient_boosting", "hgb", "histgb"} and len(np.unique(y_train)) > 1:
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, clf__sample_weight=weights)
    else:
        model.fit(X_train, y_train)


def _extract_feature_importance(model: Pipeline | ConstantScoreModel) -> np.ndarray | None:
    if isinstance(model, ConstantScoreModel):
        return None
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if clf is None:
        return None
    if hasattr(clf, "coef_"):
        return np.ravel(clf.coef_)
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    return None


def _write_predictions_by_split(pred: pd.DataFrame, output_dir: Path, output_format: str) -> dict[str, str]:
    suffix = ".csv" if output_format.lower() == "csv" else f".{output_format.lower().lstrip('.')}"
    paths: dict[str, str] = {}
    all_path = output_dir / f"predictions_all{suffix}"
    write_table(pred, all_path)
    paths["all"] = str(all_path)
    if "split" in pred.columns:
        for split_value, split_df in pred.groupby("split", dropna=False):
            split_name = str(split_value).lower().replace(" ", "_")
            path = output_dir / f"predictions_{split_name}{suffix}"
            write_table(split_df.reset_index(drop=True), path)
            paths[str(split_value)] = str(path)
    return paths


def run_supervised_attribution(config: SupervisedConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    df = read_table(config.features_path)
    summary: dict[str, Any] = {"config": asdict(config), "features_path": str(config.features_path)}

    if df.empty:
        summary.update({"status": "skipped", "reason": "empty feature table"})
        save_json(summary, output_dir / "summary.json")
        return summary

    if not has_labels(df, "true_malicious"):
        summary.update(
            {
                "status": "skipped",
                "reason": "true_malicious labels unavailable; supervised attribution cannot train. Unsupervised scoring remains available.",
            }
        )
        save_json(summary, output_dir / "summary.json")
        LOGGER.warning(summary["reason"])
        return summary

    feature_cols = get_feature_columns(df, feature_mode=config.feature_mode)
    if not feature_cols:
        summary.update({"status": "skipped", "reason": f"No {config.feature_mode} numeric feature columns available."})
        save_json(summary, output_dir / "summary.json")
        return summary

    train_mask = split_mask(df, "train")
    validation_mask = split_mask(df, "validation")
    test_mask = split_mask(df, "test")
    if not train_mask.any():
        summary.update({"status": "skipped", "reason": "No training rows found."})
        save_json(summary, output_dir / "summary.json")
        return summary

    y_train = clean_binary_labels(df.loc[train_mask, "true_malicious"])
    X_train = df.loc[train_mask, feature_cols]
    if len(np.unique(y_train)) < 2:
        LOGGER.warning(
            "Training labels contain one class only. A constant-score fallback model will be saved; metrics will likely be uninformative."
        )
        model: Pipeline | ConstantScoreModel = ConstantScoreModel(score=float(np.mean(y_train)) if len(y_train) else 0.0)
        model.fit(X_train, y_train)
    else:
        model = build_model(config)
        _fit_model(model, X_train, y_train, config)

    val_scores: np.ndarray | None = None
    val_labels: np.ndarray | None = None
    if validation_mask.any():
        val_scores = _score_model(model, df.loc[validation_mask, feature_cols])
        val_labels = clean_binary_labels(df.loc[validation_mask, "true_malicious"])
        threshold, sweep, threshold_info = tune_threshold_from_validation(
            val_labels,
            val_scores,
            strategy=config.threshold_strategy,
            default_threshold=config.default_threshold,
            contamination=config.contamination,
        )
    else:
        threshold = float(config.default_threshold)
        sweep = pd.DataFrame()
        threshold_info = {
            "strategy": config.threshold_strategy,
            "fallback": "no_validation_split_default_threshold",
            "default_threshold": float(config.default_threshold),
        }
        LOGGER.warning("No validation split found. Using default supervised threshold %.6f.", threshold)

    all_scores = _score_model(model, df[feature_cols])
    predicted = (all_scores >= threshold).astype(int)
    pred = make_prediction_frame(
        df,
        scores=all_scores,
        predicted=predicted,
        method=f"supervised_{config.model_type}",
        feature_mode=config.feature_mode,
        threshold=threshold,
        label_col="true_malicious",
    )

    prediction_paths = _write_predictions_by_split(pred, output_dir, config.output_format)
    if not sweep.empty:
        sweep.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)

    metrics = metrics_by_split_and_dataset(pred, top_ks=config.top_ks, label_col="true_malicious")
    all_metrics = {
        "candidate_metrics": binary_metrics(pred["true_malicious"], pred["score"], pred["predicted_malicious"]),
        "ranking_metrics": ranking_metrics(pred, top_ks=config.top_ks, label_col="true_malicious"),
        "by_split": metrics,
    }
    save_json(all_metrics, output_dir / "metrics_all.json")
    if validation_mask.any():
        val_pred = pred.loc[validation_mask].reset_index(drop=True)
        save_json(
            {
                "candidate_metrics": binary_metrics(val_pred["true_malicious"], val_pred["score"], val_pred["predicted_malicious"]),
                "ranking_metrics": ranking_metrics(val_pred, top_ks=config.top_ks, label_col="true_malicious"),
            },
            output_dir / "metrics_validation.json",
        )
    if test_mask.any():
        test_pred = pred.loc[test_mask].reset_index(drop=True)
        save_json(
            {
                "candidate_metrics": binary_metrics(test_pred["true_malicious"], test_pred["score"], test_pred["predicted_malicious"]),
                "ranking_metrics": ranking_metrics(test_pred, top_ks=config.top_ks, label_col="true_malicious"),
                "note": "Test metrics use the frozen validation-selected threshold; no test retuning was performed.",
            },
            output_dir / "metrics_test.json",
        )

    save_per_window_ip_lists(pred, output_dir, label_col="true_malicious")
    save_standard_plots(pred, output_dir, sweep_df=sweep, top_ks=config.top_ks, label_col="true_malicious")

    importances = _extract_feature_importance(model)
    if importances is not None:
        plot_feature_importance(
            feature_cols,
            importances,
            output_dir / "plots" / "feature_importance.png",
            title=f"Supervised {config.model_type} feature importance",
        )

    model_artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_mode": config.feature_mode,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "config": config,
    }
    model_path = output_dir / "supervised_model.joblib"
    joblib.dump(model_artifact, model_path)

    summary.update(
        {
            "status": "ok",
            "method": f"supervised_{config.model_type}",
            "feature_mode": config.feature_mode,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "threshold": threshold,
            "threshold_info": threshold_info,
            "prediction_paths": prediction_paths,
            "model_path": str(model_path),
            "metrics_path": str(output_dir / "metrics_all.json"),
        }
    )
    save_json(summary, output_dir / "summary.json")
    LOGGER.info("Supervised attribution complete: %s", output_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate supervised malicious source-IP attribution.")
    parser.add_argument("--features-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-mode", default="combined", choices=["classical", "quantum", "combined"])
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest", "hist_gradient_boosting"])
    parser.add_argument("--threshold-strategy", default="f1", choices=["f1", "precision_recall_balance", "quantile", "fixed"])
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--hgb-max-iter", type=int, default=250)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--top-ks", default="1,5,10,20")
    parser.add_argument("--output-format", default="csv", choices=["csv", "parquet", "feather", "pkl"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> SupervisedConfig:
    return SupervisedConfig(
        features_path=args.features_path,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        model_type=args.model_type,
        threshold_strategy=args.threshold_strategy,
        default_threshold=args.default_threshold,
        contamination=args.contamination,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
        logistic_c=args.logistic_c,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        top_ks=parse_top_ks(args.top_ks),
        output_format=args.output_format,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args)
    run_supervised_attribution(config)


if __name__ == "__main__":
    main()
