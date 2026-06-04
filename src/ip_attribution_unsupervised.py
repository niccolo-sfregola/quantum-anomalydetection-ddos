"""Unsupervised malicious source-IP attribution.

The unsupervised detector trains without candidate malicious-IP labels as targets.
By default it fits on candidate behavior from normal/benign training datasets
when such metadata are available. Validation labels, if present, are used only
for optional threshold calibration and evaluation; otherwise thresholding falls
back to a robust contamination quantile of training anomaly scores.
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
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
        normalize_name,
        parse_top_ks,
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
        normalize_name,
        parse_top_ks,
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

LOGGER = logging.getLogger("ip_attribution.unsupervised")


@dataclass
class UnsupervisedConfig:
    features_path: Path
    output_dir: Path
    feature_mode: str = "combined"
    model_type: str = "isolation_forest"
    fit_scope: str = "normal_train"
    threshold_strategy: str = "quantile"
    default_threshold: float = 0.0
    contamination: float = 0.02
    random_seed: int = 42
    iforest_n_estimators: int = 300
    iforest_max_samples: str | int | float = "auto"
    ocsvm_nu: float | None = None
    lof_n_neighbors: int = 35
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS
    output_format: str = "csv"

    def __post_init__(self) -> None:
        self.features_path = Path(self.features_path)
        self.output_dir = Path(self.output_dir)
        self.top_ks = tuple(int(k) for k in self.top_ks)


def build_detector(config: UnsupervisedConfig) -> Pipeline:
    model_type = normalize_name(config.model_type)
    contamination = float(np.clip(config.contamination, 1e-5, 0.49))
    if model_type in {"isolation_forest", "iforest", "isoforest"}:
        clf = IsolationForest(
            n_estimators=config.iforest_n_estimators,
            max_samples=config.iforest_max_samples,
            contamination=contamination,
            random_state=config.random_seed,
            n_jobs=-1,
        )
    elif model_type in {"one_class_svm", "ocsvm", "oneclasssvm"}:
        clf = OneClassSVM(kernel="rbf", gamma="scale", nu=config.ocsvm_nu or contamination)
    elif model_type in {"local_outlier_factor", "lof"}:
        clf = LocalOutlierFactor(
            n_neighbors=config.lof_n_neighbors,
            novelty=True,
            contamination=contamination,
            n_jobs=-1,
        )
    elif model_type in {"elliptic_envelope", "elliptic"}:
        clf = EllipticEnvelope(contamination=contamination, random_state=config.random_seed, support_fraction=None)
    else:
        raise ValueError(f"Unsupported unsupervised model_type={config.model_type!r}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def select_fit_mask(df: pd.DataFrame, config: UnsupervisedConfig) -> tuple[pd.Series, dict[str, Any]]:
    train = split_mask(df, "train")
    if not train.any():
        LOGGER.warning("No explicit train split found; unsupervised detector will fit on all rows.")
        train = pd.Series(True, index=df.index)
    scope = normalize_name(config.fit_scope)
    info: dict[str, Any] = {"fit_scope": config.fit_scope, "uses_candidate_labels_as_targets": False}

    if scope in {"train_all", "all_train", "train"}:
        mask = train.copy()
        info["description"] = "all training candidates"
    elif scope in {"normal_train", "benign_train"}:
        if "kind" in df.columns:
            kind = df["kind"].astype(str).map(normalize_name)
            normal = kind.str.contains("normal|benign|clean", regex=True, na=False)
            mask = train & normal
            if not mask.any():
                LOGGER.warning("No normal/benign training candidates found; falling back to all training candidates.")
                mask = train.copy()
                info["fallback"] = "train_all_no_normal_kind_rows"
            info["description"] = "normal/benign training datasets only when available"
        else:
            mask = train.copy()
            info["fallback"] = "train_all_no_kind_column"
            info["description"] = "all training candidates because kind metadata is missing"
    elif scope in {"non_attack_windows", "label_non_attack_windows"}:
        if "window_has_malicious" not in df.columns:
            mask = train.copy()
            info["fallback"] = "train_all_no_window_has_malicious_column"
        else:
            win_benign = pd.to_numeric(df["window_has_malicious"], errors="coerce").fillna(0).astype(int) == 0
            mask = train & win_benign
            info["uses_audit_labels_for_fit_filter"] = True
            info["description"] = "training candidates from audit-labeled non-attack windows only"
            if not mask.any():
                mask = train.copy()
                info["fallback"] = "train_all_no_non_attack_windows"
    elif scope in {"label_benign_train", "benign_label_train"}:
        if "true_malicious" not in df.columns:
            mask = train.copy()
            info["fallback"] = "train_all_no_true_malicious_column"
        else:
            benign = pd.to_numeric(df["true_malicious"], errors="coerce").fillna(0).astype(int) == 0
            mask = train & benign
            info["uses_audit_labels_for_fit_filter"] = True
            info["description"] = "training candidates excluding audit-labeled malicious source IPs"
            if not mask.any():
                mask = train.copy()
                info["fallback"] = "train_all_no_label_benign_rows"
    else:
        raise ValueError(f"Unsupported fit_scope={config.fit_scope!r}")

    info["n_fit_rows"] = int(mask.sum())
    return mask, info


def _fit_detector(model: Pipeline, X_fit: pd.DataFrame) -> None:
    model.fit(X_fit)


def _score_detector(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        normality = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    elif hasattr(model, "score_samples"):
        normality = np.asarray(model.score_samples(X), dtype=float).reshape(-1)
    else:
        raise RuntimeError("Unsupervised detector does not expose decision_function or score_samples.")
    # Convert normality to anomaly score: higher means more suspicious.
    return -normality


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


def run_unsupervised_attribution(config: UnsupervisedConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    df = read_table(config.features_path)
    summary: dict[str, Any] = {"config": asdict(config), "features_path": str(config.features_path)}

    if df.empty:
        summary.update({"status": "skipped", "reason": "empty feature table"})
        save_json(summary, output_dir / "summary.json")
        return summary

    feature_cols = get_feature_columns(df, feature_mode=config.feature_mode)
    if not feature_cols:
        summary.update({"status": "skipped", "reason": f"No {config.feature_mode} numeric feature columns available."})
        save_json(summary, output_dir / "summary.json")
        return summary

    fit_mask, fit_info = select_fit_mask(df, config)
    if not fit_mask.any():
        summary.update({"status": "skipped", "reason": "No rows selected for unsupervised fitting.", "fit_info": fit_info})
        save_json(summary, output_dir / "summary.json")
        return summary

    model = build_detector(config)
    X_fit = df.loc[fit_mask, feature_cols]
    _fit_detector(model, X_fit)

    fit_scores = _score_detector(model, X_fit)
    validation_mask = split_mask(df, "validation")
    test_mask = split_mask(df, "test")
    threshold_strategy = normalize_name(config.threshold_strategy)

    threshold_info: dict[str, Any]
    sweep = pd.DataFrame()
    if threshold_strategy in {"f1", "validation_f1", "max_f1", "precision_recall_balance", "balanced"}:
        if validation_mask.any() and has_labels(df.loc[validation_mask], "true_malicious"):
            val_scores = _score_detector(model, df.loc[validation_mask, feature_cols])
            val_labels = clean_binary_labels(df.loc[validation_mask, "true_malicious"])
            threshold, sweep, threshold_info = tune_threshold_from_validation(
                val_labels,
                val_scores,
                strategy=threshold_strategy,
                default_threshold=config.default_threshold,
                contamination=config.contamination,
            )
            threshold_info["validation_labels_used_for_threshold_only"] = True
        else:
            q = float(np.clip(1.0 - config.contamination, 0.0, 1.0))
            threshold = float(np.quantile(fit_scores[np.isfinite(fit_scores)], q))
            threshold_info = {
                "strategy": threshold_strategy,
                "fallback": "fit_score_quantile_no_validation_labels",
                "contamination": float(config.contamination),
                "quantile": q,
            }
    elif threshold_strategy in {"fixed", "default"}:
        threshold = float(config.default_threshold)
        threshold_info = {"strategy": threshold_strategy, "default_threshold": threshold}
    else:
        q = float(np.clip(1.0 - config.contamination, 0.0, 1.0))
        threshold = float(np.quantile(fit_scores[np.isfinite(fit_scores)], q))
        threshold_info = {
            "strategy": threshold_strategy,
            "contamination": float(config.contamination),
            "quantile": q,
            "source": "fit_scores",
        }

    all_scores = _score_detector(model, df[feature_cols])
    predicted = (all_scores >= threshold).astype(int)
    pred = make_prediction_frame(
        df,
        scores=all_scores,
        predicted=predicted,
        method=f"unsupervised_{config.model_type}",
        feature_mode=config.feature_mode,
        threshold=threshold,
        label_col="true_malicious",
    )

    prediction_paths = _write_predictions_by_split(pred, output_dir, config.output_format)
    if not sweep.empty:
        sweep.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)

    if has_labels(pred, "true_malicious"):
        all_metrics = {
            "candidate_metrics": binary_metrics(pred["true_malicious"], pred["score"], pred["predicted_malicious"]),
            "ranking_metrics": ranking_metrics(pred, top_ks=config.top_ks, label_col="true_malicious"),
            "by_split": metrics_by_split_and_dataset(pred, top_ks=config.top_ks, label_col="true_malicious"),
        }
    else:
        all_metrics = {
            "candidate_metrics": binary_metrics(None, pred["score"], pred["predicted_malicious"]),
            "ranking_metrics": {"labels_available": False},
            "by_split": {},
        }
    save_json(all_metrics, output_dir / "metrics_all.json")

    if validation_mask.any() and has_labels(pred.loc[validation_mask], "true_malicious"):
        val_pred = pred.loc[validation_mask].reset_index(drop=True)
        save_json(
            {
                "candidate_metrics": binary_metrics(val_pred["true_malicious"], val_pred["score"], val_pred["predicted_malicious"]),
                "ranking_metrics": ranking_metrics(val_pred, top_ks=config.top_ks, label_col="true_malicious"),
            },
            output_dir / "metrics_validation.json",
        )
    if test_mask.any() and has_labels(pred.loc[test_mask], "true_malicious"):
        test_pred = pred.loc[test_mask].reset_index(drop=True)
        save_json(
            {
                "candidate_metrics": binary_metrics(test_pred["true_malicious"], test_pred["score"], test_pred["predicted_malicious"]),
                "ranking_metrics": ranking_metrics(test_pred, top_ks=config.top_ks, label_col="true_malicious"),
                "note": "Test metrics use the frozen unsupervised threshold; no test retuning was performed.",
            },
            output_dir / "metrics_test.json",
        )

    save_per_window_ip_lists(pred, output_dir, label_col="true_malicious")
    save_standard_plots(pred, output_dir, sweep_df=sweep, top_ks=config.top_ks, label_col="true_malicious")

    model_artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_mode": config.feature_mode,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "fit_info": fit_info,
        "config": config,
    }
    model_path = output_dir / "unsupervised_model.joblib"
    joblib.dump(model_artifact, model_path)

    summary.update(
        {
            "status": "ok",
            "method": f"unsupervised_{config.model_type}",
            "feature_mode": config.feature_mode,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "threshold": threshold,
            "threshold_info": threshold_info,
            "fit_info": fit_info,
            "prediction_paths": prediction_paths,
            "model_path": str(model_path),
            "metrics_path": str(output_dir / "metrics_all.json"),
        }
    )
    save_json(summary, output_dir / "summary.json")
    LOGGER.info("Unsupervised attribution complete: %s", output_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate unsupervised malicious source-IP attribution.")
    parser.add_argument("--features-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-mode", default="combined", choices=["classical", "quantum", "combined"])
    parser.add_argument("--model-type", default="isolation_forest", choices=["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"])
    parser.add_argument("--fit-scope", default="normal_train", choices=["normal_train", "train_all", "non_attack_windows", "label_benign_train"])
    parser.add_argument("--threshold-strategy", default="quantile", choices=["quantile", "fixed", "f1", "precision_recall_balance"])
    parser.add_argument("--default-threshold", type=float, default=0.0)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--iforest-n-estimators", type=int, default=300)
    parser.add_argument("--iforest-max-samples", default="auto")
    parser.add_argument("--ocsvm-nu", type=float, default=None)
    parser.add_argument("--lof-n-neighbors", type=int, default=35)
    parser.add_argument("--top-ks", default="1,5,10,20")
    parser.add_argument("--output-format", default="csv", choices=["csv", "parquet", "feather", "pkl"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> UnsupervisedConfig:
    max_samples: str | int | float
    try:
        if str(args.iforest_max_samples).lower() == "auto":
            max_samples = "auto"
        elif "." in str(args.iforest_max_samples):
            max_samples = float(args.iforest_max_samples)
        else:
            max_samples = int(args.iforest_max_samples)
    except ValueError:
        max_samples = "auto"
    return UnsupervisedConfig(
        features_path=args.features_path,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        model_type=args.model_type,
        fit_scope=args.fit_scope,
        threshold_strategy=args.threshold_strategy,
        default_threshold=args.default_threshold,
        contamination=args.contamination,
        random_seed=args.random_seed,
        iforest_n_estimators=args.iforest_n_estimators,
        iforest_max_samples=max_samples,
        ocsvm_nu=args.ocsvm_nu,
        lof_n_neighbors=args.lof_n_neighbors,
        top_ks=parse_top_ks(args.top_ks),
        output_format=args.output_format,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args)
    run_unsupervised_attribution(config)


if __name__ == "__main__":
    main()
