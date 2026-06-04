"""Shared utilities for source-IP attribution experiments.

This module intentionally contains no repository-specific imports. It is designed to
be copied into an existing project and used by the candidate feature generator,
quantum-inspired feature map, supervised attribution, unsupervised attribution,
and orchestration scripts.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

LOGGER_NAME = "ip_attribution"

AUDIT_COLUMNS = {
    "scenario",
    "split",
    "dataset_id",
    "row_in_window",
    "is_seeded_ddos",
    "burst_id",
    "burst_phase",
    "source_dataset",
}

# Columns that identify rows or carry labels/metadata. They are useful for
# grouping, outputs, metrics, and traceability, but must never be model inputs.
KEY_COLUMNS = {
    "dataset",
    "dataset_name",
    "dataset_family",
    "dataset_id",
    "source_dataset",
    "split",
    "kind",
    "scenario",
    "window_id",
    "window_start",
    "window_end",
    "window_label",
    "src_ip",
    "source_ip",
    "dst_ip",
    "destination_ip",
    "path",
    "file_path",
}

LABEL_COLUMNS = {
    "true_malicious",
    "label_available",
    "window_has_malicious",
    "candidate_has_seeded_flow",
    "is_seeded_ddos",
}

PREDICTION_COLUMNS = {
    "score",
    "predicted_malicious",
    "method",
    "feature_mode",
    "threshold",
    "rank_in_window",
    "preserved_by_label_for_eval",
}

NON_FEATURE_COLUMNS = AUDIT_COLUMNS | KEY_COLUMNS | LABEL_COLUMNS | PREDICTION_COLUMNS

QUANTUM_PREFIXES = ("q_", "quantum_")

DEFAULT_GROUP_COLUMNS = ["dataset", "split", "kind", "window_id"]
DEFAULT_TOP_KS = (1, 5, 10, 20)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)
    return logger


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=json_default)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format for {path}. Use CSV, parquet, feather, or pickle.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix in {".feather", ".ft"}:
        df.reset_index(drop=True).to_feather(path)
    elif suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported table format for {path}. Use CSV, parquet, feather, or pickle.")


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_split_value(value: Any) -> str:
    text = normalize_name(value)
    if text in {"train", "training", "tr"}:
        return "train"
    if text in {"val", "valid", "validation", "dev"}:
        return "validation"
    if text in {"test", "testing", "te"}:
        return "test"
    return text


def split_mask(df: pd.DataFrame, split_name: str) -> pd.Series:
    if "split" not in df.columns:
        if split_name == "train":
            return pd.Series(True, index=df.index)
        return pd.Series(False, index=df.index)
    normalized = df["split"].map(normalize_split_value)
    requested = normalize_split_value(split_name)
    return normalized == requested


def has_labels(df: pd.DataFrame, label_col: str = "true_malicious") -> bool:
    if label_col not in df.columns:
        return False
    labels = pd.to_numeric(df[label_col], errors="coerce")
    return labels.notna().any()


def clean_binary_labels(values: pd.Series | np.ndarray) -> np.ndarray:
    labels = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(int)
    return (labels > 0).astype(int).to_numpy()


def is_quantum_column(col: str) -> bool:
    return any(col.startswith(prefix) for prefix in QUANTUM_PREFIXES)


def get_feature_columns(df: pd.DataFrame, feature_mode: str = "combined") -> list[str]:
    """Return numeric model-input columns for a feature mode.

    feature_mode:
      - classical: candidate aggregation features only
      - quantum: quantum-inspired features only
      - combined: both classical and quantum-inspired features
    """
    feature_mode = normalize_name(feature_mode)
    cols: list[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("__"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q_col = is_quantum_column(col)
        if feature_mode == "classical" and q_col:
            continue
        if feature_mode == "quantum" and not q_col:
            continue
        if feature_mode not in {"classical", "quantum", "combined", "all"}:
            raise ValueError(f"Unsupported feature_mode={feature_mode!r}")
        cols.append(col)
    return cols


def available_metadata_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset",
        "dataset_id",
        "source_dataset",
        "split",
        "kind",
        "scenario",
        "window_id",
        "src_ip",
    ]
    return [c for c in preferred if c in df.columns]


def make_prediction_frame(
    df: pd.DataFrame,
    scores: Sequence[float],
    predicted: Sequence[int],
    method: str,
    feature_mode: str,
    threshold: float | None,
    label_col: str = "true_malicious",
) -> pd.DataFrame:
    out_cols = available_metadata_columns(df)
    if label_col in df.columns:
        out_cols.append(label_col)
    pred = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)
    pred["score"] = np.asarray(scores, dtype=float)
    pred["predicted_malicious"] = np.asarray(predicted, dtype=int)
    pred["method"] = method
    pred["feature_mode"] = feature_mode
    pred["threshold"] = np.nan if threshold is None else float(threshold)
    group_cols = [c for c in DEFAULT_GROUP_COLUMNS if c in pred.columns]
    if group_cols:
        pred["rank_in_window"] = pred.groupby(group_cols)["score"].rank(method="first", ascending=False).astype(int)
    return pred


def binary_metrics(
    y_true: Sequence[int] | np.ndarray | None,
    scores: Sequence[float] | np.ndarray,
    predicted: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    scores_arr = np.asarray(scores, dtype=float)
    pred_arr = np.asarray(predicted, dtype=int)
    if y_true is None:
        return {
            "labels_available": False,
            "n_candidates": int(len(scores_arr)),
            "mean_score": float(np.nanmean(scores_arr)) if len(scores_arr) else None,
            "predicted_positive": int(np.nansum(pred_arr)) if len(pred_arr) else 0,
        }

    y = clean_binary_labels(y_true)
    if len(y) == 0:
        return {"labels_available": False, "n_candidates": 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred_arr, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, pred_arr, labels=[0, 1])
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    metrics: dict[str, Any] = {
        "labels_available": True,
        "n_candidates": int(len(y)),
        "support_positive": positives,
        "support_negative": negatives,
        "predicted_positive": int(pred_arr.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "mean_score": float(np.nanmean(scores_arr)) if len(scores_arr) else None,
    }
    if positives > 0:
        try:
            metrics["average_precision"] = float(average_precision_score(y, scores_arr))
        except Exception:
            metrics["average_precision"] = None
    else:
        metrics["average_precision"] = None

    if positives > 0 and negatives > 0:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, scores_arr))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def threshold_sweep(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_thresholds: int = 256,
) -> pd.DataFrame:
    y = clean_binary_labels(y_true)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    if len(s) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "predicted_positive"])

    unique_scores = np.unique(s)
    if len(unique_scores) > max_thresholds:
        qs = np.linspace(0.0, 1.0, max_thresholds)
        thresholds = np.unique(np.quantile(s, qs))
    else:
        thresholds = unique_scores
    # Add thresholds outside the score range so the sweep includes all-positive
    # and all-negative operating points.
    thresholds = np.unique(np.concatenate(([np.nanmin(s) - 1e-12], thresholds, [np.nanmax(s) + 1e-12])))

    rows = []
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def tune_threshold_from_validation(
    y_true: Sequence[int] | np.ndarray | None,
    scores: Sequence[float] | np.ndarray,
    strategy: str = "f1",
    default_threshold: float = 0.5,
    contamination: float = 0.02,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Pick a threshold without ever touching test labels.

    Supported strategies:
      - f1 / validation_f1: maximize validation F1 if labels exist
      - precision_recall_balance: maximize min(precision, recall)
      - quantile: use score quantile based on contamination
      - fixed: return default_threshold
    """
    strategy = normalize_name(strategy)
    scores_arr = np.asarray(scores, dtype=float)
    finite_scores = scores_arr[np.isfinite(scores_arr)]
    if len(finite_scores) == 0:
        return float(default_threshold), pd.DataFrame(), {"strategy": strategy, "reason": "no finite scores"}

    if strategy in {"fixed", "default"}:
        return float(default_threshold), pd.DataFrame(), {"strategy": strategy}

    if strategy in {"quantile", "contamination", "robust_quantile"}:
        q = float(np.clip(1.0 - contamination, 0.0, 1.0))
        threshold = float(np.quantile(finite_scores, q))
        return threshold, pd.DataFrame(), {"strategy": strategy, "contamination": float(contamination), "quantile": q}

    if y_true is None:
        q = float(np.clip(1.0 - contamination, 0.0, 1.0))
        threshold = float(np.quantile(finite_scores, q))
        return threshold, pd.DataFrame(), {
            "strategy": strategy,
            "fallback": "quantile_no_validation_labels",
            "contamination": float(contamination),
            "quantile": q,
        }

    y = clean_binary_labels(y_true)
    if y.sum() == 0 or y.sum() == len(y):
        q = float(np.clip(1.0 - contamination, 0.0, 1.0))
        threshold = float(np.quantile(finite_scores, q))
        return threshold, pd.DataFrame(), {
            "strategy": strategy,
            "fallback": "quantile_one_class_validation_labels",
            "contamination": float(contamination),
            "quantile": q,
        }

    sweep = threshold_sweep(y, scores_arr)
    if sweep.empty:
        return float(default_threshold), sweep, {"strategy": strategy, "fallback": "empty_sweep"}

    if strategy in {"f1", "validation_f1", "max_f1"}:
        sort_cols = ["f1", "recall", "precision"]
    elif strategy in {"precision_recall_balance", "balanced"}:
        sweep = sweep.copy()
        sweep["balance"] = np.minimum(sweep["precision"], sweep["recall"])
        sort_cols = ["balance", "f1"]
    else:
        sort_cols = ["f1", "recall", "precision"]

    best = sweep.sort_values(sort_cols, ascending=False).iloc[0]
    threshold = float(best["threshold"])
    info = {"strategy": strategy, "selected": best.to_dict()}
    return threshold, sweep, info


def _manual_average_precision(sorted_labels: np.ndarray) -> float:
    positives = int(sorted_labels.sum())
    if positives == 0:
        return float("nan")
    hit_count = 0
    precision_sum = 0.0
    for idx, label in enumerate(sorted_labels, start=1):
        if label == 1:
            hit_count += 1
            precision_sum += hit_count / idx
    return precision_sum / positives


def ranking_metrics(
    pred_df: pd.DataFrame,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    score_col: str = "score",
    label_col: str = "true_malicious",
    pred_col: str = "predicted_malicious",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> dict[str, Any]:
    if label_col not in pred_df.columns:
        return {"labels_available": False}

    work = pred_df.copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int)
    groups = [c for c in group_cols if c in work.columns]
    if not groups:
        groups = ["window_id"] if "window_id" in work.columns else []
    if not groups:
        return {"labels_available": True, "reason": "no group columns available"}

    top_ks = sorted({int(k) for k in top_ks if int(k) > 0})
    per_window_rows = []
    for key, grp in work.groupby(groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        true_total = int(grp[label_col].sum())
        if true_total <= 0:
            continue
        ranked = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        labels = ranked[label_col].to_numpy(dtype=int)
        row: dict[str, Any] = {col: val for col, val in zip(groups, key)}
        row["true_malicious_ip_count"] = true_total
        row["candidate_count"] = int(len(ranked))
        row["average_precision"] = _manual_average_precision(labels)
        for k in top_ks:
            row[f"recall_at_{k}"] = float(labels[:k].sum() / true_total)
            row[f"hits_at_{k}"] = int(labels[:k].sum())
        if pred_col in ranked.columns:
            predicted = pd.to_numeric(ranked[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
            true_pred_hits = int(((predicted == 1) & (labels == 1)).sum())
            false_pos = int(((predicted == 1) & (labels == 0)).sum())
            row["predicted_malicious_ip_count"] = int(predicted.sum())
            row["true_malicious_ips_recovered"] = true_pred_hits
            row["false_positives"] = false_pos
        per_window_rows.append(row)

    if not per_window_rows:
        return {
            "labels_available": True,
            "attack_windows_with_positive_ips": 0,
            "note": "No windows with positive source-IP labels were found.",
        }

    per_window = pd.DataFrame(per_window_rows)
    aggregate: dict[str, Any] = {
        "labels_available": True,
        "attack_windows_with_positive_ips": int(len(per_window)),
        "mean_average_precision_over_windows": float(per_window["average_precision"].mean()),
        "median_average_precision_over_windows": float(per_window["average_precision"].median()),
        "mean_true_malicious_ips_per_attack_window": float(per_window["true_malicious_ip_count"].mean()),
    }
    for k in top_ks:
        col = f"recall_at_{k}"
        if col in per_window.columns:
            aggregate[col] = float(per_window[col].mean())
            aggregate[f"median_{col}"] = float(per_window[col].median())
    if "predicted_malicious_ip_count" in per_window.columns:
        aggregate["mean_predicted_malicious_ips_per_attack_window"] = float(
            per_window["predicted_malicious_ip_count"].mean()
        )
        aggregate["mean_true_malicious_ips_recovered_per_attack_window"] = float(
            per_window["true_malicious_ips_recovered"].mean()
        )
        aggregate["mean_false_positives_per_attack_window"] = float(per_window["false_positives"].mean())

    per_dataset: dict[str, Any] = {}
    if "dataset" in per_window.columns:
        for dataset, grp in per_window.groupby("dataset"):
            ds_metrics: dict[str, Any] = {
                "attack_windows_with_positive_ips": int(len(grp)),
                "mean_average_precision_over_windows": float(grp["average_precision"].mean()),
            }
            for k in top_ks:
                col = f"recall_at_{k}"
                if col in grp.columns:
                    ds_metrics[col] = float(grp[col].mean())
            if "predicted_malicious_ip_count" in grp.columns:
                ds_metrics["mean_false_positives_per_attack_window"] = float(grp["false_positives"].mean())
                ds_metrics["mean_predicted_malicious_ips_per_attack_window"] = float(
                    grp["predicted_malicious_ip_count"].mean()
                )
            per_dataset[str(dataset)] = ds_metrics
    aggregate["per_dataset"] = per_dataset
    return aggregate


def metrics_by_split_and_dataset(
    pred_df: pd.DataFrame,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    label_col: str = "true_malicious",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    split_values = ["all"]
    if "split" in pred_df.columns:
        split_values.extend(sorted({normalize_split_value(v) for v in pred_df["split"].dropna().unique()}))
    for split in split_values:
        if split == "all":
            subset = pred_df
        else:
            subset = pred_df[pred_df["split"].map(normalize_split_value) == split]
        if subset.empty:
            continue
        y = subset[label_col] if label_col in subset.columns else None
        result[split] = {
            "candidate_metrics": binary_metrics(y, subset["score"], subset["predicted_malicious"]),
            "ranking_metrics": ranking_metrics(subset, top_ks=top_ks, label_col=label_col),
        }
    return result


def save_per_window_ip_lists(
    pred_df: pd.DataFrame,
    output_dir: str | Path,
    label_col: str = "true_malicious",
) -> Path:
    output_dir = ensure_dir(output_dir)
    group_cols = [c for c in DEFAULT_GROUP_COLUMNS if c in pred_df.columns]
    if not group_cols:
        group_cols = ["window_id"] if "window_id" in pred_df.columns else []
    rows: list[dict[str, Any]] = []
    if not group_cols or "src_ip" not in pred_df.columns:
        path = output_dir / "per_window_malicious_ip_lists.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    for key, grp in pred_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        predicted_ips = grp.loc[grp["predicted_malicious"].astype(int) == 1, "src_ip"].astype(str).tolist()
        row["predicted_malicious_ips"] = ";".join(predicted_ips)
        row["predicted_malicious_ip_count"] = len(predicted_ips)
        if label_col in grp.columns:
            true_ips = grp.loc[pd.to_numeric(grp[label_col], errors="coerce").fillna(0).astype(int) == 1, "src_ip"].astype(str).tolist()
            row["true_malicious_ips"] = ";".join(true_ips)
            row["true_malicious_ip_count"] = len(true_ips)
        rows.append(row)
    path = output_dir / "per_window_malicious_ip_lists.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_score_histogram(pred_df: pd.DataFrame, path: str | Path, label_col: str = "true_malicious") -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(7, 4.5))
    if label_col in pred_df.columns and pred_df[label_col].notna().any():
        labels = clean_binary_labels(pred_df[label_col])
        scores = pred_df["score"].to_numpy(dtype=float)
        benign_scores = scores[labels == 0]
        malicious_scores = scores[labels == 1]
        if len(benign_scores):
            plt.hist(benign_scores, bins=50, alpha=0.65, label="benign")
        if len(malicious_scores):
            plt.hist(malicious_scores, bins=50, alpha=0.65, label="malicious")
        plt.legend()
    else:
        plt.hist(pred_df["score"].to_numpy(dtype=float), bins=50, alpha=0.8, label="unlabeled")
        plt.legend()
    plt.xlabel("Score (higher means more suspicious)")
    plt.ylabel("Candidate source-IP count")
    plt.title("Candidate score distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_precision_recall(pred_df: pd.DataFrame, path: str | Path, label_col: str = "true_malicious") -> None:
    if label_col not in pred_df.columns:
        return
    y = clean_binary_labels(pred_df[label_col])
    if y.sum() == 0:
        return
    scores = pred_df["score"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc(pred_df: pd.DataFrame, path: str | Path, label_col: str = "true_malicious") -> None:
    if label_col not in pred_df.columns:
        return
    y = clean_binary_labels(pred_df[label_col])
    if y.sum() == 0 or y.sum() == len(y):
        return
    scores = pred_df["score"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y, scores)
    auc_value = roc_auc_score(y, scores)
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_threshold_sweep(sweep_df: pd.DataFrame, path: str | Path) -> None:
    if sweep_df is None or sweep_df.empty:
        return
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(7, 4.5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Validation threshold sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion(pred_df: pd.DataFrame, path: str | Path, label_col: str = "true_malicious") -> None:
    if label_col not in pred_df.columns:
        return
    y = clean_binary_labels(pred_df[label_col])
    pred = pd.to_numeric(pred_df["predicted_malicious"], errors="coerce").fillna(0).astype(int).to_numpy()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xticks([0, 1], ["pred benign", "pred malicious"], rotation=20)
    plt.yticks([0, 1], ["true benign", "true malicious"])
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_per_window_counts(pred_df: pd.DataFrame, path: str | Path, label_col: str = "true_malicious") -> None:
    group_cols = [c for c in DEFAULT_GROUP_COLUMNS if c in pred_df.columns]
    if not group_cols:
        return
    rows = []
    for key, grp in pred_df.groupby(group_cols, dropna=False):
        true_count = 0
        if label_col in grp.columns:
            true_count = int(clean_binary_labels(grp[label_col]).sum())
        pred_count = int(pd.to_numeric(grp["predicted_malicious"], errors="coerce").fillna(0).astype(int).sum())
        rows.append({"true": true_count, "predicted": pred_count})
    if not rows:
        return
    counts = pd.DataFrame(rows).sort_values(["true", "predicted"], ascending=False).reset_index(drop=True)
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot(counts.index, counts["predicted"], label="predicted")
    if label_col in pred_df.columns:
        plt.plot(counts.index, counts["true"], label="true")
    plt.xlabel("Attack-window rank / window index")
    plt.ylabel("Malicious source-IP count")
    plt.title("Per-window predicted vs true malicious-IP counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_topk_recovery(pred_df: pd.DataFrame, path: str | Path, top_ks: Iterable[int] = DEFAULT_TOP_KS, label_col: str = "true_malicious") -> None:
    metrics = ranking_metrics(pred_df, top_ks=top_ks, label_col=label_col)
    if not metrics.get("labels_available") or metrics.get("attack_windows_with_positive_ips", 0) == 0:
        return
    xs = []
    ys = []
    for k in sorted({int(k) for k in top_ks if int(k) > 0}):
        key = f"recall_at_{k}"
        if key in metrics:
            xs.append(k)
            ys.append(metrics[key])
    if not xs:
        return
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Top-k candidates per attack window")
    plt.ylabel("Mean recall@k")
    plt.title("Top-k malicious-IP recovery")
    plt.xticks(xs)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    path: str | Path,
    top_n: int = 30,
    title: str = "Feature importance",
) -> None:
    if feature_names is None or importances is None:
        return
    names = np.asarray(list(feature_names), dtype=object)
    vals = np.asarray(importances, dtype=float)
    if len(names) == 0 or len(vals) == 0:
        return
    finite = np.isfinite(vals)
    names = names[finite]
    vals = vals[finite]
    if len(vals) == 0:
        return
    order = np.argsort(np.abs(vals))[-top_n:]
    names = names[order]
    vals = vals[order]
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(8, max(4.5, 0.28 * len(names))))
    plt.barh(np.arange(len(names)), vals)
    plt.yticks(np.arange(len(names)), names)
    plt.xlabel("Importance / coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_standard_plots(
    pred_df: pd.DataFrame,
    output_dir: str | Path,
    sweep_df: pd.DataFrame | None = None,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    label_col: str = "true_malicious",
) -> None:
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    plot_score_histogram(pred_df, plots_dir / "score_histogram.png", label_col=label_col)
    plot_precision_recall(pred_df, plots_dir / "precision_recall_curve.png", label_col=label_col)
    plot_roc(pred_df, plots_dir / "roc_curve.png", label_col=label_col)
    plot_confusion(pred_df, plots_dir / "confusion_matrix.png", label_col=label_col)
    plot_per_window_counts(pred_df, plots_dir / "per_window_predicted_vs_true_counts.png", label_col=label_col)
    plot_topk_recovery(pred_df, plots_dir / "topk_recovery.png", top_ks=top_ks, label_col=label_col)
    if sweep_df is not None and not sweep_df.empty:
        plot_threshold_sweep(sweep_df, plots_dir / "threshold_sweep.png")


def parse_top_ks(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_TOP_KS
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(sorted({int(p) for p in parts if int(p) > 0}))
    return tuple(sorted({int(v) for v in value if int(v) > 0}))
