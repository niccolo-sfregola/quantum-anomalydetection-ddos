import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, average_precision_score,
    precision_score, recall_score,
)

# Reuse all the data loading and plotting machinery from the unsupervised file.
# This guarantees consistency between the two pipelines.
from unsupervised import (
    select_feature_columns,
    load_split,
    load_all,
    plot_score_histogram,
    plot_pr_roc,
    plot_threshold_sweep,
    plot_confusion_matrix,
    plot_score_timeline,
    metrics_for_split,
    _save,
    N_WINDOWS,
    COLOR_BENIGN, COLOR_ATTACK,
)


# ── Logistic-regression-specific plot ─────────────────────────────────────────

def plot_feature_importance(weights: np.ndarray,
                             feature_names: list[str],
                             out: Path):
    """
    Sorted bar chart of |coefficient| from the fitted logistic regression.
    Tells us which features the linear model relied on most.
    """
    importance = np.abs(weights)
    order = np.argsort(importance)[::-1]
    sorted_w = weights[order]
    sorted_n = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(feature_names))))
    colors = [COLOR_ATTACK if w > 0 else COLOR_BENIGN for w in sorted_w]
    ax.barh(range(len(sorted_w)), sorted_w, color=colors)
    ax.set_yticks(range(len(sorted_w)))
    ax.set_yticklabels(sorted_n, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Coefficient (positive → predicts attack)")
    ax.set_title("Logistic regression feature weights")
    _save(fig, out / "06_feature_importance.png")


# ── Wrap a fitted sklearn model so the unsupervised plot helpers work ─────────

class _FittedModelResult:
    """
    Mirrors the dict-shape that phase3_unsupervised plot helpers expect:
        result["scores"][(kind, split)]      → np.ndarray
        result["labels"][(kind, split)]      → np.ndarray
        result["predictions"][(kind, split)] → np.ndarray
        result["threshold"]                  → float
        result["name"]                       → str
    """
    def __init__(self, name: str, threshold: float):
        self.name        = name
        self.threshold   = threshold
        self.scores      = {}
        self.labels      = {}
        self.predictions = {}

    def __getitem__(self, key):
        return getattr(self, key)


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Pick the threshold that maximises F1 on the (validation) scores."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        # No positives or no negatives — fallback to median
        return float(np.median(scores)), 0.0
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    # `thresholds` has length len(precisions) - 1; use only matching indices
    if len(thresholds) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_supervised(
    enriched_root: str | Path,
    output_dir: str | Path,
    audit_root: str | Path,
    feature_set: str = "combined",
    C: float = 1.0,
    seed: int = 42,
    plot_check: bool = False 
):
    """
    Full supervised pipeline. Fits a logistic regression on (X, y),
    tunes the decision threshold on validation, evaluates on test, saves plots.

    Args:
        enriched_root : Phase 2 output (with attack/normal × train/val/test)
        output_dir    : where to save plots, summary.json, model.npz
        audit_root    : root of the cleaned/audit tree (for ground-truth labels)
        feature_set   : "combined" | "quantum_only" | "classical_only"
        C             : inverse L2 regularisation strength (smaller = stronger)
        seed          : random seed
    """
    enriched_root = Path(enriched_root)
    audit_root    = Path(audit_root)
    output_dir    = Path(output_dir)
    plots_dir     = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    print(f"[supervised] Enriched root: {enriched_root}")
    print(f"[supervised] Audit root   : {audit_root}")
    print(f"[supervised] Feature set  : {feature_set}")

    # ── Load all data ─────────────────────────────────────────────────────
    data = load_all(enriched_root, audit_root, feature_set)
    feat_cols = data[("normal", "train")]["feat_cols"]
    print(f"[supervised] Feature columns ({len(feat_cols)}): {feat_cols}")

    # Sanity: confirm labels are NOT among the features (no leakage).
    forbidden = {"is_seeded_ddos", "burst_id", "burst_phase", "Label", "Attack"}
    leaks = forbidden.intersection(feat_cols)
    if leaks:
        raise RuntimeError(
            f"Leakage detected: ground-truth columns appear in features: {leaks}"
        )

    for (kind, split), blk in data.items():
        n_windows = len(blk["X"])
        n_attack  = int(blk["y"].sum())
        print(f"[supervised]   {kind:6s}/{split:10s}: {n_windows:4d} windows  "
              f"({n_attack} attack)")

    # ── Assemble training set ─────────────────────────────────────────────
    # Supervised: combine attack/train AND normal/train, with their labels.
    X_train = np.vstack([data[("attack", "train")]["X"],
                          data[("normal", "train")]["X"]])
    y_train = np.concatenate([data[("attack", "train")]["y"],
                               data[("normal", "train")]["y"]])

    if y_train.sum() == 0:
        raise RuntimeError(
            "No attack-labelled windows in training set. "
            "Verify that attack/train enriched files have audit CSVs available."
        )
    if y_train.sum() == len(y_train):
        raise RuntimeError("All training windows labelled attack — likely a bug.")

    print(f"[supervised] Training set: {X_train.shape}  "
          f"({int(y_train.sum())} attack / {y_train.size})")

    # ── Standardise (fit ONLY on training, transform on all) ──────────────
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    # ── Fit logistic regression with class-balanced weights ───────────────
    model = LogisticRegression(
        C            = C,
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = seed,
        solver       = "liblinear",
    )
    model.fit(X_train_s, y_train)
    print(f"[supervised] Fitted. Coefficients: {model.coef_.shape}")

    # ── Score every (kind, split) using the fitted model ─────────────────
    def score_block(X):
        if X.size == 0:
            return np.array([])
        return model.predict_proba(scaler.transform(X))[:, 1]

    raw_scores = {key: score_block(blk["X"]) for key, blk in data.items()}
    raw_labels = {key: blk["y"]              for key, blk in data.items()}

    # ── Tune threshold on the FULL validation set (attack + normal) ──────
    val_scores = np.concatenate([raw_scores[("attack", "validation")],
                                 raw_scores[("normal", "validation")]])
    val_labels = np.concatenate([raw_labels[("attack", "validation")],
                                 raw_labels[("normal", "validation")]])
    threshold, val_f1_at_t = tune_threshold(val_scores, val_labels)
    print(f"[supervised] Best validation threshold: {threshold:.4f}  "
          f"(F1 = {val_f1_at_t:.3f})")

    # ── Build a result object compatible with the unsupervised plot helpers
    result = _FittedModelResult(name="logistic_regression", threshold=threshold)
    for key in raw_scores:
        result.scores[key]      = raw_scores[key]
        result.labels[key]      = raw_labels[key]
        result.predictions[key] = (raw_scores[key] >= threshold).astype(int)

    # ── Plots ────────────────────────────────────────────────────────────

    if plot_check:
        print("[supervised] Generating plots ...")
        for split in ("validation", "test"):
            plot_score_histogram(result,  split, plots_dir)
            plot_pr_roc(result,           split, plots_dir)
            plot_threshold_sweep(result,  split, plots_dir)
            plot_confusion_matrix(result, split, plots_dir)
            plot_score_timeline(result,   split, plots_dir)

        # Logistic-specific plot
        plot_feature_importance(model.coef_.ravel(), feat_cols, plots_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "feature_set"   : feature_set,
        "feature_cols"  : feat_cols,
        "C"             : C,
        "threshold"     : threshold,
        "validation_f1_at_threshold": val_f1_at_t,
        "n_train"       : int(X_train.shape[0]),
        "n_train_attack": int(y_train.sum()),
        "metrics": {
            "validation": metrics_for_split(result, "validation"),
            "test"      : metrics_for_split(result, "test"),
        },
    }
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # ── Save fitted model parameters ─────────────────────────────────────
    np.savez(output_dir / "model.npz",
             coef        = model.coef_,
             intercept   = model.intercept_,
             scaler_mean = scaler.mean_,
             scaler_std  = scaler.scale_,
             feat_cols   = np.array(feat_cols))

    # ── Print compact table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Method':<25s} {'Split':<12s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'AP':>6s}")
    print("-" * 70)
    for split in ("validation", "test"):
        stats = summary["metrics"][split]
        ap = stats.get("ap", float("nan"))
        print(f"{result.name:<25s} {split:<12s} "
              f"{stats['f1']:6.3f} {stats['precision']:6.3f} "
              f"{stats['recall']:6.3f} {ap:6.3f}")
    print("=" * 70)
    print(f"\n[supervised] Done. Outputs in {output_dir}")
    return summary


if __name__ == "__main__":
    # Edit these paths to match your local layout.
    ENRICHED_ROOT = "outputs/option_1/minimal/enriched"
    OUTPUT_DIR    = "outputs/option_1/minimal/supervised_classical_1"
    AUDIT_ROOT    = "cleaned_dataset/option_1"
    FEATURE_SET   = "classical_only"
    

    run_supervised(
        enriched_root = ENRICHED_ROOT,
        output_dir    = OUTPUT_DIR,
        audit_root    = AUDIT_ROOT,
        feature_set   = FEATURE_SET,
        plot_check = False
    )