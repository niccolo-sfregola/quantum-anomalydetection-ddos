import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, average_precision_score,
)


N_WINDOWS    = 15
COLOR_TRAIN  = "#4A90C2"
COLOR_VAL    = "#D85A30"
COLOR_BENIGN = "#4A90C2"
COLOR_ATTACK = "#D85A30"


 
 
# ── Style ─────────────────────────────────────────────────────────────────────
 
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
 
N_WINDOWS    = 15
COLOR_TRAIN  = "#4A90C2"
COLOR_VAL    = "#D85A30"
COLOR_BENIGN = "#4A90C2"
COLOR_ATTACK = "#D85A30"
 
 
# ── Dataset ───────────────────────────────────────────────────────────────────
 
def select_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """
    Select which columns from a Phase 2 enriched CSV are model input.
 
    Feature sets:
        classical_only — non-quantum features (no z_qubit_*, s_rho,
                         trace_distance, s_entanglement)
        quantum_only   — only quantum features (z_qubit_*, s_rho,
                         trace_distance, s_entanglement)
        combined       — everything except `window_id`
    """
    quantum_cols = [c for c in df.columns
                    if c.startswith("z_qubit_")
                    or c in ("s_rho", "trace_distance", "s_entanglement")]
    all_features = [c for c in df.columns if c != "window_id"]
 
    if feature_set == "combined":
        return all_features
    if feature_set == "quantum_only":
        return quantum_cols
    if feature_set == "classical_only":
        return [c for c in all_features if c not in quantum_cols]
    raise ValueError(f"Unknown feature_set '{feature_set}'")
 
 
def load_dataset_split(
    phase2_root: Path,
    split: str,                 # "train" | "validation" | "test"
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load all enriched datasets of a given split.
 
    Returns:
        X         : np.ndarray [N_datasets, 15, n_features]
        y         : np.ndarray [N_datasets, 15]   (0 = benign, 1 = attack)
        stems     : list of dataset stems (for traceability)
        feat_cols : list of feature column names actually used
    """
    enriched_files = []
    for kind in ("attack", "normal"):
        d = phase2_root / kind / split
        if d.exists():
            enriched_files.extend(sorted(d.rglob("*_enriched.csv")))
 
    if not enriched_files:
        raise FileNotFoundError(f"No *_enriched.csv under {phase2_root}/*/{split}")
 
    X_list, y_list, stems = [], [], []
    feat_cols: list[str] = []
 
    for f in enriched_files:
        df = pd.read_csv(f).sort_values("window_id").reset_index(drop=True)
        if not feat_cols:
            feat_cols = select_feature_columns(df, feature_set)
 
        X_list.append(df[feat_cols].values.astype(np.float32))
 
        # Reconstruct labels by joining audit info: a window is "attack" if
        # at least one of its rows had is_seeded_ddos == 1.
        stem      = f.stem.replace("_enriched", "")
        audit_csv = _find_audit_csv(phase2_root, stem)
        labels    = _window_labels_from_audit(audit_csv)
        y_list.append(labels.astype(np.float32))
        stems.append(stem)
 
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y, stems, feat_cols
 
 
def _find_audit_csv(phase2_root: Path, stem: str) -> Path:
    """
    Locate the audit CSV that goes with a given enriched dataset.
    Audit files live under a sibling 'cleaned_dataset/...' tree;
    we expect the user to keep the same folder structure.
    Fallback: look in any subfolder by stem.
    """
    candidates = list(phase2_root.parent.rglob(f"{stem}_audit.csv"))
    if not candidates:
        raise FileNotFoundError(f"No audit CSV found for stem '{stem}'")
    return candidates[0]
 
 
def _window_labels_from_audit(audit_csv: Path) -> np.ndarray:
    """A window is labelled 1 if any of its flows had is_seeded_ddos == 1."""
    audit = pd.read_csv(audit_csv, low_memory=False)
    rows_per_window = 100_000 // N_WINDOWS
    audit["window_id"] = (audit["row_in_window"] // rows_per_window).clip(0, N_WINDOWS - 1)
    return (audit.groupby("window_id")["is_seeded_ddos"].max()
                 .reindex(range(N_WINDOWS), fill_value=0).values)



def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Pick the threshold that maximises F1 on the validation scores."""
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]), float(f1s[best_idx])
 

# ── Plotting ──────────────────────────────────────────────────────────────────
 
def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")
 
 
 
 
def plot_pr_roc(scores: np.ndarray, labels: np.ndarray, out: Path,
                split_name: str = "test"):
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    pr_auc  = average_precision_score(labels, scores)
    roc_auc = auc(fpr, tpr)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(recalls, precisions, color=COLOR_VAL, lw=2)
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title(f"PR curve ({split_name}, AP = {pr_auc:.3f})")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
 
    ax2.plot(fpr, tpr, color=COLOR_VAL, lw=2)
    ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC curve ({split_name}, AUC = {roc_auc:.3f})")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
 
    _save(fig, out / "02_pr_roc.png")
 
 
def plot_threshold_sweep(scores: np.ndarray, labels: np.ndarray, out: Path):
    thresholds = np.linspace(0.01, 0.99, 50)
    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        if pred.sum() == 0:
            f1s.append(0.0); precisions.append(0.0); recalls.append(0.0)
            continue
        from sklearn.metrics import precision_score, recall_score
        f1s.append(f1_score(labels, pred, zero_division=0))
        precisions.append(precision_score(labels, pred, zero_division=0))
        recalls.append(recall_score(labels, pred, zero_division=0))
 
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(thresholds, f1s,        color=COLOR_VAL,   label="F1",        lw=2)
    ax.plot(thresholds, precisions, color=COLOR_TRAIN, label="precision", lw=1.5, alpha=0.8)
    ax.plot(thresholds, recalls,    color="#888",      label="recall",    lw=1.5, alpha=0.8)
    best_t = thresholds[int(np.argmax(f1s))]
    ax.axvline(best_t, color="black", ls="--", alpha=0.4, label=f"best t = {best_t:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold sweep on validation"); ax.legend()
    _save(fig, out / "03_threshold_sweep.png")
 
 
def plot_score_histogram(scores: np.ndarray, labels: np.ndarray, out: Path,
                         split_name: str = "test"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(scores[labels == 0], bins=bins, color=COLOR_BENIGN, alpha=0.6,
            label="benign", density=True)
    ax.hist(scores[labels == 1], bins=bins, color=COLOR_ATTACK, alpha=0.6,
            label="attack", density=True)
    ax.set_xlabel("Predicted score"); ax.set_ylabel("Density")
    ax.set_title(f"Score distribution ({split_name})"); ax.legend()
    _save(fig, out / "04_score_histogram.png")
 
 
def plot_confusion_matrix(scores: np.ndarray, labels: np.ndarray,
                          threshold: float, out: Path):
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, pred)
 
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["benign", "attack"])
    ax.set_yticklabels(["benign", "attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, ax=ax)
    _save(fig, out / "05_confusion_matrix.png")








# ── Plot specific to logistic regression ──────────────────────────────────────

def plot_feature_importance(weights: np.ndarray, feature_names: list[str], out: Path):
    """Sorted bar chart of |coefficient| — what the linear model learned to weight."""
    importance = np.abs(weights)
    order = np.argsort(importance)[::-1]
    sorted_w = weights[order]
    sorted_n = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(feature_names))))
    colors = [COLOR_VAL if w > 0 else COLOR_TRAIN for w in sorted_w]
    ax.barh(range(len(sorted_w)), sorted_w, color=colors)
    ax.set_yticks(range(len(sorted_w)))
    ax.set_yticklabels(sorted_n, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Coefficient (positive → predicts attack)")
    ax.set_title("Logistic regression feature weights")
    _save(fig, out / "06_feature_importance.png")


# ── Top-level orchestrator ────────────────────────────────────────────────────

def train_and_evaluate(
    phase2_root: str | Path,
    output_dir: str | Path,
    feature_set: str = "combined",
    C: float = 1.0,
    seed: int = 42,
):
    """
    Fit a logistic regression on per-window features, evaluate, save plots.

    Args:
        C : inverse regularization strength. Smaller → stronger L2 penalty.
    """
    phase2_root = Path(phase2_root)
    output_dir  = Path(output_dir)
    plots_dir   = output_dir / "plots"
    preds_dir   = output_dir / "predictions"
    plots_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    print(f"[phase3-logistic] Feature set: {feature_set}")

    # ── Load data ─────────────────────────────────────────────────────────
    X_train_seq, y_train_seq, _,         feat_cols = load_dataset_split(phase2_root, "train",      feature_set)
    X_val_seq,   y_val_seq,   _,         _         = load_dataset_split(phase2_root, "validation", feature_set)
    X_test_seq,  y_test_seq,  test_stems, _        = load_dataset_split(phase2_root, "test",       feature_set)

    # Flatten sequences: each window is one independent sample.
    # [N_datasets, 15, n_features] → [N_datasets * 15, n_features]
    n_features = X_train_seq.shape[2]
    X_train = X_train_seq.reshape(-1, n_features)
    y_train = y_train_seq.ravel()
    X_val   = X_val_seq.reshape(-1, n_features)
    y_val   = y_val_seq.ravel()
    X_test  = X_test_seq.reshape(-1, n_features)
    y_test  = y_test_seq.ravel()

    print(f"[phase3-logistic] train: {X_train.shape}  ({int(y_train.sum())} attack / {y_train.size})")
    print(f"[phase3-logistic] val  : {X_val.shape}    ({int(y_val.sum())} attack / {y_val.size})")
    print(f"[phase3-logistic] test : {X_test.shape}   ({int(y_test.sum())} attack / {y_test.size})")
    print(f"[phase3-logistic] features used ({n_features}): {feat_cols}")

    # ── Scale + fit ───────────────────────────────────────────────────────
    scaler  = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(
        C            = C,
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = seed,
        solver       = "liblinear",
    )
    model.fit(X_train_s, y_train)
    print(f"[phase3-logistic] Fitted. Coefficients: {model.coef_.shape}")

    # ── Predict scores ────────────────────────────────────────────────────
    val_scores  = model.predict_proba(X_val_s)[:, 1]
    test_scores = model.predict_proba(X_test_s)[:, 1]

    # ── Threshold tuning on validation ────────────────────────────────────
    threshold, val_f1_at_t = tune_threshold(val_scores, y_val)
    print(f"[phase3-logistic] Best validation threshold: {threshold:.3f}  (F1 = {val_f1_at_t:.3f})")

    # ── Diagnostics ───────────────────────────────────────────────────────
    plot_threshold_sweep(val_scores,  y_val,  plots_dir)
    plot_score_histogram(val_scores,  y_val,  plots_dir, split_name="validation")
    plot_pr_roc(test_scores,          y_test, plots_dir, split_name="test")
    plot_confusion_matrix(test_scores, y_test, threshold, plots_dir)
    plot_feature_importance(model.coef_.ravel(), feat_cols, plots_dir)

    # ── Test metrics ──────────────────────────────────────────────────────
    test_pred = (test_scores >= threshold).astype(int)
    test_f1   = f1_score(y_test, test_pred, zero_division=0)
    test_ap   = average_precision_score(y_test, test_scores)
    print(f"[phase3-logistic] Test F1 = {test_f1:.3f}  AP = {test_ap:.3f}")

    # ── Save per-dataset predictions ──────────────────────────────────────
    test_scores_2d = test_scores.reshape(-1, N_WINDOWS)
    test_pred_2d   = test_pred.reshape(-1, N_WINDOWS)
    for stem, scores_seq, pred_seq in zip(test_stems, test_scores_2d, test_pred_2d):
        df = pd.DataFrame({
            "window_id"       : range(N_WINDOWS),
            "score"           : scores_seq,
            "predicted_attack": pred_seq,
        })
        df.to_csv(preds_dir / f"{stem}_pred.csv", index=False)

    # ── Save model + summary ─────────────────────────────────────────────
    np.savez(output_dir / "model.npz",
             coef       = model.coef_,
             intercept  = model.intercept_,
             scaler_mean= scaler.mean_,
             scaler_std = scaler.scale_,
             feat_cols  = np.array(feat_cols))

    summary = {
        "feature_set"   : feature_set,
        "n_features"    : int(n_features),
        "feature_cols"  : feat_cols,
        "C"             : C,
        "best_threshold": float(threshold),
        "val_f1"        : float(val_f1_at_t),
        "test_f1"       : float(test_f1),
        "test_ap"       : float(test_ap),
    }
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[phase3-logistic] Done. Outputs in {output_dir}")
    return summary


if __name__ == "__main__":
    train_and_evaluate(
        phase2_root  = "quantum-anomalydetection-ddos/outputs/option_1/enriched_data_full",
        output_dir   = "quantum-anomalydetection-ddos/outputs/option_1/logistic_results_1",
        feature_set  = "combined",
    )