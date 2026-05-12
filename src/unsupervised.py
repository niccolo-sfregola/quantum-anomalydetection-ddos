import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, average_precision_score,
    precision_score, recall_score,
)


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

N_WINDOWS    = 15
COLOR_BENIGN = "#4A90C2"
COLOR_ATTACK = "#D85A30"
COLOR_LINE   = "#444"

PERCENTILE_THRESHOLD = 99   # threshold = 95th percentile of scores on normal/train


# ── Data loading ──────────────────────────────────────────────────────────────

def select_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Select which columns of the enriched CSV are model input."""
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


def _find_audit_csv(stem: str, audit_root: Path) -> Optional[Path]:
    """Find the audit CSV for a given dataset stem under `audit_root`."""
    candidates = list(audit_root.rglob(f"{stem}_audit.csv"))
    return candidates[0] if candidates else None


def _window_labels_from_audit(audit_csv: Path) -> np.ndarray:
    """A window is labelled 1 if any of its flows had is_seeded_ddos == 1."""
    audit = pd.read_csv(audit_csv, low_memory=False)
    rows_per_window = 100_000 // N_WINDOWS
    audit["window_id"] = (audit["row_in_window"] // rows_per_window).clip(0, N_WINDOWS - 1)
    return (audit.groupby("window_id")["is_seeded_ddos"].max()
                 .reindex(range(N_WINDOWS), fill_value=0).values)


def load_split(
    enriched_root: Path,
    audit_root: Path,
    kind: str,                  # "attack" | "normal"
    split: str,                 # "train" | "validation" | "test"
    feature_set: str,
    require_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load all enriched files for one (kind, split) combination.

    Args:
        require_labels : if True, raise an error if audit CSVs cannot be
                         found for `attack` files (so silent failures don't
                         produce empty label arrays).
    """
    folder = enriched_root / kind / split
    files = sorted(folder.rglob("*_enriched.csv"))
    if not files:
        return (np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float32), [], [])

    X_list, y_list, stems = [], [], []
    feat_cols: list[str] = []

    for f in files:
        df = pd.read_csv(f).sort_values("window_id").reset_index(drop=True)
        if not feat_cols:
            feat_cols = select_feature_columns(df, feature_set)
        X_list.append(df[feat_cols].values.astype(np.float32))

        stem      = f.stem.replace("_enriched", "")
        audit_csv = _find_audit_csv(stem, audit_root)

        if audit_csv is None:
            if kind == "normal":
                # Normal datasets are all-benign by construction: no audit needed
                labels = np.zeros(N_WINDOWS, dtype=np.float32)
            elif require_labels:
                raise FileNotFoundError(
                    f"Audit CSV for '{stem}' not found under {audit_root}.\n"
                    f"  Pass the correct audit_root to run_unsupervised(...).\n"
                    f"  Hint: search with `find . -name '{stem}_audit.csv'`."
                )
            else:
                labels = np.zeros(N_WINDOWS, dtype=np.float32)
        else:
            labels = _window_labels_from_audit(audit_csv).astype(np.float32)

        y_list.append(labels)
        stems.append(stem)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, feat_cols, stems


def load_all(enriched_root: Path, audit_root: Path, feature_set: str) -> dict:
    """Load every (kind, split) combination into a dict."""
    out: dict = {}
    for kind in ("attack", "normal"):
        for split in ("train", "validation", "test"):
            X, y, feat_cols, stems = load_split(
                enriched_root, audit_root, kind, split, feature_set,
            )
            out[(kind, split)] = {
                "X": X, "y": y, "feat_cols": feat_cols, "stems": stems,
            }
    return out


# ── Methods ───────────────────────────────────────────────────────────────────

def method_A_trace_distance(data: dict, feat_cols: list[str]) -> dict:
    """
    Method A — trace_distance threshold.
    No training. Score = trace_distance column. Threshold = percentile on normal/train.
    """
    if "trace_distance" not in feat_cols:
        raise ValueError("trace_distance column not found in features. "
                         "Make sure feature_set includes quantum features.")
    td_idx = feat_cols.index("trace_distance")

    def get_score(X):
        return X[:, td_idx] if X.size else np.array([])

    # Threshold on normal/train (no labels involved)
    train_scores = get_score(data[("normal", "train")]["X"])
    if len(train_scores) == 0:
        raise ValueError("No normal/train data found")
    threshold = float(np.percentile(train_scores, PERCENTILE_THRESHOLD))

    # Apply to all splits
    return _apply_score_threshold(data, get_score, threshold, name="A_trace_distance")


def method_B_isolation_forest(data: dict, feat_cols: list[str], seed: int = 42) -> dict:
    """
    Method B — Isolation Forest, fitted only on normal/train.
    Score = -decision_function (higher = more anomalous).
    Threshold = percentile on normal/train scores.
    """
    X_train = data[("normal", "train")]["X"]
    if X_train.size == 0:
        raise ValueError("No normal/train data found")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    iso = IsolationForest(
        n_estimators = 200,
        contamination= "auto",
        random_state = seed,
        n_jobs       = -1,
    ).fit(X_train_s)

    def get_score(X):
        if X.size == 0:
            return np.array([])
        Xs = scaler.transform(X)
        return -iso.decision_function(Xs)   # higher = more anomalous

    train_scores = get_score(X_train)
    threshold    = float(np.percentile(train_scores, PERCENTILE_THRESHOLD))

    return _apply_score_threshold(data, get_score, threshold, name="B_isolation_forest")


def method_C_hybrid(method_A_result: dict, method_B_result: dict) -> dict:
    """
    Method C — hybrid score = mean of normalised(A) and normalised(B).
    Each method's scores are min-max normalised to [0, 1] before averaging.
    Threshold is recomputed on normal/train of the hybrid score.
    """
    out: dict = {"name": "C_hybrid", "scores": {}, "labels": {}}

    # Use train scores from A and B to set the normalisation bounds
    a_train = method_A_result["scores"][("normal", "train")]
    b_train = method_B_result["scores"][("normal", "train")]
    a_min, a_max = float(a_train.min()), float(a_train.max())
    b_min, b_max = float(b_train.min()), float(b_train.max())

    def normalise(s, lo, hi):
        if hi - lo < 1e-12:
            return np.zeros_like(s)
        return np.clip((s - lo) / (hi - lo), 0, 1)

    for key in method_A_result["scores"]:
        sa = normalise(method_A_result["scores"][key], a_min, a_max)
        sb = normalise(method_B_result["scores"][key], b_min, b_max)
        out["scores"][key] = 0.5 * (sa + sb)
        out["labels"][key] = method_A_result["labels"][key]

    threshold = float(np.percentile(out["scores"][("normal", "train")],
                                    PERCENTILE_THRESHOLD))
    out["threshold"] = threshold
    out["predictions"] = {
        key: (s >= threshold).astype(int) for key, s in out["scores"].items()
    }
    return out


def _apply_score_threshold(data: dict, score_fn, threshold: float, name: str) -> dict:
    """Apply a scoring function and threshold across all splits."""
    out: dict = {"name": name, "threshold": threshold,
                 "scores": {}, "labels": {}, "predictions": {}}
    for key, blk in data.items():
        s = score_fn(blk["X"])
        out["scores"][key]      = s
        out["labels"][key]      = blk["y"]
        out["predictions"][key] = (s >= threshold).astype(int)
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def plot_score_histogram(method_result: dict, split: str, out: Path):
    """Histogram of scores: benign vs attack, on the chosen split."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Combine attack and normal of this split
    a_scores = method_result["scores"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    n_scores = method_result["scores"].get(("normal", split), np.array([]))

    benign_scores = np.concatenate([n_scores, a_scores[a_labels == 0]])
    attack_scores = a_scores[a_labels == 1]

    if len(benign_scores) == 0 and len(attack_scores) == 0:
        plt.close(fig); return

    all_s = np.concatenate([benign_scores, attack_scores])
    bins = np.linspace(all_s.min(), all_s.max(), 40)

    if len(benign_scores):
        ax.hist(benign_scores, bins=bins, color=COLOR_BENIGN, alpha=0.6,
                label="benign", density=True)
    if len(attack_scores):
        ax.hist(attack_scores, bins=bins, color=COLOR_ATTACK, alpha=0.6,
                label="attack", density=True)

    ax.axvline(method_result["threshold"], color="black", ls="--", alpha=0.6,
               label=f"threshold = {method_result['threshold']:.3f}")
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Density")
    ax.set_title(f"Score distribution ({split}) — {method_result['name']}")
    ax.legend()
    _save(fig, out / f"01_score_histogram_{split}.png")


def plot_pr_roc(method_result: dict, split: str, out: Path):
    """PR curve and ROC on the chosen split (combining attack+normal)."""
    a_scores = method_result["scores"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    n_scores = method_result["scores"].get(("normal", split), np.array([]))
    n_labels = method_result["labels"].get(("normal", split), np.array([]))

    scores = np.concatenate([a_scores, n_scores])
    labels = np.concatenate([a_labels, n_labels])
    if len(scores) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return

    precisions, recalls, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    ap      = average_precision_score(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(recalls, precisions, color=COLOR_ATTACK, lw=2)
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title(f"PR curve ({split}, AP = {ap:.3f})")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)

    ax2.plot(fpr, tpr, color=COLOR_ATTACK, lw=2)
    ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title(f"ROC ({split}, AUC = {roc_auc:.3f})")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
    _save(fig, out / f"02_pr_roc_{split}.png")


def plot_threshold_sweep(method_result: dict, split: str, out: Path):
    """How F1/precision/recall change as the threshold varies."""
    a_scores = method_result["scores"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    n_scores = method_result["scores"].get(("normal", split), np.array([]))
    n_labels = method_result["labels"].get(("normal", split), np.array([]))

    scores = np.concatenate([a_scores, n_scores])
    labels = np.concatenate([a_labels, n_labels])
    if len(scores) == 0 or labels.sum() == 0:
        return

    candidates = np.linspace(scores.min(), scores.max(), 60)
    f1s, prs, rcs = [], [], []
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1s.append(f1_score(labels, pred, zero_division=0))
        prs.append(precision_score(labels, pred, zero_division=0))
        rcs.append(recall_score(labels, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(candidates, f1s, color=COLOR_ATTACK, label="F1", lw=2)
    ax.plot(candidates, prs, color=COLOR_BENIGN, label="precision", lw=1.5, alpha=0.8)
    ax.plot(candidates, rcs, color="#888",       label="recall",    lw=1.5, alpha=0.8)
    ax.axvline(method_result["threshold"], color="black", ls="--", alpha=0.5,
               label=f"used threshold = {method_result['threshold']:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Threshold sweep ({split}) — {method_result['name']}")
    ax.legend()
    _save(fig, out / f"03_threshold_sweep_{split}.png")


def plot_confusion_matrix(method_result: dict, split: str, out: Path):
    """Confusion matrix at the chosen threshold."""
    a_pred   = method_result["predictions"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    n_pred   = method_result["predictions"].get(("normal", split), np.array([]))
    n_labels = method_result["labels"].get(("normal", split), np.array([]))

    pred   = np.concatenate([a_pred, n_pred]).astype(int)
    labels = np.concatenate([a_labels, n_labels]).astype(int)
    if len(pred) == 0:
        return

    cm = confusion_matrix(labels, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["benign", "attack"])
    ax.set_yticklabels(["benign", "attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split}) — {method_result['name']}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    _save(fig, out / f"04_confusion_matrix_{split}.png")


def plot_score_timeline(method_result: dict, split: str, out: Path,
                         max_datasets: int = 4):
    """
    Per-dataset timeline of scores across the 15 windows. Attack windows are
    highlighted. Useful to see whether the model spikes at the right place.
    """
    a_scores = method_result["scores"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    if len(a_scores) == 0:
        return

    # Reshape into [N_datasets, 15]
    n_datasets = len(a_scores) // N_WINDOWS
    if n_datasets == 0:
        return
    s2d = a_scores[: n_datasets * N_WINDOWS].reshape(n_datasets, N_WINDOWS)
    l2d = a_labels[: n_datasets * N_WINDOWS].reshape(n_datasets, N_WINDOWS)

    n_show = min(max_datasets, n_datasets)
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, (s, l, ax) in enumerate(zip(s2d[:n_show], l2d[:n_show], axes)):
        ax.plot(range(N_WINDOWS), s, "o-", color=COLOR_LINE, lw=1.5)
        for w in np.where(l == 1)[0]:
            ax.axvspan(w - 0.4, w + 0.4, color=COLOR_ATTACK, alpha=0.25)
        ax.axhline(method_result["threshold"], color="black", ls="--",
                   alpha=0.5, lw=1)
        ax.set_ylabel(f"attack[{i}]\nscore")
        ax.set_xticks(range(N_WINDOWS))

    axes[-1].set_xlabel("Window ID")
    fig.suptitle(f"Score timeline ({split}) — {method_result['name']}\n"
                 f"orange = ground-truth attack window, dashed = threshold",
                 y=1.02)
    _save(fig, out / f"05_score_timeline_{split}.png")


# ── Evaluation ────────────────────────────────────────────────────────────────

def metrics_for_split(method_result: dict, split: str) -> dict:
    """Compute F1/AP/precision/recall on the merged attack+normal of a split."""
    a_scores = method_result["scores"].get(("attack", split), np.array([]))
    a_labels = method_result["labels"].get(("attack", split), np.array([]))
    n_scores = method_result["scores"].get(("normal", split), np.array([]))
    n_labels = method_result["labels"].get(("normal", split), np.array([]))

    scores = np.concatenate([a_scores, n_scores])
    labels = np.concatenate([a_labels, n_labels])
    if len(scores) == 0:
        return {}

    pred = (scores >= method_result["threshold"]).astype(int)
    out  = {
        "n_samples"   : int(len(labels)),
        "n_attack"    : int(labels.sum()),
        "f1"          : float(f1_score(labels, pred, zero_division=0)),
        "precision"   : float(precision_score(labels, pred, zero_division=0)),
        "recall"      : float(recall_score(labels, pred, zero_division=0)),
    }
    if 0 < labels.sum() < len(labels):
        out["ap"]      = float(average_precision_score(labels, scores))
        fpr, tpr, _    = roc_curve(labels, scores)
        out["roc_auc"] = float(auc(fpr, tpr))
    return out


def make_all_plots(method_result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        plot_score_histogram(method_result, split, output_dir)
        plot_pr_roc(method_result, split, output_dir)
        plot_threshold_sweep(method_result, split, output_dir)
        plot_confusion_matrix(method_result, split, output_dir)
        plot_score_timeline(method_result, split, output_dir)


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_unsupervised(
    enriched_root: str | Path,
    output_dir: str | Path,
    audit_root: str | Path,
    feature_set: str = "combined",
    seed: int = 42,
):
    """
    Full unsupervised pipeline. Saves plots and metrics for methods A, B, C.
 
    Args:
        enriched_root : directory with attack/{train,val,test} and
                        normal/{train,val,test} subfolders containing
                        *_enriched.csv files (Phase 2 output)
        output_dir    : where to save plots and summary.json
        audit_root    : directory containing *_audit.csv files from Phase 0
                        (will be searched recursively with rglob)
        feature_set   : "combined" | "quantum_only" | "classical_only"
    """
    enriched_root = Path(enriched_root)
    output_dir    = Path(output_dir)
    audit_root    = Path(audit_root)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Load all data ─────────────────────────────────────────────────────
    print(f"[unsupervised] Enriched root: {enriched_root}")
    print(f"[unsupervised] Audit root   : {audit_root}")
    print(f"[unsupervised] Feature set  : {feature_set}")
    data = load_all(enriched_root, audit_root, feature_set)
 
    feat_cols = data[("normal", "train")]["feat_cols"]
    print(f"[unsupervised] Feature columns ({len(feat_cols)}): {feat_cols}")
 
    for (kind, split), blk in data.items():
        n_windows = len(blk["X"])
        n_attack  = int(blk["y"].sum())
        print(f"[unsupervised]   {kind:6s}/{split:10s}: {n_windows:4d} windows  "
              f"({n_attack} attack)")
 
    # ── Method A: trace_distance threshold ───────────────────────────────
    has_trace_distance = "trace_distance" in feat_cols
    res_A = None
    if has_trace_distance:
        print("\n[unsupervised] === Method A: trace_distance threshold ===")
        res_A = method_A_trace_distance(data, feat_cols)
        print(f"  threshold = {res_A['threshold']:.4f}")
        make_all_plots(res_A, output_dir / "plots" / "method_A")
    else:
        print("\n[unsupervised] === Method A: SKIPPED ===")
        print("  trace_distance not in feature set → skipping method A and C")
 
    # ── Method B: Isolation Forest ───────────────────────────────────────
    print("\n[unsupervised] === Method B: Isolation Forest ===")
    res_B = method_B_isolation_forest(data, feat_cols, seed=seed)
    print(f"  threshold = {res_B['threshold']:.4f}")
    make_all_plots(res_B, output_dir / "plots" / "method_B")
 
    # ── Method C: Hybrid (only if A is available) ────────────────────────
    res_C = None
    if res_A is not None:
        print("\n[unsupervised] === Method C: Hybrid (mean of A and B) ===")
        res_C = method_C_hybrid(res_A, res_B)
        print(f"  threshold = {res_C['threshold']:.4f}")
        make_all_plots(res_C, output_dir / "plots" / "method_C")
 
    # ── Summary ──────────────────────────────────────────────────────────
    methods_summary = {}
    for res in (res_A, res_B, res_C):
        if res is None:
            continue
        methods_summary[res["name"]] = {
            "threshold" : res["threshold"],
            "validation": metrics_for_split(res, "validation"),
            "test"      : metrics_for_split(res, "test"),
        }
 
    summary = {
        "feature_set"   : feature_set,
        "feature_cols"  : feat_cols,
        "percentile"    : PERCENTILE_THRESHOLD,
        "methods"       : methods_summary,
    }
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
 
    # Print compact table to stdout
    print("\n" + "=" * 70)
    print(f"{'Method':<25s} {'Split':<12s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'AP':>6s}")
    print("-" * 70)
    for mname, m in methods_summary.items():
        for split in ("validation", "test"):
            stats = m[split]
            ap = stats.get("ap", float("nan"))
            print(f"{mname:<25s} {split:<12s} "
                  f"{stats['f1']:6.3f} {stats['precision']:6.3f} "
                  f"{stats['recall']:6.3f} {ap:6.3f}")
    print("=" * 70)
    print(f"\n[unsupervised] Done. Outputs in {output_dir}")
    return summary


if __name__ == "__main__":
    # Edit these paths to match your local layout
    ENRICHED_ROOT = "outputs/option_2/minimal/enriched"
    OUTPUT_DIR    = "outputs/option_2/minimal/unsupervised_results"
    AUDIT_ROOT    = "cleaned_dataset/option_2"   # contains *_audit.csv files
    FEATURE_SET   = "combined"

    run_unsupervised(ENRICHED_ROOT, OUTPUT_DIR, AUDIT_ROOT, FEATURE_SET)