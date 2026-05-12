"""
Feature sets:
    "minimal"  — 4 features: only the strongest IP-diversity signals
    "full"     — 10 features: minimal + flow signature + dst concentration
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

N_WINDOWS = 15

# The two feature sets you can pick from. Each entry is the column name that
# will appear in the output CSV; the values are computed in compute_window_features().
FEATURE_SETS = {
    "minimal": [
        "unique_src_ip",
        "entropy_src_ip",
        "unique_dst_ip",
        "entropy_dst_ip",
    ],
    "full": [
        # Source diversity
        "unique_src_ip",
        "entropy_src_ip",
        # Destination pattern
        "unique_dst_ip",
        "entropy_dst_ip",
        "dst_concentration",
        # Flow signature
        "mean_duration",
        "mean_total_bytes",
        "mean_pkts_per_sec",
        "mean_outbound_ratio",
        "mean_pkt_size",
    ],
}


# ── Statistical helpers ───────────────────────────────────────────────────────

def shannon_entropy(series: pd.Series) -> float:
    """Shannon entropy in bits of a categorical series."""
    if len(series) == 0:
        return 0.0
    p = series.value_counts(normalize=True)
    return float(-(p * np.log2(p + 1e-12)).sum())


def top1_concentration(series: pd.Series) -> float:
    """Fraction of values equal to the most common one. 1.0 = all same."""
    if len(series) == 0:
        return 0.0
    return float(series.value_counts(normalize=True).iloc[0])


# ── Core aggregation ──────────────────────────────────────────────────────────

def compute_window_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """
    Aggregate one cleaned dataset into window-level features.

    Args:
        df          : cleaned flow-level DataFrame with `window_id` column
        feature_set : "minimal" (4) or "full" (10)

    Returns:
        DataFrame with N_WINDOWS rows (one per window) and the columns listed
        in FEATURE_SETS[feature_set] plus `window_id`.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. "
            f"Valid options: {list(FEATURE_SETS.keys())}"
        )

    grp     = df.groupby("window_id")
    columns = FEATURE_SETS[feature_set]
    out     = pd.DataFrame(index=range(N_WINDOWS))
    out.index.name = "window_id"

    # Compute every feature lazily; only those in `columns` are kept.
    available = {
        # Source diversity
        "unique_src_ip"      : grp["src_ip"].nunique(),
        "entropy_src_ip"     : grp["src_ip"].apply(shannon_entropy),

        # Destination pattern
        "unique_dst_ip"      : grp["dst_ip"].nunique(),
        "entropy_dst_ip"     : grp["dst_ip"].apply(shannon_entropy),
        "dst_concentration"  : grp["dst_ip"].apply(top1_concentration),

        # Flow signature (numeric means)
        "mean_duration"      : grp["duration"].mean(),
        "mean_total_bytes"   : grp["total_bytes"].mean(),
        "mean_pkts_per_sec"  : grp["packets_per_second"].mean(),
        "mean_outbound_ratio": grp["outbound_byte_ratio"].mean(),
        "mean_pkt_size"      : grp["packet_size_avg"].mean(),
    }

    for col in columns:
        out[col] = available[col].reindex(range(N_WINDOWS), fill_value=0.0)

    return out.reset_index()


def compute_ip_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-window src_ip frequency table. Used by Phase 2 for density matrix
    construction.

    Returns a DataFrame with columns: window_id, src_ip, count, freq
    """
    counts = (
        df.groupby(["window_id", "src_ip"])
          .size()
          .reset_index(name="count")
    )
    totals = counts.groupby("window_id")["count"].transform("sum")
    counts["freq"] = counts["count"] / totals
    return counts


# ── File-level entry points ───────────────────────────────────────────────────

def aggregate_file(clean_csv: str | Path,
                   output_dir: str | Path,
                   feature_set: str = "full",
                   save_ip_distributions: bool = True,
                   verbose: bool = True) -> Path:
    """
    Aggregate one cleaned flow CSV into window-level features.

    Args:
        clean_csv             : path to *_clean.csv from phase0
        output_dir            : where to save outputs
        feature_set           : "minimal" or "full"
        save_ip_distributions : if True, also save IP frequency table for Phase 2

    Returns:
        path to the windows CSV
    """
    clean_csv  = Path(clean_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[phase1] Loading {clean_csv.name} (feature_set={feature_set}) ...")

    df = pd.read_csv(clean_csv, low_memory=False)
    if "window_id" not in df.columns:
        raise ValueError(
            f"{clean_csv.name} has no `window_id` column. "
            f"Run phase0_preprocessing first."
        )

    # Window features
    features_df = compute_window_features(df, feature_set=feature_set)

    # Save (use clean stem without the trailing _clean)
    stem = clean_csv.stem.replace("_clean", "")
    out_windows = output_dir / f"{stem}_windows.csv"
    features_df.to_csv(out_windows, index=False)
    if verbose:
        print(f"[phase1]   {out_windows.name}  shape {features_df.shape}")

    # IP distributions for Phase 2
    if save_ip_distributions:
        ip_dist_df = compute_ip_distributions(df)
        out_ip = output_dir / f"{stem}_ip_distributions.csv"
        ip_dist_df.to_csv(out_ip, index=False)
        if verbose:
            print(f"[phase1]   {out_ip.name}  shape {ip_dist_df.shape}")

    return out_windows


def aggregate_tree(cleaned_root: str | Path,
                   output_root: str | Path,
                   feature_set: str = "full",
                   save_ip_distributions: bool = True) -> list[Path]:
    """
    Walk the cleaned-data tree and aggregate every *_clean.csv.
    Mirrors the folder structure under `output_root`.
    """
    cleaned_root = Path(cleaned_root)
    output_root  = Path(output_root)

    clean_files = sorted(cleaned_root.rglob("*_clean.csv"))
    print(f"[phase1] Found {len(clean_files)} cleaned file(s) under {cleaned_root}")
    print(f"[phase1] Feature set: '{feature_set}' "
          f"({len(FEATURE_SETS[feature_set])} features per window)")

    out_paths: list[Path] = []
    for clean in clean_files:
        rel_dir = clean.parent.relative_to(cleaned_root)
        out_dir = output_root / rel_dir
        out = aggregate_file(
            clean_csv             = clean,
            output_dir            = out_dir,
            feature_set           = feature_set,
            save_ip_distributions = save_ip_distributions,
        )
        out_paths.append(out)

    print(f"\n[phase1] Done. {len(out_paths)} datasets aggregated → {output_root}")
    return out_paths


if __name__ == "__main__":
    # Edit these paths to match your local layout.
    CLEANED_ROOT = "cleaned_dataset/option_1"
    OUTPUT_ROOT  = "outputs/option_1/full"
    FEATURE_SET  = "full"   # or "full"

    aggregate_tree(
        cleaned_root = CLEANED_ROOT,
        output_root  = OUTPUT_ROOT,
        feature_set  = FEATURE_SET,
    )