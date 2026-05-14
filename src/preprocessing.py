import numpy as np
import pandas as pd
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

DROP_CONSTANT = ["scenario", "split", "dataset_id"]

AUDIT_COLS = [
    "Label", "Attack", "is_seeded_ddos",
    "burst_id", "burst_phase", "source_dataset",
    "row_in_window",
]

HEAVY_TAILED = [
    "duration",
    "packets_per_second", "bytes_per_second",
    "inter_packet_arrival_mean", "inter_packet_arrival_std",
    "total_packets", "total_bytes",
    "packet_size_avg", "packet_size_std",
]

TOP_PROTOCOLS = ["TCP", "UDP", "ICMP"]

# Redundant features identified from the correlation matrix on the cleaned
# (log1p-transformed) flow data. Drop list is explicit (not auto-computed)
# because the schema is small and we want a fixed, reproducible reduction.
#
# Justification:
#   - bytes_per_second        : r > 0.95  with packets_per_second (≈ pkt/s × pkt_size)
#   - inter_packet_arrival_std: r > 0.9   with inter_packet_arrival_mean
#   - inter_packet_arrival_mean: r ≈ -0.97 with packets_per_second (same info, opposite sign)
DROP_REDUNDANT = [
    "bytes_per_second",
    "inter_packet_arrival_std",
    "inter_packet_arrival_mean",
]

DATASET_SCALING = 1_000
ROWS_PER_DATASET = DATASET_SCALING
N_WINDOWS        = 15
ROWS_PER_WINDOW  = ROWS_PER_DATASET // N_WINDOWS    


# ── Cleaning steps ────────────────────────────────────────────────────────────

def split_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate audit (ground-truth) columns from feature columns."""
    audit_present = [c for c in AUDIT_COLS if c in df.columns]
    audit_df      = df[audit_present].copy()
    features_df   = df.drop(columns=audit_present, errors="ignore")
    return features_df, audit_df


def drop_constant(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_CONSTANT if c in df.columns],
                   errors="ignore")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in HEAVY_TAILED:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features that are highly correlated with others (see DROP_REDUNDANT)."""
    return df.drop(columns=[c for c in DROP_REDUNDANT if c in df.columns],
                   errors="ignore")


def encode_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `protocol` (TCP, UDP, ICMP, other). Drops the original."""
    if "protocol" not in df.columns:
        return df
    df = df.copy()
    proto = df["protocol"].astype(str)
    for p in TOP_PROTOCOLS:
        df[f"proto_{p}"] = (proto == p).astype(int)
    df["proto_other"] = (~proto.isin(TOP_PROTOCOLS)).astype(int)
    return df.drop(columns=["protocol"])


def assign_window_id(features_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced schema has no millisecond timestamp.
    We use audit['row_in_window'] (0..99999) to bin into 15 windows.
    """
    features_df = features_df.copy()
    if "row_in_window" not in audit_df.columns:
        raise ValueError("Reduced schema requires audit['row_in_window'] for windowing")
    wid = (audit_df["row_in_window"] // ROWS_PER_WINDOW).clip(0, N_WINDOWS - 1).astype(int)
    features_df["window_id"] = wid.values
    return features_df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline on a single DataFrame."""
    features_df, audit_df = split_audit(df)
    features_df = drop_constant(features_df)
    features_df = impute_missing(features_df)
    features_df = log_transform(features_df)
    features_df = drop_redundant(features_df)
    features_df = encode_protocol(features_df)
    features_df = assign_window_id(features_df, audit_df)
    return features_df, audit_df


def preprocess_file(input_path: str | Path,
                    output_dir: str | Path,
                    verbose: bool = True,
                    dataset_scaling: int | None = None) -> Path:
    """
    Preprocess one CSV. Saves two files in `output_dir`:
        <stem>_clean.csv : feature columns + window_id (model input)
        <stem>_audit.csv : ground-truth columns (validation only)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[phase0] Reading {input_path.name} ...")
    df = pd.read_csv(input_path, low_memory=False)

    if dataset_scaling is not None:
        df = df.iloc[:dataset_scaling].copy()

    if verbose:
        print(f"[phase0]   {len(df):,} rows x {df.shape[1]} cols")

    features_df, audit_df = preprocess_dataframe(df)

    clean_path = output_dir / f"{input_path.stem}_clean.csv"
    audit_path = output_dir / f"{input_path.stem}_audit.csv"

    features_df.to_csv(clean_path, index=False)
    audit_df.to_csv(audit_path,    index=False)

    if verbose:
        print(f"[phase0]   features → {clean_path.name}  ({features_df.shape[1]} cols)")
        print(f"[phase0]   audit    → {audit_path.name}  ({audit_df.shape[1]} cols)")

    return clean_path


def preprocess_tree(input_root: str | Path,
                    output_root: str | Path,
                    pattern: str = "*.csv",
                    dataset_scaling: int | None = None) -> list[Path]:
    """
    Walk the dataset tree and preprocess every CSV, mirroring the folder
    structure under `output_root`.
    """
    input_root  = Path(input_root)
    output_root = Path(output_root)

    csv_files = list(input_root.rglob(pattern))
    print(f"[phase0] Found {len(csv_files)} CSV file(s) under {input_root}")

    cleaned_paths: list[Path] = []
    for csv in csv_files:
        rel_dir = csv.parent.relative_to(input_root)
        out_dir = output_root / rel_dir
        cleaned = preprocess_file(csv, out_dir, verbose=True, dataset_scaling=dataset_scaling)
        cleaned_paths.append(cleaned)

    print(f"\n[phase0] Done. {len(cleaned_paths)} files cleaned → {output_root}")
    return cleaned_paths


if __name__ == "__main__":
    # Edit these paths to match your local layout.
    INPUT_ROOT  = "Datasets/Datasets/Option_2/option2_nf_unsw_base_cse_native_ddos_reduced_schema"
    OUTPUT_ROOT = "cleaned_dataset_1000/option_2"

    preprocess_tree(INPUT_ROOT, OUTPUT_ROOT, dataset_scaling = DATASET_SCALING)