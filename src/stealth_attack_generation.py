"""
Generate stealthier attack datasets from the original raw QCentroid CSV files.

This script creates a complete raw input tree that can be fed directly into
the existing preprocessing -> aggregation -> baseline -> reservoir -> detection
pipeline.

Output layout mirrors the original dataset tree, for example:

    stealth_datasets_100000/option_2/stealth_mimic70_vol100x/
        normal/train/*.csv
        normal/validation/*.csv
        normal/test/*.csv
        attack/train/*_stealth_mimic70_vol100x.csv
        attack/validation/*_stealth_mimic70_vol100x.csv
        attack/test/*_stealth_mimic70_vol100x.csv

Normal files are copied/sliced unchanged. Attack files keep their benign
background rows unchanged, while seeded attack rows are modified as follows:

1. Volume/rate reduction:
       packets_per_second, bytes_per_second, total_packets, total_bytes
   are divided by VOLUME_REDUCTION_FACTOR.

2. Behavioral mimicry:
   selected non-audit numeric attack-flow features are shifted toward the
   benign reference distribution by MIMICRY_STRENGTH_PCT.

The audit columns are preserved for evaluation only. They must still be
excluded from model input by preprocessing/detection.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# User settings
# =============================================================================

OPTION = "option_2"
ROWS_PER_DATASET = 100_000

# Edit this if your raw dataset folder is elsewhere.
SOURCE_ROOT = Path(
    "Datasets/Datasets/Option_2/option2_nf_unsw_base_cse_native_ddos_reduced_schema"
)

# Strength of behavioral mimicry:
#   0   = no behavioral mimicry
#   100 = fully remap attack feature distribution to benign mean/std
MIMICRY_STRENGTH_PCT = 90.0

# Required by the requested stealth attack setting.
VOLUME_REDUCTION_FACTOR = 100.0

# If True, normal datasets are copied into the generated raw tree too.
# Keep this True so the existing preprocessing.py can run on one INPUT_ROOT.
COPY_NORMAL_DATASETS = True

# If not None, each source CSV is sliced to this many rows before writing.
# For the full challenge-sized setting, keep this at 100_000.
DATASET_ROW_LIMIT = ROWS_PER_DATASET

# Sampling limit for estimating benign mean/std. This prevents very large
# memory usage if you later use larger raw files.
BENIGN_SAMPLE_ROWS_PER_FILE = 25_000

# Seed for benign sampling.
RANDOM_SEED = 42

# Whether volume features should also be shifted toward benign behavior after
# the 100x downscaling. Default False, because the volume reduction is the
# deliberate attack-hiding mechanism and should not be undone by mimicry.
MIMIC_VOLUME_FEATURES = False

# Keep this False unless you explicitly want to use benign background rows from
# attack datasets as part of the benign reference distribution.
INCLUDE_BENIGN_ROWS_FROM_ATTACK_FILES_IN_REFERENCE = False

STEALTH_TAG = (
    f"stealth_mimic{int(round(MIMICRY_STRENGTH_PCT))}"
    f"_vol{int(round(VOLUME_REDUCTION_FACTOR))}x"
)

OUTPUT_ROOT = Path(f"stealth_datasets_{ROWS_PER_DATASET}") / OPTION / STEALTH_TAG


# =============================================================================
# Schema settings
# =============================================================================

AUDIT_COLS = {
    "scenario",
    "split",
    "dataset_id",
    "Label",
    "Attack",
    "is_seeded_ddos",
    "burst_id",
    "burst_phase",
    "source_dataset",
    "row_in_window",
}

VOLUME_FEATURES = [
    "packets_per_second",
    "bytes_per_second",
    "total_packets",
    "total_bytes",
]

INTEGER_VOLUME_FEATURES = {
    "total_packets",
    "total_bytes",
}

# Non-audit numeric flow-behavior columns. Do not include ports here: shifting
# ports creates invalid synthetic port numbers and changes the categorical
# service profile in a hard-to-interpret way.
DEFAULT_MIMICRY_FEATURES = [
    "duration",
    "inter_packet_arrival_mean",
    "inter_packet_arrival_std",
    "packet_size_avg",
    "packet_size_std",
    "outbound_byte_ratio",
]

BOUNDED_0_1_FEATURES = {
    "outbound_byte_ratio",
}

NON_NEGATIVE_FEATURES = {
    "duration",
    "packets_per_second",
    "bytes_per_second",
    "inter_packet_arrival_mean",
    "inter_packet_arrival_std",
    "total_packets",
    "total_bytes",
    "packet_size_avg",
    "packet_size_std",
}

SKIP_INPUT_PATTERNS = (
    "_clean.csv",
    "_audit.csv",
    "_windows.csv",
    "_ip_distributions.csv",
    "_enriched.csv",
)


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class FileReport:
    source: str
    output: str
    kind: str
    rows: int
    attack_rows: int
    mimicry_strength_pct: float
    volume_reduction_factor: float
    transformed_columns: list[str]


def _candidate_csv_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.csv")):
        name = path.name
        if any(name.endswith(pattern) for pattern in SKIP_INPUT_PATTERNS):
            continue
        files.append(path)
    return files


def _infer_kind_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)

    if any(p in {"attack", "attacks"} for p in parts):
        return "attack"
    if any(p in {"normal", "benign"} for p in parts):
        return "normal"

    # Fallback for less clean directory names.
    if "attack" in joined or "ddos" in joined:
        return "attack"
    if "normal" in joined or "benign" in joined:
        return "normal"

    return None


def _attack_row_mask(df: pd.DataFrame) -> pd.Series:
    """
    Identify seeded attack rows using audit/label columns.

    Preferred:
        is_seeded_ddos == 1

    Fallbacks:
        Attack != Benign/Normal/0
        Label  != 0
    """
    if "is_seeded_ddos" in df.columns:
        s = pd.to_numeric(df["is_seeded_ddos"], errors="coerce").fillna(0)
        return s > 0

    if "Attack" in df.columns:
        s = df["Attack"].astype(str).str.strip().str.lower()
        benign_values = {"", "0", "nan", "none", "benign", "normal"}
        return ~s.isin(benign_values)

    if "Label" in df.columns:
        numeric = pd.to_numeric(df["Label"], errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0) != 0
        s = df["Label"].astype(str).str.strip().str.lower()
        benign_values = {"", "0", "nan", "none", "benign", "normal"}
        return ~s.isin(benign_values)

    return pd.Series(False, index=df.index)


def _read_source_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if DATASET_ROW_LIMIT is not None:
        df = df.iloc[:DATASET_ROW_LIMIT].copy()
    return df


def _present_numeric_columns(df: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    out = []
    for col in requested:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            out.append(col)
    return out


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    if not np.isfinite(value) or value < 1e-12:
        return 1.0
    return value


def _mimicry_columns() -> list[str]:
    cols = list(DEFAULT_MIMICRY_FEATURES)
    if MIMIC_VOLUME_FEATURES:
        cols.extend(VOLUME_FEATURES)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(cols))


def _build_benign_reference(files: list[Path]) -> dict[str, dict[str, float]]:
    """
    Estimate benign mean/std for mimicry columns from normal datasets.

    Optionally includes benign background rows from attack datasets.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    wanted_cols = _mimicry_columns()

    samples = []

    for path in files:
        kind = _infer_kind_from_path(path)
        if kind not in {"normal", "attack"}:
            continue

        if kind == "attack" and not INCLUDE_BENIGN_ROWS_FROM_ATTACK_FILES_IN_REFERENCE:
            continue

        df = _read_source_csv(path)

        if kind == "attack":
            benign_mask = ~_attack_row_mask(df)
            df = df.loc[benign_mask].copy()

        present = _present_numeric_columns(df, wanted_cols)
        if not present:
            continue

        ref = df[present].apply(pd.to_numeric, errors="coerce")

        if BENIGN_SAMPLE_ROWS_PER_FILE is not None and len(ref) > BENIGN_SAMPLE_ROWS_PER_FILE:
            sampled_idx = rng.choice(ref.index.to_numpy(), size=BENIGN_SAMPLE_ROWS_PER_FILE, replace=False)
            ref = ref.loc[sampled_idx]

        samples.append(ref)

    if not samples:
        raise RuntimeError(
            "Could not build benign reference statistics. "
            "Check SOURCE_ROOT and make sure normal/benign CSVs are present."
        )

    benign = pd.concat(samples, axis=0, ignore_index=True)

    stats: dict[str, dict[str, float]] = {}
    for col in benign.columns:
        s = pd.to_numeric(benign[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "mean": float(s.mean()),
            "std": _safe_std(s),
        }

    if not stats:
        raise RuntimeError("Benign reference statistics are empty after numeric conversion.")

    return stats


def _reduce_volume_features(df: pd.DataFrame, mask: pd.Series) -> list[str]:
    changed = []

    for col in VOLUME_FEATURES:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df.loc[mask, col], errors="coerce")
        scaled = values / float(VOLUME_REDUCTION_FACTOR)

        if col in INTEGER_VOLUME_FEATURES:
            # Keep zero as zero; otherwise enforce at least one unit after scaling.
            scaled = np.where(
                values.fillna(0).to_numpy() > 0,
                np.maximum(1, np.rint(np.nan_to_num(scaled, nan=0.0))),
                0,
            ).astype(np.int64)
            df.loc[mask, col] = scaled
        else:
            df.loc[mask, col] = np.maximum(0.0, np.nan_to_num(scaled, nan=0.0))

        changed.append(col)

    return changed


def _apply_behavioral_mimicry(
    df: pd.DataFrame,
    mask: pd.Series,
    benign_stats: dict[str, dict[str, float]],
) -> list[str]:
    """
    Shift attack-row feature values toward the benign distribution.

    For each feature x:
        z_attack = (x - mean_attack) / std_attack
        x_benign_like = mean_benign + z_attack * std_benign
        x_new = (1 - alpha) * x + alpha * x_benign_like

    alpha = MIMICRY_STRENGTH_PCT / 100
    """
    alpha = float(MIMICRY_STRENGTH_PCT) / 100.0
    alpha = min(max(alpha, 0.0), 1.0)

    if alpha == 0.0:
        return []

    changed = []

    for col in _mimicry_columns():
        if col not in df.columns or col not in benign_stats:
            continue

        x = pd.to_numeric(df.loc[mask, col], errors="coerce")
        valid = x.notna()

        if valid.sum() == 0:
            continue

        x_valid = x.loc[valid]
        attack_mean = float(x_valid.mean())
        attack_std = _safe_std(x_valid)

        benign_mean = benign_stats[col]["mean"]
        benign_std = benign_stats[col]["std"]

        z_attack = (x_valid - attack_mean) / attack_std
        benign_like = benign_mean + z_attack * benign_std
        shifted = (1.0 - alpha) * x_valid + alpha * benign_like

        if col in NON_NEGATIVE_FEATURES:
            shifted = shifted.clip(lower=0)

        if col in BOUNDED_0_1_FEATURES:
            shifted = shifted.clip(lower=0, upper=1)

        df.loc[x_valid.index, col] = shifted
        changed.append(col)

    return changed


def _transform_attack_file(
    source_path: Path,
    output_path: Path,
    benign_stats: dict[str, dict[str, float]],
) -> FileReport:
    df = _read_source_csv(source_path)
    mask = _attack_row_mask(df)

    transformed_cols: list[str] = []

    if mask.any():
        transformed_cols.extend(_reduce_volume_features(df, mask))
        transformed_cols.extend(_apply_behavioral_mimicry(df, mask, benign_stats))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return FileReport(
        source=str(source_path),
        output=str(output_path),
        kind="attack",
        rows=int(len(df)),
        attack_rows=int(mask.sum()),
        mimicry_strength_pct=float(MIMICRY_STRENGTH_PCT),
        volume_reduction_factor=float(VOLUME_REDUCTION_FACTOR),
        transformed_columns=list(dict.fromkeys(transformed_cols)),
    )


def _copy_or_slice_normal_file(source_path: Path, output_path: Path) -> FileReport:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if DATASET_ROW_LIMIT is None:
        shutil.copy2(source_path, output_path)
        rows = int(sum(1 for _ in open(output_path, "rb")) - 1)
    else:
        df = _read_source_csv(source_path)
        df.to_csv(output_path, index=False)
        rows = int(len(df))

    return FileReport(
        source=str(source_path),
        output=str(output_path),
        kind="normal",
        rows=rows,
        attack_rows=0,
        mimicry_strength_pct=0.0,
        volume_reduction_factor=1.0,
        transformed_columns=[],
    )


def _stealth_attack_filename(path: Path) -> str:
    return f"{path.stem}_{STEALTH_TAG}{path.suffix}"


def generate_stealth_datasets() -> list[FileReport]:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SOURCE_ROOT does not exist: {SOURCE_ROOT}")

    csv_files = _candidate_csv_files(SOURCE_ROOT)
    if not csv_files:
        raise FileNotFoundError(f"No source CSV files found under: {SOURCE_ROOT}")

    print(f"[stealth] Source root : {SOURCE_ROOT}")
    print(f"[stealth] Output root : {OUTPUT_ROOT}")
    print(f"[stealth] CSV files   : {len(csv_files)}")
    print(f"[stealth] Mimicry     : {MIMICRY_STRENGTH_PCT:.1f}%")
    print(f"[stealth] Volume      : /{VOLUME_REDUCTION_FACTOR:g}")

    benign_stats = _build_benign_reference(csv_files)
    print(f"[stealth] Benign reference columns: {sorted(benign_stats)}")

    reports: list[FileReport] = []

    for source_path in csv_files:
        rel = source_path.relative_to(SOURCE_ROOT)
        kind = _infer_kind_from_path(source_path)

        if kind is None:
            # Last-resort inference by labels.
            probe = _read_source_csv(source_path)
            kind = "attack" if _attack_row_mask(probe).any() else "normal"

        if kind == "normal":
            if not COPY_NORMAL_DATASETS:
                continue
            output_path = OUTPUT_ROOT / rel
            report = _copy_or_slice_normal_file(source_path, output_path)

        elif kind == "attack":
            output_path = OUTPUT_ROOT / rel.parent / _stealth_attack_filename(source_path)
            report = _transform_attack_file(source_path, output_path, benign_stats)

        else:
            print(f"[stealth] Skipping unknown file kind: {source_path}")
            continue

        reports.append(report)
        print(
            f"[stealth] {report.kind:6s} rows={report.rows:7d} "
            f"attack_rows={report.attack_rows:5d} -> {Path(report.output).name}"
        )

    manifest = {
        "option": OPTION,
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "rows_per_dataset": ROWS_PER_DATASET,
        "dataset_row_limit": DATASET_ROW_LIMIT,
        "stealth_tag": STEALTH_TAG,
        "mimicry_strength_pct": MIMICRY_STRENGTH_PCT,
        "volume_reduction_factor": VOLUME_REDUCTION_FACTOR,
        "mimic_volume_features": MIMIC_VOLUME_FEATURES,
        "copy_normal_datasets": COPY_NORMAL_DATASETS,
        "benign_reference_stats": benign_stats,
        "files": [asdict(r) for r in reports],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "stealth_generation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\n[stealth] Done. Wrote {len(reports)} file(s).")
    print(f"[stealth] Manifest: {manifest_path}")

    return reports


if __name__ == "__main__":
    generate_stealth_datasets()