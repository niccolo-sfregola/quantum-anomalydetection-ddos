"""Candidate source-IP feature generation for DDoS attack-window attribution.

The feature generator starts from cleaned/preprocessed flow tables produced by the
existing repository. It does not read raw datasets and it does not recreate raw
features that cleaning removed. Every candidate feature is derived from columns
that are present in the cleaned input files.

The output is one row per candidate (window_id, src_ip), with metadata keys,
optional audit-derived labels for supervised training/evaluation, and numeric
candidate features safe for model input. Audit columns are never emitted as
model-input features; downstream selectors in ip_attribution_common exclude them.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # Works both as package import and as direct script execution.
    from .ip_attribution_common import (
        AUDIT_COLUMNS,
        ensure_dir,
        normalize_name,
        normalize_split_value,
        read_table,
        save_json,
        setup_logging,
        write_table,
    )
except ImportError:  # pragma: no cover
    from ip_attribution_common import (
        AUDIT_COLUMNS,
        ensure_dir,
        normalize_name,
        normalize_split_value,
        read_table,
        save_json,
        setup_logging,
        write_table,
    )

LOGGER = logging.getLogger("ip_attribution.features")

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".pq", ".feather", ".ft", ".pkl", ".pickle")

SRC_IP_CANDIDATES = ["src_ip", "source_ip", "srcip", "src_addr", "source_address", "srcaddr"]
DST_IP_CANDIDATES = ["dst_ip", "destination_ip", "dstip", "dst_addr", "destination_address", "dstaddr"]
SRC_PORT_CANDIDATES = ["src_port", "source_port", "sport"]
DST_PORT_CANDIDATES = ["dst_port", "destination_port", "dport"]
PROTOCOL_CANDIDATES = ["protocol", "proto", "ip_protocol"]
WINDOW_CANDIDATES = ["window_id", "window", "window_idx", "window_index", "time_window", "flow_window_id"]
TIMESTAMP_CANDIDATES = ["timestamp", "time", "ts", "start_time", "flow_start_time"]

CANONICAL_NUMERIC_COLUMNS: dict[str, list[str]] = {
    "total_packets": ["total_packets", "packets", "packet_count", "tot_pkts", "num_packets"],
    "total_bytes": ["total_bytes", "bytes", "byte_count", "tot_bytes", "num_bytes"],
    "packets_per_second": ["packets_per_second", "pps", "packets_s", "packet_rate"],
    "bytes_per_second": ["bytes_per_second", "bps", "bytes_s", "byte_rate"],
    "duration": ["duration", "flow_duration", "dur"],
    "packet_size_avg": ["packet_size_avg", "avg_packet_size", "packet_size_mean", "pkt_size_avg"],
    "packet_size_std": ["packet_size_std", "std_packet_size", "packet_size_stddev", "pkt_size_std"],
    "outbound_byte_ratio": ["outbound_byte_ratio", "outbound_ratio", "out_bytes_ratio"],
    "inter_packet_arrival_mean": ["inter_packet_arrival_mean", "iat_mean", "inter_arrival_mean"],
    "inter_packet_arrival_std": ["inter_packet_arrival_std", "iat_std", "inter_arrival_std"],
}

# The challenge labels these as audit fields. They may be used to construct
# targets/metrics, never model-input features.
LABEL_AUDIT_COLUMN = "is_seeded_ddos"


@dataclass
class FeatureConfig:
    cleaned_data_dir: Path
    output_dir: Path
    file_glob: str = "*"
    recursive: bool = True
    output_format: str = "csv"
    rows_per_window: int = 2000
    window_seconds: int | None = None
    src_ip_col: str | None = None
    window_col: str | None = None
    timestamp_col: str | None = None
    sort_by_row_in_window: bool = True
    derive_window_from_row_order: bool = True
    min_flow_count: int = 1
    top_k_by_flow: int | None = None
    top_k_by_packets: int | None = None
    keep_all_candidates: bool = True
    preserve_labeled_positives: bool = False
    protocol_max_categories: int = 10
    include_rank_features: bool = True
    include_numeric_std: bool = True
    dataset_size: str | None = None
    option_name: str | None = None
    agg_feature_set: str | None = None
    stealth: bool | None = None
    mimicry_strength_pct: float | None = None
    random_seed: int = 42
    unavailable_log: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cleaned_data_dir = Path(self.cleaned_data_dir)
        self.output_dir = Path(self.output_dir)


def _casefold_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in columns}


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], preferred: str | None = None) -> str | None:
    if preferred:
        if preferred in df.columns:
            return preferred
        lower_map = _casefold_map(df.columns)
        if preferred.strip().lower() in lower_map:
            return lower_map[preferred.strip().lower()]
    lower_map = _casefold_map(df.columns)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _safe_feature_name(value: Any, max_len: int = 60) -> str:
    text = normalize_name(value)
    text = re.sub(r"_+", "_", text)
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "unknown"


def discover_cleaned_files(config: FeatureConfig) -> list[Path]:
    root = config.cleaned_data_dir
    if not root.exists():
        raise FileNotFoundError(f"Cleaned data directory does not exist: {root}")

    files: list[Path] = []
    glob_pattern = config.file_glob or "*"
    if glob_pattern == "*":
        patterns = [f"*{suffix}" for suffix in SUPPORTED_SUFFIXES]
    else:
        patterns = [glob_pattern]

    for pattern in patterns:
        iterator = root.rglob(pattern) if config.recursive else root.glob(pattern)
        for path in iterator:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)

    # Avoid accidentally re-consuming attribution artifacts if output_dir is under
    # cleaned_data_dir.
    output_dir_resolved = config.output_dir.resolve()
    filtered = []
    for path in sorted(set(files)):
        try:
            if output_dir_resolved in path.resolve().parents:
                continue
        except FileNotFoundError:
            pass
        filtered.append(path)
    return filtered


def _first_constant_value(df: pd.DataFrame, col: str, default: str) -> str:
    if col in df.columns and df[col].notna().any():
        values = df[col].dropna().unique()
        if len(values) == 1:
            return str(values[0])
        # Multiple values in one file are uncommon but possible after concatenation.
        return str(values[0])
    return default


def infer_split(path: Path, df: pd.DataFrame) -> str:
    if "split" in df.columns and df["split"].notna().any():
        return normalize_split_value(_first_constant_value(df, "split", "unknown"))
    text = normalize_name("_".join([path.stem, *[p.name for p in path.parents][:4]]))
    if any(token in text for token in ["training", "train"]):
        return "train"
    if any(token in text for token in ["validation", "valid", "val", "dev"]):
        return "validation"
    if "test" in text:
        return "test"
    return "unknown"


def infer_kind(path: Path, df: pd.DataFrame) -> str:
    for col in ["kind", "scenario"]:
        if col in df.columns and df[col].notna().any():
            value = normalize_name(_first_constant_value(df, col, "unknown"))
            if any(token in value for token in ["attack", "ddos", "dos", "malicious"]):
                return "attack"
            if any(token in value for token in ["normal", "benign", "clean"]):
                return "normal"
            return value
    text = normalize_name("_".join([path.stem, *[p.name for p in path.parents][:5]]))
    if any(token in text for token in ["attack", "ddos", "dos", "malicious"]):
        return "attack"
    if any(token in text for token in ["normal", "benign", "clean"]):
        return "normal"
    return "unknown"


def infer_dataset(path: Path, df: pd.DataFrame) -> tuple[str, str]:
    dataset = path.stem
    dataset_id = dataset
    for col in ["dataset", "dataset_name"]:
        if col in df.columns and df[col].notna().any():
            dataset = _first_constant_value(df, col, dataset)
            break
    if "dataset_id" in df.columns and df["dataset_id"].notna().any():
        dataset_id = _first_constant_value(df, "dataset_id", dataset_id)
    elif "source_dataset" in df.columns and df["source_dataset"].notna().any():
        dataset_id = _first_constant_value(df, "source_dataset", dataset_id)
    return str(dataset), str(dataset_id)


def add_window_id(df: pd.DataFrame, path: Path, config: FeatureConfig) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    window_col = resolve_column(out, WINDOW_CANDIDATES, preferred=config.window_col)
    if window_col is not None:
        out["window_id"] = out[window_col].astype(str)
        return out, f"existing_column:{window_col}"

    ts_col = resolve_column(out, TIMESTAMP_CANDIDATES, preferred=config.timestamp_col)
    if ts_col is not None and config.window_seconds is not None and config.window_seconds > 0:
        ts = pd.to_datetime(out[ts_col], errors="coerce")
        if ts.notna().any():
            origin = ts.dropna().min()
            seconds = (ts - origin).dt.total_seconds().fillna(0)
            out["window_id"] = (seconds // int(config.window_seconds)).astype(int).astype(str)
            return out, f"timestamp_floor:{ts_col}:{config.window_seconds}s"

    if config.derive_window_from_row_order:
        if config.sort_by_row_in_window and "row_in_window" in out.columns:
            out = out.sort_values("row_in_window", kind="mergesort").reset_index(drop=True)
            source = "row_order_sorted_by_row_in_window"
        else:
            out = out.reset_index(drop=True)
            source = "row_order"
        if config.rows_per_window <= 0:
            raise ValueError("rows_per_window must be positive when deriving window_id from row order.")
        out["window_id"] = (np.arange(len(out)) // int(config.rows_per_window)).astype(str)
        return out, f"derived_from_{source}:rows_per_window={config.rows_per_window}"

    raise ValueError(
        f"Could not identify a window column in {path}. Provide --window-col, "
        "--timestamp-col with --window-seconds, or enable row-order derivation."
    )


def canonical_numeric_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical, candidates in CANONICAL_NUMERIC_COLUMNS.items():
        col = resolve_column(df, candidates)
        if col is None:
            continue
        if col in AUDIT_COLUMNS:
            continue
        resolved[canonical] = col
    return resolved


def _top_share(df: pd.DataFrame, group_cols: list[str], value_col: str, denom: pd.Series, out_name: str) -> pd.Series:
    counts = df.groupby(group_cols + [value_col], dropna=False).size()
    if counts.empty:
        return pd.Series(dtype=float, name=out_name)
    max_counts = counts.groupby(level=list(range(len(group_cols)))).max()
    share = max_counts / denom.reindex(max_counts.index).replace(0, np.nan)
    share = share.fillna(0.0)
    share.name = out_name
    return share


def build_candidate_features_for_file(path: Path, config: FeatureConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_table(path)
    file_info: dict[str, Any] = {
        "path": str(path),
        "n_input_rows": int(len(df)),
        "missing_optional_columns": [],
        "used_columns": {},
        "window_source": None,
        "labels_available": False,
    }
    if df.empty:
        LOGGER.warning("Skipping empty cleaned data file: %s", path)
        return pd.DataFrame(), file_info

    src_col = resolve_column(df, SRC_IP_CANDIDATES, preferred=config.src_ip_col)
    if src_col is None:
        LOGGER.warning("Skipping %s because no source-IP column was found in cleaned data.", path)
        file_info["skipped_reason"] = "missing_source_ip_column"
        return pd.DataFrame(), file_info

    df, window_source = add_window_id(df, path, config)
    file_info["window_source"] = window_source

    dataset, dataset_id = infer_dataset(path, df)
    split = infer_split(path, df)
    kind = infer_kind(path, df)

    work = pd.DataFrame(index=df.index)
    work["dataset"] = dataset
    work["dataset_id"] = dataset_id
    work["split"] = split
    work["kind"] = kind
    work["window_id"] = df["window_id"].astype(str)
    work["src_ip"] = df[src_col].astype(str)

    dst_col = resolve_column(df, DST_IP_CANDIDATES)
    dst_port_col = resolve_column(df, DST_PORT_CANDIDATES)
    src_port_col = resolve_column(df, SRC_PORT_CANDIDATES)
    protocol_col = resolve_column(df, PROTOCOL_CANDIDATES)
    numeric_cols = canonical_numeric_columns(df)
    file_info["used_columns"] = {
        "src_ip": src_col,
        "dst_ip": dst_col,
        "dst_port": dst_port_col,
        "src_port": src_port_col,
        "protocol": protocol_col,
        "numeric": numeric_cols,
    }

    for required_name, resolved in [
        ("dst_ip", dst_col),
        ("dst_port", dst_port_col),
        ("protocol", protocol_col),
    ]:
        if resolved is None:
            file_info["missing_optional_columns"].append(required_name)

    for canonical in CANONICAL_NUMERIC_COLUMNS:
        if canonical not in numeric_cols:
            file_info["missing_optional_columns"].append(canonical)

    window_keys = ["dataset", "dataset_id", "split", "kind", "window_id"]
    candidate_keys = window_keys + ["src_ip"]

    # Base counts.
    flow_count = work.groupby(candidate_keys, dropna=False).size().rename("flow_count")
    cand = flow_count.reset_index()
    window_flow_count = work.groupby(window_keys, dropna=False).size().rename("window_flow_count")
    cand = cand.merge(window_flow_count.reset_index(), on=window_keys, how="left")
    cand["flow_fraction"] = cand["flow_count"] / cand["window_flow_count"].replace(0, np.nan)
    cand["flow_fraction"] = cand["flow_fraction"].fillna(0.0)

    # Optional audit-derived target labels. These are retained only for supervised
    # training/evaluation and metrics; downstream model feature selectors exclude them.
    if LABEL_AUDIT_COLUMN in df.columns:
        seeded = pd.to_numeric(df[LABEL_AUDIT_COLUMN], errors="coerce").fillna(0).astype(int)
        work["__seeded"] = (seeded > 0).astype(int)
        label = work.groupby(candidate_keys, dropna=False)["__seeded"].max().rename("true_malicious")
        win_label = work.groupby(window_keys, dropna=False)["__seeded"].max().rename("window_has_malicious")
        cand = cand.merge(label.reset_index(), on=candidate_keys, how="left")
        cand = cand.merge(win_label.reset_index(), on=window_keys, how="left")
        cand["true_malicious"] = cand["true_malicious"].fillna(0).astype(int)
        cand["window_has_malicious"] = cand["window_has_malicious"].fillna(0).astype(int)
        cand["label_available"] = 1
        file_info["labels_available"] = True
    else:
        cand["label_available"] = 0
        file_info["labels_available"] = False
        file_info["missing_optional_columns"].append(LABEL_AUDIT_COLUMN)

    # Destination diversity and concentration.
    if dst_col is not None:
        work["__dst_ip"] = df[dst_col].astype(str)
        dst_unique = work.groupby(candidate_keys, dropna=False)["__dst_ip"].nunique(dropna=True).rename("unique_dst_ip_count")
        cand = cand.merge(dst_unique.reset_index(), on=candidate_keys, how="left")
        dst_top_share = _top_share(work, candidate_keys, "__dst_ip", flow_count, "top_dst_ip_share")
        cand = cand.merge(dst_top_share.reset_index(), on=candidate_keys, how="left")
        # Window baseline ratios.
        win_unique = work.groupby(window_keys, dropna=False)["__dst_ip"].nunique(dropna=True).rename("window_unique_dst_ip_count")
        cand = cand.merge(win_unique.reset_index(), on=window_keys, how="left")
        cand["unique_dst_ip_ratio_to_window"] = cand["unique_dst_ip_count"] / cand["window_unique_dst_ip_count"].replace(0, np.nan)

    if dst_port_col is not None:
        work["__dst_port"] = df[dst_port_col].astype(str)
        dst_port_unique = work.groupby(candidate_keys, dropna=False)["__dst_port"].nunique(dropna=True).rename("unique_dst_port_count")
        cand = cand.merge(dst_port_unique.reset_index(), on=candidate_keys, how="left")
        port_top_share = _top_share(work, candidate_keys, "__dst_port", flow_count, "top_dst_port_share")
        cand = cand.merge(port_top_share.reset_index(), on=candidate_keys, how="left")
        win_port_unique = work.groupby(window_keys, dropna=False)["__dst_port"].nunique(dropna=True).rename("window_unique_dst_port_count")
        cand = cand.merge(win_port_unique.reset_index(), on=window_keys, how="left")
        cand["unique_dst_port_ratio_to_window"] = cand["unique_dst_port_count"] / cand["window_unique_dst_port_count"].replace(0, np.nan)

    if src_port_col is not None:
        work["__src_port"] = df[src_port_col].astype(str)
        src_port_unique = work.groupby(candidate_keys, dropna=False)["__src_port"].nunique(dropna=True).rename("unique_src_port_count")
        cand = cand.merge(src_port_unique.reset_index(), on=candidate_keys, how="left")

    # Protocol distribution. This gracefully skips if the cleaned data no longer
    # contains a protocol column.
    if protocol_col is not None:
        protocol_values = df[protocol_col].astype(str).map(_safe_feature_name)
        top_protocols = protocol_values.value_counts().head(config.protocol_max_categories).index.tolist()
        work["__protocol"] = protocol_values.where(protocol_values.isin(top_protocols), other="other")
        proto_counts = work.groupby(candidate_keys + ["__protocol"], dropna=False).size().rename("count").reset_index()
        proto_wide = proto_counts.pivot_table(
            index=candidate_keys,
            columns="__protocol",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        proto_wide.columns = [f"protocol_share_{_safe_feature_name(c)}" for c in proto_wide.columns]
        proto_wide = proto_wide.div(flow_count.reindex(proto_wide.index).replace(0, np.nan), axis=0).fillna(0.0)
        cand = cand.merge(proto_wide.reset_index(), on=candidate_keys, how="left")

    # Numeric aggregations and ratios to whole-window baselines.
    for canonical, col in numeric_cols.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        work[f"__num_{canonical}"] = numeric
        ops = ["mean"]
        if canonical in {"total_packets", "total_bytes"}:
            ops.append("sum")
        if config.include_numeric_std:
            ops.append("std")
        grouped = work.groupby(candidate_keys, dropna=False)[f"__num_{canonical}"].agg(ops)
        grouped = grouped.rename(columns={op: f"{canonical}_{op}" for op in ops})
        cand = cand.merge(grouped.reset_index(), on=candidate_keys, how="left")

        win_ops = ["mean", "sum"]
        win_grouped = work.groupby(window_keys, dropna=False)[f"__num_{canonical}"].agg(win_ops)
        win_grouped = win_grouped.rename(
            columns={"mean": f"window_{canonical}_mean", "sum": f"window_{canonical}_sum"}
        )
        cand = cand.merge(win_grouped.reset_index(), on=window_keys, how="left")

        if f"{canonical}_mean" in cand.columns:
            denom = cand[f"window_{canonical}_mean"].replace(0, np.nan)
            cand[f"{canonical}_mean_ratio_to_window"] = (cand[f"{canonical}_mean"] / denom).replace([np.inf, -np.inf], np.nan)
        if f"{canonical}_sum" in cand.columns:
            denom = cand[f"window_{canonical}_sum"].replace(0, np.nan)
            cand[f"{canonical}_sum_fraction_of_window"] = (cand[f"{canonical}_sum"] / denom).replace([np.inf, -np.inf], np.nan)

    # Within-window ranks are useful for source-IP attribution because the task is
    # naturally a window-conditioned ranking problem. Ranks are derived only from
    # candidate features, not labels.
    if config.include_rank_features:
        rank_cols = ["flow_count"]
        for maybe_col in ["total_packets_sum", "total_bytes_sum", "packets_per_second_mean", "bytes_per_second_mean"]:
            if maybe_col in cand.columns:
                rank_cols.append(maybe_col)
        for col in rank_cols:
            cand[f"{col}_rank_desc"] = cand.groupby(window_keys, dropna=False)[col].rank(method="dense", ascending=False)
            cand[f"{col}_rank_pct"] = cand.groupby(window_keys, dropna=False)[col].rank(method="average", ascending=True, pct=True)

    cand = cand.replace([np.inf, -np.inf], np.nan)
    # Fill concentration/diversity missing values introduced by unavailable data.
    for col in cand.columns:
        if col not in candidate_keys and pd.api.types.is_numeric_dtype(cand[col]):
            cand[col] = cand[col].fillna(0.0)

    cand = prune_candidates(cand, config, window_keys=window_keys)
    file_info["n_candidates"] = int(len(cand))
    file_info["n_windows"] = int(cand["window_id"].nunique()) if "window_id" in cand.columns else 0
    return cand, file_info


def prune_candidates(cand: pd.DataFrame, config: FeatureConfig, window_keys: list[str]) -> pd.DataFrame:
    if cand.empty:
        return cand
    keep = pd.Series(True, index=cand.index)
    if config.min_flow_count and config.min_flow_count > 1:
        keep &= cand["flow_count"] >= int(config.min_flow_count)

    if not config.keep_all_candidates:
        top_masks: list[pd.Series] = []
        if config.top_k_by_flow and config.top_k_by_flow > 0:
            rank = cand.groupby(window_keys, dropna=False)["flow_count"].rank(method="first", ascending=False)
            top_masks.append(rank <= int(config.top_k_by_flow))
        if config.top_k_by_packets and config.top_k_by_packets > 0 and "total_packets_sum" in cand.columns:
            rank = cand.groupby(window_keys, dropna=False)["total_packets_sum"].rank(method="first", ascending=False)
            top_masks.append(rank <= int(config.top_k_by_packets))
        if top_masks:
            top_keep = top_masks[0].copy()
            for mask in top_masks[1:]:
                top_keep |= mask
            keep &= top_keep

    cand = cand.copy()
    cand["preserved_by_label_for_eval"] = 0
    if config.preserve_labeled_positives and "true_malicious" in cand.columns:
        positive = pd.to_numeric(cand["true_malicious"], errors="coerce").fillna(0).astype(int) == 1
        cand.loc[positive & ~keep, "preserved_by_label_for_eval"] = 1
        keep |= positive

    pruned = cand.loc[keep].reset_index(drop=True)
    return pruned


def generate_candidate_features(config: FeatureConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dir(config.output_dir)
    files = discover_cleaned_files(config)
    if not files:
        raise FileNotFoundError(f"No cleaned data files found under {config.cleaned_data_dir}")

    LOGGER.info("Discovered %d cleaned/preprocessed files under %s", len(files), config.cleaned_data_dir)
    frames: list[pd.DataFrame] = []
    file_infos: list[dict[str, Any]] = []
    for path in files:
        try:
            cand, info = build_candidate_features_for_file(path, config)
            file_infos.append(info)
            if not cand.empty:
                frames.append(cand)
            LOGGER.info("Processed %s -> %d candidates", path, len(cand))
        except Exception as exc:  # Keep the runner useful when one reduced split/file is missing a column.
            LOGGER.exception("Failed to process %s: %s", path, exc)
            file_infos.append({"path": str(path), "skipped_reason": repr(exc)})

    if frames:
        all_candidates = pd.concat(frames, ignore_index=True, sort=False)
    else:
        all_candidates = pd.DataFrame()

    if not all_candidates.empty:
        sort_cols = [c for c in ["dataset", "split", "kind", "window_id", "flow_count", "src_ip"] if c in all_candidates.columns]
        ascending = [True] * len(sort_cols)
        if "flow_count" in sort_cols:
            ascending[sort_cols.index("flow_count")] = False
        all_candidates = all_candidates.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    suffix = ".csv" if config.output_format.lower() == "csv" else f".{config.output_format.lower().lstrip('.')}"
    all_path = config.output_dir / f"candidate_ip_features_all{suffix}"
    write_table(all_candidates, all_path)

    split_paths: dict[str, str] = {}
    if "split" in all_candidates.columns:
        for split_value, split_df in all_candidates.groupby("split", dropna=False):
            safe_split = normalize_split_value(split_value)
            split_path = config.output_dir / f"candidate_ip_features_{safe_split}{suffix}"
            write_table(split_df.reset_index(drop=True), split_path)
            split_paths[str(safe_split)] = str(split_path)

    summary = {
        "config": asdict(config),
        "n_files_discovered": len(files),
        "n_files_with_candidates": len(frames),
        "n_candidates": int(len(all_candidates)),
        "n_windows": int(all_candidates["window_id"].nunique()) if "window_id" in all_candidates.columns else 0,
        "labels_available_any": bool("true_malicious" in all_candidates.columns and all_candidates["true_malicious"].notna().any()),
        "all_candidates_path": str(all_path),
        "split_paths": split_paths,
        "file_infos": file_infos,
    }
    save_json(summary, config.output_dir / "candidate_feature_summary.json")
    LOGGER.info("Saved candidate features to %s", all_path)
    return all_candidates, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate candidate source-IP features from cleaned/preprocessed data.")
    parser.add_argument("--cleaned-data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--file-glob", default="*")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--output-format", default="csv", choices=["csv", "parquet", "feather", "pkl"])
    parser.add_argument("--rows-per-window", type=int, default=2000)
    parser.add_argument("--window-seconds", type=int, default=None)
    parser.add_argument("--src-ip-col", default=None)
    parser.add_argument("--window-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--no-sort-by-row-in-window", action="store_true")
    parser.add_argument("--no-row-order-windowing", action="store_true")
    parser.add_argument("--min-flow-count", type=int, default=1)
    parser.add_argument("--top-k-by-flow", type=int, default=None)
    parser.add_argument("--top-k-by-packets", type=int, default=None)
    parser.add_argument("--keep-all-candidates", action="store_true", default=False)
    parser.add_argument("--prune-candidates", action="store_true", help="Use top-k/min-count pruning instead of keeping all candidates.")
    parser.add_argument("--preserve-labeled-positives", action="store_true")
    parser.add_argument("--protocol-max-categories", type=int, default=10)
    parser.add_argument("--no-rank-features", action="store_true")
    parser.add_argument("--no-numeric-std", action="store_true")
    parser.add_argument("--dataset-size", default=None)
    parser.add_argument("--option-name", default=None)
    parser.add_argument("--agg-feature-set", default=None)
    parser.add_argument("--stealth", action="store_true")
    parser.add_argument("--mimicry-strength-pct", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> FeatureConfig:
    keep_all = True
    if args.prune_candidates:
        keep_all = False
    if args.keep_all_candidates:
        keep_all = True
    return FeatureConfig(
        cleaned_data_dir=args.cleaned_data_dir,
        output_dir=args.output_dir,
        file_glob=args.file_glob,
        recursive=not args.no_recursive,
        output_format=args.output_format,
        rows_per_window=args.rows_per_window,
        window_seconds=args.window_seconds,
        src_ip_col=args.src_ip_col,
        window_col=args.window_col,
        timestamp_col=args.timestamp_col,
        sort_by_row_in_window=not args.no_sort_by_row_in_window,
        derive_window_from_row_order=not args.no_row_order_windowing,
        min_flow_count=args.min_flow_count,
        top_k_by_flow=args.top_k_by_flow,
        top_k_by_packets=args.top_k_by_packets,
        keep_all_candidates=keep_all,
        preserve_labeled_positives=args.preserve_labeled_positives,
        protocol_max_categories=args.protocol_max_categories,
        include_rank_features=not args.no_rank_features,
        include_numeric_std=not args.no_numeric_std,
        dataset_size=args.dataset_size,
        option_name=args.option_name,
        agg_feature_set=args.agg_feature_set,
        stealth=args.stealth,
        mimicry_strength_pct=args.mimicry_strength_pct,
        random_seed=args.random_seed,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args)
    generate_candidate_features(config)


if __name__ == "__main__":
    main()
