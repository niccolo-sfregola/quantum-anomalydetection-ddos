"""End-to-end malicious source-IP attribution runner.

This orchestrator preserves the existing repository's window-detection workflow by
adding a separate source-IP attribution branch that starts from cleaned or
preprocessed flow tables. It can run feature generation only, quantum enrichment
only, supervised attribution only, unsupervised attribution only, or both methods
with classical-only, quantum-only, and combined feature modes.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .ip_attribution_common import ensure_dir, parse_top_ks, save_json, setup_logging
    from .ip_attribution_features import FeatureConfig, generate_candidate_features
    from .ip_quantum_enrichment import QuantumReservoirConfig, enrich_candidate_features
    from .ip_attribution_supervised import SupervisedConfig, run_supervised_attribution
    from .ip_attribution_unsupervised import UnsupervisedConfig, run_unsupervised_attribution
except ImportError:  # pragma: no cover
    from ip_attribution_common import ensure_dir, parse_top_ks, save_json, setup_logging
    from ip_attribution_features import FeatureConfig, generate_candidate_features
    from ip_quantum_enrichment import QuantumReservoirConfig, enrich_candidate_features
    from ip_attribution_supervised import SupervisedConfig, run_supervised_attribution
    from ip_attribution_unsupervised import UnsupervisedConfig, run_unsupervised_attribution

LOGGER = logging.getLogger("ip_attribution.pipeline")


@dataclass
class PipelineConfig:
    cleaned_data_dir: Path
    output_dir: Path = Path(".")
    dataset_size: str = "FULL"
    run_tag: str = ""
    option_name: str = "option_2"
    agg_feature_set: str = "full"
    mode: str = "both"
    feature_modes: tuple[str, ...] = ("combined",)
    candidate_features_path: Path | None = None
    enriched_features_path: Path | None = None
    skip_feature_generation: bool = False
    skip_enrichment: bool = False
    file_glob: str = "*"
    recursive: bool = True
    output_format: str = "csv"
    rows_per_window: int = 2000
    window_seconds: int | None = None
    src_ip_col: str | None = None
    window_col: str | None = None
    timestamp_col: str | None = None
    min_flow_count: int = 1
    top_k_by_flow: int | None = None
    top_k_by_packets: int | None = None
    keep_all_candidates: bool = True
    preserve_labeled_positives: bool = False
    protocol_max_categories: int = 10
    include_rank_features: bool = True
    include_numeric_std: bool = True
    stealth: bool = False
    mimicry_strength_pct: float | None = None
    random_seed: int = 42
    n_qubits: int = 6
    n_layers: int = 2
    max_input_features: int = 24
    quantum_clip_value: float = 3.0
    quantum_input_scale: float = 1.0
    quantum_weight_scale: float = 0.75
    quantum_shot_count: int = 0
    quantum_baseline_scope: str = "normal_train"
    include_pairwise_zz: bool = True
    include_entropy: bool = True
    include_entanglement_entropy: bool = False
    supervised_model_type: str = "logistic"
    supervised_threshold_strategy: str = "f1"
    unsupervised_model_type: str = "isolation_forest"
    unsupervised_fit_scope: str = "normal_train"
    unsupervised_threshold_strategy: str = "quantile"
    contamination: float = 0.02
    default_supervised_threshold: float = 0.5
    default_unsupervised_threshold: float = 0.0
    top_ks: tuple[int, ...] = (1, 5, 10, 20)

    def __post_init__(self) -> None:
        self.cleaned_data_dir = Path(self.cleaned_data_dir)
        self.output_dir = Path(self.output_dir)
        if self.candidate_features_path is not None:
            self.candidate_features_path = Path(self.candidate_features_path)
        if self.enriched_features_path is not None:
            self.enriched_features_path = Path(self.enriched_features_path)
        self.feature_modes = tuple(self.feature_modes)
        self.top_ks = tuple(int(k) for k in self.top_ks)


def build_output_base(config: PipelineConfig) -> Path:
    tag = config.run_tag or ""
    if tag and not tag.startswith("_"):
        tag = f"_{tag}"
    return (
        Path(config.output_dir)
        / f"outputs_{config.dataset_size}{tag}"
        / config.option_name
        / config.agg_feature_set
        / "ip_attribution"
    )


def _suffix(output_format: str) -> str:
    return ".csv" if output_format.lower() == "csv" else f".{output_format.lower().lstrip('.')}"


def _feature_modes_from_arg(value: str) -> tuple[str, ...]:
    value = value.strip().lower()
    if value == "all":
        return ("classical", "quantum", "combined")
    parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    valid = {"classical", "quantum", "combined"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise ValueError(f"Unsupported feature mode(s): {bad}. Use classical, quantum, combined, or all.")
    return parts or ("combined",)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_row(summary: dict[str, Any], metrics_path: Path, split: str) -> dict[str, Any] | None:
    metrics = _load_json(metrics_path)
    if not metrics:
        return None
    if split == "all":
        candidate = metrics.get("candidate_metrics", {})
        ranking = metrics.get("ranking_metrics", {})
    else:
        split_metrics = metrics.get("by_split", {}).get(split, {})
        candidate = split_metrics.get("candidate_metrics", {})
        ranking = split_metrics.get("ranking_metrics", {})
    if not candidate and not ranking:
        return None
    row: dict[str, Any] = {
        "method": summary.get("method"),
        "feature_mode": summary.get("feature_mode"),
        "split": split,
        "status": summary.get("status"),
        "threshold": summary.get("threshold"),
        "n_features": summary.get("n_features"),
    }
    for key in [
        "n_candidates",
        "support_positive",
        "support_negative",
        "predicted_positive",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
    ]:
        if key in candidate:
            row[key] = candidate[key]
    for key, value in ranking.items():
        if key.startswith("recall_at_") or key in {
            "mean_average_precision_over_windows",
            "mean_false_positives_per_attack_window",
            "mean_predicted_malicious_ips_per_attack_window",
            "mean_true_malicious_ips_per_attack_window",
            "attack_windows_with_positive_ips",
        }:
            row[key] = value
    return row


def write_comparison_summary(run_summaries: list[dict[str, Any]], comparison_dir: Path) -> dict[str, Any]:
    comparison_dir = ensure_dir(comparison_dir)
    rows: list[dict[str, Any]] = []
    for summary in run_summaries:
        if summary.get("status") != "ok":
            rows.append(
                {
                    "method": summary.get("method"),
                    "feature_mode": summary.get("feature_mode"),
                    "split": "all",
                    "status": summary.get("status"),
                    "reason": summary.get("reason"),
                }
            )
            continue
        metrics_path = Path(summary.get("metrics_path", ""))
        for split in ["validation", "test", "all"]:
            row = _extract_metric_row(summary, metrics_path, split)
            if row is not None:
                rows.append(row)
    comparison_df = pd.DataFrame(rows)
    csv_path = comparison_dir / "supervised_vs_unsupervised_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    save_json({"rows": rows, "comparison_csv": str(csv_path)}, comparison_dir / "comparison_summary.json")

    if not comparison_df.empty and "f1" in comparison_df.columns:
        plot_df = comparison_df[(comparison_df["split"].isin(["validation", "test"])) & comparison_df["f1"].notna()].copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["method"].astype(str) + "\n" + plot_df["feature_mode"].astype(str) + "\n" + plot_df["split"].astype(str)
            plt.figure(figsize=(max(7, 0.65 * len(plot_df)), 4.8))
            plt.bar(range(len(plot_df)), plot_df["f1"].astype(float))
            plt.xticks(range(len(plot_df)), plot_df["label"], rotation=45, ha="right")
            plt.ylabel("Candidate-level F1")
            plt.title("Supervised vs unsupervised attribution comparison")
            plt.tight_layout()
            plt.savefig(comparison_dir / "comparison_f1.png", dpi=160)
            plt.close()

    if not comparison_df.empty:
        recall_cols = [c for c in comparison_df.columns if c.startswith("recall_at_")]
        test_df = comparison_df[comparison_df["split"] == "test"].copy()
        if recall_cols and not test_df.empty:
            plt.figure(figsize=(7, 4.8))
            for _, row in test_df.iterrows():
                xs = []
                ys = []
                for col in sorted(recall_cols, key=lambda x: int(x.split("_")[-1])):
                    if pd.notna(row.get(col)):
                        xs.append(int(col.split("_")[-1]))
                        ys.append(float(row[col]))
                if xs:
                    label = f"{row.get('method')} / {row.get('feature_mode')}"
                    plt.plot(xs, ys, marker="o", label=label)
            plt.xlabel("Top-k candidates per attack window")
            plt.ylabel("Mean test recall@k")
            plt.title("Test top-k source-IP recovery comparison")
            plt.ylim(0, 1.02)
            plt.legend()
            plt.tight_layout()
            plt.savefig(comparison_dir / "comparison_test_topk_recovery.png", dpi=160)
            plt.close()

    return {"comparison_csv": str(csv_path), "n_rows": int(len(comparison_df))}


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    base_dir = ensure_dir(build_output_base(config))
    features_dir = ensure_dir(base_dir / "features")
    enriched_dir = ensure_dir(base_dir / "enriched")
    comparison_dir = ensure_dir(base_dir / "comparison")
    save_json(asdict(config), base_dir / "pipeline_config.json")

    mode = config.mode.lower()
    if mode not in {"features", "enrich", "supervised", "unsupervised", "both"}:
        raise ValueError("mode must be one of: features, enrich, supervised, unsupervised, both")

    # 1) Candidate source-IP aggregation.
    if config.skip_feature_generation and config.candidate_features_path is not None:
        candidate_path = config.candidate_features_path
        feature_summary = {"status": "reused", "all_candidates_path": str(candidate_path)}
        LOGGER.info("Reusing candidate feature table: %s", candidate_path)
    elif config.candidate_features_path is not None and config.candidate_features_path.exists():
        candidate_path = config.candidate_features_path
        feature_summary = {"status": "reused", "all_candidates_path": str(candidate_path)}
        LOGGER.info("Using provided candidate feature table: %s", candidate_path)
    else:
        feature_config = FeatureConfig(
            cleaned_data_dir=config.cleaned_data_dir,
            output_dir=features_dir,
            file_glob=config.file_glob,
            recursive=config.recursive,
            output_format=config.output_format,
            rows_per_window=config.rows_per_window,
            window_seconds=config.window_seconds,
            src_ip_col=config.src_ip_col,
            window_col=config.window_col,
            timestamp_col=config.timestamp_col,
            min_flow_count=config.min_flow_count,
            top_k_by_flow=config.top_k_by_flow,
            top_k_by_packets=config.top_k_by_packets,
            keep_all_candidates=config.keep_all_candidates,
            preserve_labeled_positives=config.preserve_labeled_positives,
            protocol_max_categories=config.protocol_max_categories,
            include_rank_features=config.include_rank_features,
            include_numeric_std=config.include_numeric_std,
            dataset_size=config.dataset_size,
            option_name=config.option_name,
            agg_feature_set=config.agg_feature_set,
            stealth=config.stealth,
            mimicry_strength_pct=config.mimicry_strength_pct,
            random_seed=config.random_seed,
        )
        _, feature_summary = generate_candidate_features(feature_config)
        candidate_path = Path(feature_summary["all_candidates_path"])

    if mode == "features":
        pipeline_summary = {
            "status": "ok",
            "base_dir": str(base_dir),
            "candidate_features_path": str(candidate_path),
            "feature_summary": feature_summary,
        }
        save_json(pipeline_summary, base_dir / "pipeline_summary.json")
        return pipeline_summary

    needs_quantum = any(mode_name in {"quantum", "combined"} for mode_name in config.feature_modes) or mode == "enrich"
    if config.skip_enrichment and config.enriched_features_path is not None:
        enriched_path = config.enriched_features_path
        quantum_summary = {"status": "reused", "enriched_path": str(enriched_path)}
        LOGGER.info("Reusing enriched feature table: %s", enriched_path)
    elif config.enriched_features_path is not None and config.enriched_features_path.exists():
        enriched_path = config.enriched_features_path
        quantum_summary = {"status": "reused", "enriched_path": str(enriched_path)}
        LOGGER.info("Using provided enriched feature table: %s", enriched_path)
    elif needs_quantum:
        quantum_config = QuantumReservoirConfig(
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            max_input_features=config.max_input_features,
            seed=config.random_seed,
            clip_value=config.quantum_clip_value,
            input_scale=config.quantum_input_scale,
            random_weight_scale=config.quantum_weight_scale,
            include_pairwise_zz=config.include_pairwise_zz,
            include_entropy=config.include_entropy,
            include_entanglement_entropy=config.include_entanglement_entropy,
            shot_count=config.quantum_shot_count,
            baseline_scope=config.quantum_baseline_scope,
            output_format=config.output_format,
        )
        _, quantum_summary, _ = enrich_candidate_features(candidate_path, enriched_dir, quantum_config)
        enriched_path = Path(quantum_summary["enriched_path"])
    else:
        enriched_path = None
        quantum_summary = {"status": "not_needed"}

    if mode == "enrich":
        pipeline_summary = {
            "status": "ok",
            "base_dir": str(base_dir),
            "candidate_features_path": str(candidate_path),
            "enriched_features_path": str(enriched_path) if enriched_path else None,
            "feature_summary": feature_summary,
            "quantum_summary": quantum_summary,
        }
        save_json(pipeline_summary, base_dir / "pipeline_summary.json")
        return pipeline_summary

    run_summaries: list[dict[str, Any]] = []
    for feature_mode in config.feature_modes:
        if feature_mode == "classical":
            features_path = candidate_path
        else:
            if enriched_path is None:
                raise RuntimeError(f"Feature mode {feature_mode} requires quantum enrichment, but no enriched path is available.")
            features_path = enriched_path

        if mode in {"supervised", "both"}:
            supervised_dir = ensure_dir(base_dir / "supervised" / feature_mode)
            supervised_config = SupervisedConfig(
                features_path=features_path,
                output_dir=supervised_dir,
                feature_mode=feature_mode,
                model_type=config.supervised_model_type,
                threshold_strategy=config.supervised_threshold_strategy,
                default_threshold=config.default_supervised_threshold,
                contamination=config.contamination,
                random_seed=config.random_seed,
                top_ks=config.top_ks,
                output_format=config.output_format,
            )
            run_summaries.append(run_supervised_attribution(supervised_config))

        if mode in {"unsupervised", "both"}:
            unsupervised_dir = ensure_dir(base_dir / "unsupervised" / feature_mode)
            unsupervised_config = UnsupervisedConfig(
                features_path=features_path,
                output_dir=unsupervised_dir,
                feature_mode=feature_mode,
                model_type=config.unsupervised_model_type,
                fit_scope=config.unsupervised_fit_scope,
                threshold_strategy=config.unsupervised_threshold_strategy,
                default_threshold=config.default_unsupervised_threshold,
                contamination=config.contamination,
                random_seed=config.random_seed,
                top_ks=config.top_ks,
                output_format=config.output_format,
            )
            run_summaries.append(run_unsupervised_attribution(unsupervised_config))

    comparison_summary = write_comparison_summary(run_summaries, comparison_dir)
    pipeline_summary = {
        "status": "ok",
        "base_dir": str(base_dir),
        "candidate_features_path": str(candidate_path),
        "enriched_features_path": str(enriched_path) if enriched_path else None,
        "feature_summary": feature_summary,
        "quantum_summary": quantum_summary,
        "runs": run_summaries,
        "comparison_summary": comparison_summary,
    }
    save_json(pipeline_summary, base_dir / "pipeline_summary.json")
    LOGGER.info("IP attribution pipeline complete: %s", base_dir)
    return pipeline_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the malicious source-IP attribution extension.")
    parser.add_argument("--cleaned-data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path, help="Output root. Final artifacts go under outputs_<SIZE><TAG>/<OPTION>/<AGG>/ip_attribution/.")
    parser.add_argument("--dataset-size", default="FULL")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--option-name", default="option_2")
    parser.add_argument("--agg-feature-set", default="full")
    parser.add_argument("--mode", default="both", choices=["features", "enrich", "supervised", "unsupervised", "both"])
    parser.add_argument("--feature-mode", default="combined", help="classical, quantum, combined, or all; comma-separated values allowed.")
    parser.add_argument("--candidate-features-path", type=Path, default=None)
    parser.add_argument("--enriched-features-path", type=Path, default=None)
    parser.add_argument("--skip-feature-generation", action="store_true")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--file-glob", default="*")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--output-format", default="csv", choices=["csv", "parquet", "feather", "pkl"])
    parser.add_argument("--rows-per-window", type=int, default=2000)
    parser.add_argument("--window-seconds", type=int, default=None)
    parser.add_argument("--src-ip-col", default=None)
    parser.add_argument("--window-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--min-flow-count", type=int, default=1)
    parser.add_argument("--top-k-by-flow", type=int, default=None)
    parser.add_argument("--top-k-by-packets", type=int, default=None)
    parser.add_argument("--prune-candidates", action="store_true", help="Enable top-k/min-count pruning. Default keeps all candidates except min-flow-count filters.")
    parser.add_argument("--preserve-labeled-positives", action="store_true", help="Training/evaluation convenience only: retain audit-labeled positives after pruning.")
    parser.add_argument("--protocol-max-categories", type=int, default=10)
    parser.add_argument("--no-rank-features", action="store_true")
    parser.add_argument("--no-numeric-std", action="store_true")
    parser.add_argument("--stealth", action="store_true")
    parser.add_argument("--mimicry-strength-pct", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-input-features", type=int, default=24)
    parser.add_argument("--quantum-clip-value", type=float, default=3.0)
    parser.add_argument("--quantum-input-scale", type=float, default=1.0)
    parser.add_argument("--quantum-weight-scale", type=float, default=0.75)
    parser.add_argument("--quantum-shot-count", type=int, default=0)
    parser.add_argument("--quantum-baseline-scope", default="normal_train", choices=["normal_train", "benign_train", "train", "all_train", "all"])
    parser.add_argument("--no-pairwise-zz", action="store_true")
    parser.add_argument("--no-entropy", action="store_true")
    parser.add_argument("--include-entanglement-entropy", action="store_true")
    parser.add_argument("--supervised-model-type", default="logistic", choices=["logistic", "random_forest", "hist_gradient_boosting"])
    parser.add_argument("--supervised-threshold-strategy", default="f1", choices=["f1", "precision_recall_balance", "quantile", "fixed"])
    parser.add_argument("--unsupervised-model-type", default="isolation_forest", choices=["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"])
    parser.add_argument("--unsupervised-fit-scope", default="normal_train", choices=["normal_train", "train_all", "non_attack_windows", "label_benign_train"])
    parser.add_argument("--unsupervised-threshold-strategy", default="quantile", choices=["quantile", "fixed", "f1", "precision_recall_balance"])
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--default-supervised-threshold", type=float, default=0.5)
    parser.add_argument("--default-unsupervised-threshold", type=float, default=0.0)
    parser.add_argument("--top-ks", default="1,5,10,20")
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        cleaned_data_dir=args.cleaned_data_dir,
        output_dir=args.output_dir,
        dataset_size=args.dataset_size,
        run_tag=args.run_tag,
        option_name=args.option_name,
        agg_feature_set=args.agg_feature_set,
        mode=args.mode,
        feature_modes=_feature_modes_from_arg(args.feature_mode),
        candidate_features_path=args.candidate_features_path,
        enriched_features_path=args.enriched_features_path,
        skip_feature_generation=args.skip_feature_generation,
        skip_enrichment=args.skip_enrichment,
        file_glob=args.file_glob,
        recursive=not args.no_recursive,
        output_format=args.output_format,
        rows_per_window=args.rows_per_window,
        window_seconds=args.window_seconds,
        src_ip_col=args.src_ip_col,
        window_col=args.window_col,
        timestamp_col=args.timestamp_col,
        min_flow_count=args.min_flow_count,
        top_k_by_flow=args.top_k_by_flow,
        top_k_by_packets=args.top_k_by_packets,
        keep_all_candidates=not args.prune_candidates,
        preserve_labeled_positives=args.preserve_labeled_positives,
        protocol_max_categories=args.protocol_max_categories,
        include_rank_features=not args.no_rank_features,
        include_numeric_std=not args.no_numeric_std,
        stealth=args.stealth,
        mimicry_strength_pct=args.mimicry_strength_pct,
        random_seed=args.random_seed,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_input_features=args.max_input_features,
        quantum_clip_value=args.quantum_clip_value,
        quantum_input_scale=args.quantum_input_scale,
        quantum_weight_scale=args.quantum_weight_scale,
        quantum_shot_count=args.quantum_shot_count,
        quantum_baseline_scope=args.quantum_baseline_scope,
        include_pairwise_zz=not args.no_pairwise_zz,
        include_entropy=not args.no_entropy,
        include_entanglement_entropy=args.include_entanglement_entropy,
        supervised_model_type=args.supervised_model_type,
        supervised_threshold_strategy=args.supervised_threshold_strategy,
        unsupervised_model_type=args.unsupervised_model_type,
        unsupervised_fit_scope=args.unsupervised_fit_scope,
        unsupervised_threshold_strategy=args.unsupervised_threshold_strategy,
        contamination=args.contamination,
        default_supervised_threshold=args.default_supervised_threshold,
        default_unsupervised_threshold=args.default_unsupervised_threshold,
        top_ks=parse_top_ks(args.top_ks),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
