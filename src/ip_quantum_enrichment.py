"""Quantum-inspired candidate source-IP feature enrichment.

This module implements a fixed-parameter quantum reservoir feature map for
candidate (window_id, src_ip) rows. It uses statevector simulation by default and
therefore has no dependency on Qiskit, PennyLane, or external backends. The
reservoir is intentionally non-trainable: fixed random RY/RZ rotations encode
normalized classical candidate features, and brick-wall CNOT layers introduce
nonlinear feature mixing through quantum measurement statistics.

The resulting columns are honest quantum-inspired nonlinear features, not a claim
of demonstrated quantum advantage.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from .ip_attribution_common import (
        DEFAULT_GROUP_COLUMNS,
        ensure_dir,
        get_feature_columns,
        normalize_name,
        read_table,
        save_json,
        setup_logging,
        split_mask,
        write_table,
    )
except ImportError:  # pragma: no cover
    from ip_attribution_common import (
        DEFAULT_GROUP_COLUMNS,
        ensure_dir,
        get_feature_columns,
        normalize_name,
        read_table,
        save_json,
        setup_logging,
        split_mask,
        write_table,
    )

LOGGER = logging.getLogger("ip_attribution.quantum")


@dataclass
class QuantumReservoirConfig:
    n_qubits: int = 6
    n_layers: int = 2
    max_input_features: int = 24
    seed: int = 42
    clip_value: float = 3.0
    input_scale: float = 1.0
    random_weight_scale: float = 0.75
    include_pairwise_zz: bool = True
    include_entropy: bool = True
    include_entanglement_entropy: bool = False
    shot_count: int = 0
    baseline_scope: str = "normal_train"
    output_format: str = "csv"

    def validate(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.n_qubits > 10:
            raise ValueError(
                "n_qubits > 10 is intentionally blocked for this lightweight simulator. "
                "Use fewer qubits or add a dedicated backend."
            )
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.max_input_features < 1:
            raise ValueError("max_input_features must be >= 1")
        if self.shot_count < 0:
            raise ValueError("shot_count must be non-negative; use 0 for deterministic expectations")


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    step = 1 << qubit
    period = step << 1
    out = state.copy()
    n = state.shape[0]
    for base in range(0, n, period):
        for offset in range(step):
            i0 = base + offset
            i1 = i0 + step
            a0 = state[i0]
            a1 = state[i1]
            out[i0] = gate[0, 0] * a0 + gate[0, 1] * a1
            out[i1] = gate[1, 0] * a0 + gate[1, 1] * a1
    return out


def _cnot_source_indices(n_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 1 << n_qubits
    idx = np.arange(dim)
    mask = ((idx >> control) & 1) == 1
    source = idx.copy()
    source[mask] = source[mask] ^ (1 << target)
    return source


def _z_signs(n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    idx = np.arange(dim)
    signs = np.empty((n_qubits, dim), dtype=float)
    for q in range(n_qubits):
        signs[q] = np.where(((idx >> q) & 1) == 0, 1.0, -1.0)
    return signs


def _measurement_entropy(probs: np.ndarray) -> tuple[float, float]:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0, 0.0
    shannon = -float(np.sum(probs * np.log(probs))) / np.log(len(probs) if len(probs) > 1 else 2)
    linear = 1.0 - float(np.sum(probs**2))
    return shannon, linear


def _bipartite_entanglement_entropy(state: np.ndarray, n_qubits: int) -> float:
    if n_qubits < 2:
        return 0.0
    left = n_qubits // 2
    right = n_qubits - left
    psi = state.reshape((1 << left, 1 << right))
    singular_values = np.linalg.svd(psi, compute_uv=False)
    probs = np.square(np.abs(singular_values))
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / np.log(min(1 << left, 1 << right)))


class QuantumReservoirFeatureMap:
    """Fixed random RY/RZ + CNOT reservoir for candidate-IP features."""

    def __init__(self, config: QuantumReservoirConfig | None = None):
        self.config = config or QuantumReservoirConfig()
        self.config.validate()
        self.input_columns_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.ry_weights_: np.ndarray | None = None
        self.rz_weights_: np.ndarray | None = None
        self.ry_bias_: np.ndarray | None = None
        self.rz_bias_: np.ndarray | None = None
        self.z_signs_: np.ndarray | None = None
        self.cnot_indices_: list[np.ndarray] = []
        self.baseline_z_mean_: np.ndarray | None = None
        self.quantum_feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame, feature_mode: str = "classical") -> "QuantumReservoirFeatureMap":
        all_classical = get_feature_columns(df, feature_mode=feature_mode)
        if not all_classical:
            raise ValueError("No numeric candidate features are available for quantum enrichment.")

        train = split_mask(df, "train")
        if not train.any():
            LOGGER.warning("No explicit train split found; fitting quantum normalizers on all candidate rows.")
            train = pd.Series(True, index=df.index)

        X_train_raw = df.loc[train, all_classical].apply(pd.to_numeric, errors="coerce")
        variances = X_train_raw.var(axis=0, skipna=True).fillna(0.0).sort_values(ascending=False)
        selected = variances.head(self.config.max_input_features).index.tolist()
        if not selected:
            selected = all_classical[: self.config.max_input_features]
        self.input_columns_ = selected

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        X_train = self.imputer_.fit_transform(X_train_raw[selected])
        self.scaler_.fit(X_train)

        rng = np.random.default_rng(self.config.seed)
        n_inputs = len(self.input_columns_)
        shape = (self.config.n_layers, self.config.n_qubits, n_inputs)
        self.ry_weights_ = rng.normal(0.0, self.config.random_weight_scale, size=shape)
        self.rz_weights_ = rng.normal(0.0, self.config.random_weight_scale, size=shape)
        self.ry_bias_ = rng.uniform(-np.pi, np.pi, size=(self.config.n_layers, self.config.n_qubits))
        self.rz_bias_ = rng.uniform(-np.pi, np.pi, size=(self.config.n_layers, self.config.n_qubits))
        self.z_signs_ = _z_signs(self.config.n_qubits)
        self.cnot_indices_ = self._build_cnot_schedule()
        self.quantum_feature_names_ = self._feature_names()

        baseline_mask = self._baseline_mask(df)
        if not baseline_mask.any():
            LOGGER.warning("Quantum baseline scope produced no rows; using train candidates as benign baseline.")
            baseline_mask = train
        baseline_quantum = self._compute_quantum_features(df.loc[baseline_mask])
        z_cols = [i for i, name in enumerate(self.quantum_feature_names_) if name.startswith("q_z_")]
        if z_cols:
            self.baseline_z_mean_ = np.nanmean(baseline_quantum[:, z_cols], axis=0)
        else:
            self.baseline_z_mean_ = np.zeros(self.config.n_qubits, dtype=float)
        LOGGER.info(
            "Fitted quantum reservoir with %d selected input features, %d qubits, %d layers.",
            len(self.input_columns_),
            self.config.n_qubits,
            self.config.n_layers,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        # Drop old q_ columns if a caller accidentally enriches an already enriched table.
        keep_cols = [c for c in df.columns if not c.startswith("q_") and not c.startswith("quantum_")]
        out = df[keep_cols].copy()
        quantum = self._compute_quantum_features(df)
        q_df = pd.DataFrame(quantum, columns=self.quantum_feature_names_, index=df.index)

        z_cols = [c for c in q_df.columns if c.startswith("q_z_")]
        if z_cols and self.baseline_z_mean_ is not None:
            z = q_df[z_cols].to_numpy(dtype=float)
            baseline = self.baseline_z_mean_.reshape(1, -1)
            diff = z - baseline
            q_df["q_benign_z_l2"] = np.linalg.norm(diff, axis=1)
            q_df["q_benign_z_mean_abs"] = np.mean(np.abs(diff), axis=1)
            z_norm = np.linalg.norm(z, axis=1) * np.linalg.norm(baseline)
            cosine = np.divide((z * baseline).sum(axis=1), z_norm, out=np.zeros(len(z)), where=z_norm > 1e-12)
            q_df["q_benign_z_cosine_distance"] = 1.0 - cosine
        return pd.concat([out.reset_index(drop=True), q_df.reset_index(drop=True)], axis=1)

    def fit_transform(self, df: pd.DataFrame, feature_mode: str = "classical") -> pd.DataFrame:
        return self.fit(df, feature_mode=feature_mode).transform(df)

    def _check_fitted(self) -> None:
        attrs = [
            self.input_columns_,
            self.imputer_,
            self.scaler_,
            self.ry_weights_,
            self.rz_weights_,
            self.ry_bias_,
            self.rz_bias_,
            self.z_signs_,
        ]
        if any(attr is None or (isinstance(attr, list) and not attr) for attr in attrs):
            raise RuntimeError("QuantumReservoirFeatureMap is not fitted.")

    def _baseline_mask(self, df: pd.DataFrame) -> pd.Series:
        scope = normalize_name(self.config.baseline_scope)
        train = split_mask(df, "train")
        if scope in {"normal_train", "benign_train"} and "kind" in df.columns:
            kind = df["kind"].astype(str).map(normalize_name)
            normal = kind.str.contains("normal|benign|clean", regex=True, na=False)
            return train & normal
        if scope in {"train", "all_train"}:
            return train
        if scope in {"all", "all_candidates"}:
            return pd.Series(True, index=df.index)
        LOGGER.warning("Unknown baseline_scope=%s; using normal_train fallback.", self.config.baseline_scope)
        if "kind" in df.columns:
            kind = df["kind"].astype(str).map(normalize_name)
            normal = kind.str.contains("normal|benign|clean", regex=True, na=False)
            return train & normal
        return train

    def _build_cnot_schedule(self) -> list[np.ndarray]:
        indices: list[np.ndarray] = []
        for layer in range(self.config.n_layers):
            offset = layer % 2
            pairs = [(q, q + 1) for q in range(offset, self.config.n_qubits - 1, 2)]
            if self.config.n_qubits > 2 and layer % 2 == 1:
                pairs.append((self.config.n_qubits - 1, 0))
            for control, target in pairs:
                indices.append(_cnot_source_indices(self.config.n_qubits, control, target))
        return indices

    def _feature_names(self) -> list[str]:
        names = [f"q_z_{q}" for q in range(self.config.n_qubits)]
        if self.config.include_pairwise_zz:
            for q in range(self.config.n_qubits - 1):
                names.append(f"q_zz_{q}_{q + 1}")
            if self.config.n_qubits > 2:
                names.append(f"q_zz_{self.config.n_qubits - 1}_0")
        if self.config.include_entropy:
            names.extend(["q_measurement_entropy", "q_linear_entropy"])
        if self.config.include_entanglement_entropy:
            names.append("q_entanglement_entropy")
        return names

    def _normalized_inputs(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self.imputer_ is not None
        assert self.scaler_ is not None
        X = df.reindex(columns=self.input_columns_).apply(pd.to_numeric, errors="coerce")
        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=self.config.clip_value, neginf=-self.config.clip_value)
        X_scaled = np.clip(X_scaled, -self.config.clip_value, self.config.clip_value)
        return X_scaled * self.config.input_scale

    def _compute_quantum_features(self, df: pd.DataFrame) -> np.ndarray:
        X = self._normalized_inputs(df)
        rows = [self._simulate_row(x, row_index=i) for i, x in enumerate(X)]
        if not rows:
            return np.empty((0, len(self.quantum_feature_names_)), dtype=float)
        return np.vstack(rows)

    def _simulate_row(self, x: np.ndarray, row_index: int = 0) -> np.ndarray:
        self._check_fitted()
        assert self.ry_weights_ is not None
        assert self.rz_weights_ is not None
        assert self.ry_bias_ is not None
        assert self.rz_bias_ is not None
        assert self.z_signs_ is not None
        dim = 1 << self.config.n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0.0j

        cnot_iter = iter(self.cnot_indices_)
        for layer in range(self.config.n_layers):
            for q in range(self.config.n_qubits):
                theta_y = float(np.dot(self.ry_weights_[layer, q], x) + self.ry_bias_[layer, q])
                theta_z = float(np.dot(self.rz_weights_[layer, q], x) + self.rz_bias_[layer, q])
                state = _apply_single_qubit_gate(state, _ry(theta_y), q)
                state = _apply_single_qubit_gate(state, _rz(theta_z), q)
            # Apply the same number of CNOTs as were appended for this layer.
            offset = layer % 2
            n_pairs = len([(q, q + 1) for q in range(offset, self.config.n_qubits - 1, 2)])
            if self.config.n_qubits > 2 and layer % 2 == 1:
                n_pairs += 1
            for _ in range(n_pairs):
                state = state[next(cnot_iter)]

        probs = np.square(np.abs(state))
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / prob_sum

        if self.config.shot_count and self.config.shot_count > 0:
            rng = np.random.default_rng(self.config.seed + 1000003 + int(row_index))
            counts = rng.multinomial(self.config.shot_count, probs)
            probs_obs = counts / max(1, self.config.shot_count)
        else:
            probs_obs = probs

        z = self.z_signs_ @ probs_obs
        features: list[float] = [float(v) for v in z]
        if self.config.include_pairwise_zz:
            for q in range(self.config.n_qubits - 1):
                features.append(float(np.sum(self.z_signs_[q] * self.z_signs_[q + 1] * probs_obs)))
            if self.config.n_qubits > 2:
                features.append(float(np.sum(self.z_signs_[-1] * self.z_signs_[0] * probs_obs)))
        if self.config.include_entropy:
            shannon, linear = _measurement_entropy(probs_obs)
            features.extend([shannon, linear])
        if self.config.include_entanglement_entropy:
            features.append(_bipartite_entanglement_entropy(state, self.config.n_qubits))
        return np.asarray(features, dtype=float)


def enrich_candidate_features(
    candidate_features_path: str | Path,
    output_dir: str | Path,
    config: QuantumReservoirConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], QuantumReservoirFeatureMap]:
    config = config or QuantumReservoirConfig()
    config.validate()
    output_dir = ensure_dir(output_dir)
    df = read_table(candidate_features_path)
    if df.empty:
        raise ValueError(f"Candidate feature table is empty: {candidate_features_path}")

    reservoir = QuantumReservoirFeatureMap(config)
    enriched = reservoir.fit_transform(df, feature_mode="classical")

    suffix = ".csv" if config.output_format.lower() == "csv" else f".{config.output_format.lower().lstrip('.')}"
    enriched_path = output_dir / f"candidate_ip_features_enriched_all{suffix}"
    write_table(enriched, enriched_path)

    split_paths: dict[str, str] = {}
    if "split" in enriched.columns:
        for split_value, split_df in enriched.groupby("split", dropna=False):
            split_name = normalize_name(split_value)
            split_path = output_dir / f"candidate_ip_features_enriched_{split_name}{suffix}"
            write_table(split_df.reset_index(drop=True), split_path)
            split_paths[split_name] = str(split_path)

    model_path = output_dir / "quantum_reservoir.joblib"
    joblib.dump(reservoir, model_path)
    summary = {
        "config": asdict(config),
        "input_path": str(candidate_features_path),
        "enriched_path": str(enriched_path),
        "split_paths": split_paths,
        "model_path": str(model_path),
        "n_rows": int(len(enriched)),
        "n_input_columns": int(len(reservoir.input_columns_)),
        "input_columns": reservoir.input_columns_,
        "quantum_columns": [c for c in enriched.columns if c.startswith("q_") or c.startswith("quantum_")],
        "baseline_scope": config.baseline_scope,
    }
    save_json(summary, output_dir / "quantum_enrichment_summary.json")
    LOGGER.info("Saved quantum-enriched candidate features to %s", enriched_path)
    return enriched, summary, reservoir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add fixed quantum-reservoir features to candidate source-IP rows.")
    parser.add_argument("--candidate-features-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-input-features", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-value", type=float, default=3.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--random-weight-scale", type=float, default=0.75)
    parser.add_argument("--no-pairwise-zz", action="store_true")
    parser.add_argument("--no-entropy", action="store_true")
    parser.add_argument("--include-entanglement-entropy", action="store_true")
    parser.add_argument("--shot-count", type=int, default=0)
    parser.add_argument("--baseline-scope", default="normal_train", choices=["normal_train", "benign_train", "train", "all_train", "all"])
    parser.add_argument("--output-format", default="csv", choices=["csv", "parquet", "feather", "pkl"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> QuantumReservoirConfig:
    return QuantumReservoirConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_input_features=args.max_input_features,
        seed=args.seed,
        clip_value=args.clip_value,
        input_scale=args.input_scale,
        random_weight_scale=args.random_weight_scale,
        include_pairwise_zz=not args.no_pairwise_zz,
        include_entropy=not args.no_entropy,
        include_entanglement_entropy=args.include_entanglement_entropy,
        shot_count=args.shot_count,
        baseline_scope=args.baseline_scope,
        output_format=args.output_format,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args)
    enrich_candidate_features(args.candidate_features_path, args.output_dir, config)


if __name__ == "__main__":
    main()
