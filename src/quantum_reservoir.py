import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    entropy,
    partial_trace,
)


# Density Matrix utilities

def next_power_of_2(n: int) -> int:
    """Smallest 2^k >= n. Returns 1 for n <= 1."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _stable_probability_bin(key: str, dim: int) -> int:
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % dim


def validate_density_matrix(
    rho: DensityMatrix,
    dim: Optional[int] = None,
    atol: float = 1e-8,
) -> DensityMatrix:
    data = np.asarray(rho.data, dtype=complex)

    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"Density matrix must be square, got shape={data.shape}")

    n = data.shape[0]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Density-matrix dimension must be a power of 2, got {n}")
    if dim is not None and data.shape != (dim, dim):
        raise ValueError(f"Density matrix shape {data.shape} does not match {(dim, dim)}")
    if not np.allclose(data, data.conj().T, atol=atol):
        raise ValueError("Density matrix is not Hermitian")

    tr = np.trace(data)
    if abs(tr.imag) > atol or not np.isclose(tr.real, 1.0, atol=atol):
        raise ValueError(f"Density matrix trace must be 1, got {tr}")

    hermitian_data = 0.5 * (data + data.conj().T)
    min_eval = float(np.linalg.eigvalsh(hermitian_data).min())
    if min_eval < -atol:
        raise ValueError(
            f"Density matrix is not positive semidefinite; min eigenvalue={min_eval}"
        )

    return rho


def build_density_matrix(
    ip_freq: dict[str, float],
    dim: Optional[int] = None,
) -> DensityMatrix:
    """
    Build a diagonal mixed state from source-IP frequency/count weights.

    If dim is provided, every source IP is mapped into the fixed Hilbert-space
    basis by a stable hash. This is required for reservoir use: the same IP
    must correspond to the same basis bin across windows and in the benign
    baseline.

    If dim is None, a compact per-window density matrix is still allowed for
    standalone diagnostics, but it should not be used as a reservoir state.
    """
    if not ip_freq:
        raise ValueError("ip_freq must contain at least one source-IP bin")

    weights = {str(ip): float(freq) for ip, freq in ip_freq.items()}

    if any(freq < 0 for freq in weights.values()):
        raise ValueError("ip_freq must not contain negative weights")

    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("ip_freq weights must sum to a positive value")

    if dim is None:
        dim = next_power_of_2(len(weights))
        diag = np.zeros(dim, dtype=float)
        for i, freq in enumerate(weights.values()):
            diag[i] = freq
    else:
        dim = int(dim)
        if dim <= 0 or dim & (dim - 1):
            raise ValueError(f"dim={dim} must be a positive power of 2")

        diag = np.zeros(dim, dtype=float)
        for ip, freq in weights.items():
            diag[_stable_probability_bin(ip, dim)] += freq

    diag /= diag.sum()
    return validate_density_matrix(DensityMatrix(np.diag(diag)), dim=dim)


def von_neumann_entropy(rho: DensityMatrix) -> float:
    return float(entropy(validate_density_matrix(rho), base=2))


def trace_distance(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Quantum anomaly score."""
    rho = validate_density_matrix(rho)
    sigma = validate_density_matrix(sigma, dim=rho.data.shape[0])
    diff = rho.data - sigma.data
    sv = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(np.abs(sv)))


def pad_density_matrix(rho: DensityMatrix, target_dim: int) -> DensityMatrix:
    """Block-diagonal padding with zeros."""
    rho = validate_density_matrix(rho)
    n = rho.data.shape[0]

    if n == target_dim:
        return rho
    if n > target_dim:
        raise ValueError(f"Cannot shrink rho (dim={n}) to {target_dim}")

    padded = np.zeros((target_dim, target_dim), dtype=complex)
    padded[:n, :n] = rho.data
    return validate_density_matrix(DensityMatrix(padded), dim=target_dim)


def match_density_matrix_dimension(rho: DensityMatrix, target_dim: int) -> DensityMatrix:
    """
    Return a valid density matrix with dimension target_dim.
    Smaller states are padded. Larger diagonal states are folded into the target
    bins; non-diagonal states cannot be safely compressed without extra mapping.
    """
    rho = validate_density_matrix(rho)
    n = rho.data.shape[0]

    if n == target_dim:
        return rho
    if n < target_dim:
        return pad_density_matrix(rho, target_dim)

    data = np.asarray(rho.data, dtype=complex)
    if not np.allclose(data, np.diag(np.diag(data)), atol=1e-8):
        raise ValueError(
            f"Cannot compress non-diagonal density matrix from dim={n} to {target_dim}"
        )

    folded = np.zeros(target_dim, dtype=float)
    for i, p in enumerate(np.real(np.diag(data))):
        folded[i % target_dim] += max(float(p), 0.0)

    folded /= folded.sum()
    return validate_density_matrix(DensityMatrix(np.diag(folded)), dim=target_dim)


# Reservoir

class QuantumReservoir:
    def __init__(
        self,
        n_qubits: int,
        depth: int = 4,
        seed: int = 42,
        feedback_scale: float = 0.1,
        shots: Optional[int] = None,
        plot_circuit: bool = True,
    ):
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if shots is not None and shots <= 0:
            raise ValueError("shots must be positive when provided")

        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.seed = int(seed)
        self.feedback_scale = float(feedback_scale)
        self.shots = shots
        self.plot_circuit = bool(plot_circuit)

        # Fixed reservoir random angles. These are never updated after init.
        rng = np.random.default_rng(self.seed)
        self.reservoir_angles = rng.uniform(
            0,
            2 * np.pi,
            (self.depth, self.n_qubits, 2),
        )
        self._shot_rng = np.random.default_rng(self.seed + 1)

        # Adaptive feedback memory. Reset between datasets.
        self._prev_z: Optional[np.ndarray] = None

    # Encoding with adaptive feedback

    def _adaptive_encoding_angles(self, features: np.ndarray) -> np.ndarray:
        """
        Map an arbitrary number of classical window features to exactly
        n_qubits encoding angles.

        This intentionally decouples reservoir size from the number of
        aggregate classical features:
          - fewer features than qubits: pad unused encoding slots with pi/2
          - more features than qubits: fold contiguous feature chunks by average
        """
        features = np.asarray(features, dtype=float).ravel()

        if features.size == 0:
            raise ValueError("features must contain at least one value")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain NaN or infinite values")

        f_min = float(features.min())
        f_max = float(features.max())

        if f_max - f_min < 1e-10:
            raw_angles = np.full(features.size, np.pi / 2)
        else:
            raw_angles = np.pi * (features - f_min) / (f_max - f_min)

        if raw_angles.size == self.n_qubits:
            base = raw_angles
        elif raw_angles.size < self.n_qubits:
            base = np.full(self.n_qubits, np.pi / 2)
            base[:raw_angles.size] = raw_angles
        else:
            base = np.array(
                [chunk.mean() for chunk in np.array_split(raw_angles, self.n_qubits)],
                dtype=float,
            )

        if self._prev_z is None or self.feedback_scale == 0.0:
            return base

        shift = self.feedback_scale * np.pi * self._prev_z
        return np.clip(base + shift, 0, np.pi)

    def _build_circuit(
        self,
        encoding_angles: np.ndarray,
        measure: bool = False,
    ) -> QuantumCircuit:
        """
        Build the reservoir circuit. If measure=True, append measurement
        operations for shot-based simulation.
        """
        encoding_angles = np.asarray(encoding_angles, dtype=float).ravel()

        if encoding_angles.size != self.n_qubits:
            raise ValueError(
                f"encoding_angles must have length {self.n_qubits}, "
                f"got {encoding_angles.size}"
            )

        qc = QuantumCircuit(self.n_qubits, self.n_qubits if measure else 0)

        # 1) Encoding
        for i, theta in enumerate(encoding_angles):
            qc.ry(float(theta), i)

        # 2) Reservoir: alternating RY/RZ + brick-wall CNOT
        for layer in range(self.depth):
            for i in range(self.n_qubits):
                qc.ry(float(self.reservoir_angles[layer, i, 0]), i)
                qc.rz(float(self.reservoir_angles[layer, i, 1]), i)

            offset = layer % 2
            for i in range(offset, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)

        if self.plot_circuit:
            qc.draw(output="mpl", filename="circuit_reservoir.png")

        if measure:
            qc.measure(range(self.n_qubits), range(self.n_qubits))

        return qc

    # Measurements

    def _measure_z_expectations(self, evolved_rho: DensityMatrix) -> np.ndarray:
        """
        shots=None: exact expectations from the evolved density matrix.
        shots=N: sample computational-basis outcomes from the evolved mixed state.
        """
        evolved_rho = validate_density_matrix(evolved_rho, dim=2 ** self.n_qubits)

        if self.shots is None:
            z_vals = np.zeros(self.n_qubits, dtype=float)
            for i in range(self.n_qubits):
                pauli = "I" * (self.n_qubits - 1 - i) + "Z" + "I" * i
                z_vals[i] = float(
                    evolved_rho.expectation_value(SparsePauliOp(pauli)).real
                )
            return z_vals

        probs = np.real(np.diag(evolved_rho.data))
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            raise ValueError(
                "Invalid evolved density matrix: diagonal probabilities sum to zero"
            )
        probs /= total

        samples = self._shot_rng.choice(len(probs), size=self.shots, p=probs)

        z_vals = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_qubits):
            bits = (samples >> i) & 1
            z_vals[i] = float(np.mean(1 - 2 * bits))
        return z_vals

    def _entanglement_entropy(self, evolved_rho: DensityMatrix) -> float:
        """Entropy of a half-system reduction of the evolved density matrix."""
        evolved_rho = validate_density_matrix(evolved_rho, dim=2 ** self.n_qubits)

        if self.n_qubits < 2:
            return 0.0

        half = self.n_qubits // 2
        rho_B = partial_trace(evolved_rho, list(range(half)))
        return float(entropy(rho_B, base=2))

    def _evolve_density_matrix(
        self,
        rho: DensityMatrix,
        qc_no_meas: QuantumCircuit,
    ) -> DensityMatrix:
        """
        Apply the encoding + reservoir circuit to the supplied initial rho.

        The state must already live in the reservoir Hilbert space. Do not pad
        or compress here: doing so would silently change the IP-to-basis mapping.
        """
        target_dim = 2 ** self.n_qubits
        rho = validate_density_matrix(rho, dim=target_dim)
        evolved = rho.evolve(qc_no_meas)
        return validate_density_matrix(evolved, dim=target_dim)

    # Public API

    def reset_state(self) -> None:
        self._prev_z = None

    def process_window(
        self,
        features: np.ndarray,
        ip_freq: dict[str, float],
        baseline_rho: Optional[DensityMatrix] = None,
    ) -> dict:
        """
        Process one window, initializing the reservoir from this window's
        source-IP density matrix instead of from |0...0>.
        """
        target_dim = 2 ** self.n_qubits
        rho = build_density_matrix(ip_freq, dim=target_dim)

        # Reservoir circuit with no measurements appended.
        angles = self._adaptive_encoding_angles(features)
        qc = self._build_circuit(angles, measure=False)

        rho_evolved = self._evolve_density_matrix(rho, qc)
        s_rho = von_neumann_entropy(rho_evolved)

        if baseline_rho is not None:
            sigma = validate_density_matrix(baseline_rho, dim=target_dim)
            sigma_evolved = self._evolve_density_matrix(sigma, qc)
            td = trace_distance(rho_evolved, sigma_evolved)
        else:
            td = 0.0

        z_exp = self._measure_z_expectations(rho_evolved)
        s_ent = self._entanglement_entropy(rho_evolved)

        # Update adaptive memory.
        self._prev_z = z_exp.copy()

        return {
            "z_expectations": z_exp,
            "s_rho": s_rho,
            "trace_distance": td,
            "s_entanglement": s_ent,
        }

    def process_dataset(
        self,
        window_features: np.ndarray,
        ip_distributions: pd.DataFrame,
        baseline_rho: Optional[DensityMatrix] = None,
        verbose: bool = True,
    ) -> dict:
        """Process all windows of one dataset."""
        self.reset_state()

        window_features = np.asarray(window_features, dtype=float)
        if window_features.ndim != 2:
            raise ValueError(
                f"window_features must be a 2D array, got shape={window_features.shape}"
            )
        if not np.all(np.isfinite(window_features)):
            raise ValueError("window_features contain NaN or infinite values")

        required_cols = {"window_id", "src_ip", "freq"}
        missing = sorted(required_cols - set(ip_distributions.columns))
        if missing:
            raise ValueError(
                f"ip_distributions is missing required column(s): {missing}"
            )

        n_windows = window_features.shape[0]

        z_all = np.zeros((n_windows, self.n_qubits), dtype=float)
        s_rho = np.zeros(n_windows, dtype=float)
        td = np.zeros(n_windows, dtype=float)
        s_ent = np.zeros(n_windows, dtype=float)

        for t in range(n_windows):
            sub = ip_distributions[ip_distributions["window_id"] == t]
            ip_freq = dict(zip(sub["src_ip"], sub["freq"]))

            if not ip_freq:
                ip_freq = {"__empty__": 1.0}

            result = self.process_window(
                features=window_features[t],
                ip_freq=ip_freq,
                baseline_rho=baseline_rho,
            )

            z_all[t] = result["z_expectations"]
            s_rho[t] = result["s_rho"]
            td[t] = result["trace_distance"]
            s_ent[t] = result["s_entanglement"]

            if verbose:
                print(f"[phase2]   window {t + 1:02d}/{n_windows} done")

        return {
            "z_expectations": z_all,
            "s_rho": s_rho,
            "trace_distance": td,
            "s_entanglement": s_ent,
        }


# File-level entry points

def enrich_file(
    windows_csv: str | Path,
    ip_dist_csv: str | Path,
    output_dir: str | Path,
    reservoir: QuantumReservoir,
    baseline_rho: Optional[DensityMatrix] = None,
    feature_columns: Optional[list[str]] = None,
    verbose: bool = True,
) -> Path:
    """Enrich one dataset through the quantum reservoir."""
    windows_csv = Path(windows_csv)
    ip_dist_csv = Path(ip_dist_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        mode = "statevector" if reservoir.shots is None else f"shots={reservoir.shots}"
        print(f"[phase2] {windows_csv.name}  ({mode})")

    win_df = pd.read_csv(windows_csv)
    ip_df = pd.read_csv(ip_dist_csv)

    if "window_id" not in win_df.columns:
        raise ValueError(f"{windows_csv.name} has no window_id column")

    if feature_columns is None:
        feature_columns = [c for c in win_df.columns if c != "window_id"]

    missing = [c for c in feature_columns if c not in win_df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from {windows_csv.name}: {missing}")

    forbidden_inputs = {
        "label",
        "labels",
        "attack",
        "is_seeded_ddos",
        "burst_id",
        "burst_phase",
        "split",
        "scenario",
        "dataset_id",
        "row_in_window",
        "source_dataset",
    }
    leaks = sorted(c for c in feature_columns if c.lower() in forbidden_inputs)
    if leaks:
        raise ValueError(
            f"Leakage/audit columns cannot be used as reservoir inputs: {leaks}"
        )

    if verbose and reservoir.n_qubits != len(feature_columns):
        print(
            f"[phase2]   encoding {len(feature_columns)} classical feature(s) "
            f"into {reservoir.n_qubits} reservoir qubit(s)"
        )

    win_df = win_df.sort_values("window_id").reset_index(drop=True)
    features = win_df[feature_columns].values.astype(float)

    result = reservoir.process_dataset(
        window_features=features,
        ip_distributions=ip_df,
        baseline_rho=baseline_rho,
        verbose=verbose,
    )

    # Build output DataFrame.
    out_df = win_df[["window_id"] + feature_columns].copy()
    for i in range(reservoir.n_qubits):
        out_df[f"z_qubit_{i}"] = result["z_expectations"][:, i]

    out_df["s_rho"] = result["s_rho"]
    out_df["trace_distance"] = result["trace_distance"]
    out_df["s_entanglement"] = result["s_entanglement"]

    stem = windows_csv.stem.replace("_windows", "")
    out_path = output_dir / f"{stem}_enriched.csv"
    out_df.to_csv(out_path, index=False)

    if verbose:
        print(f"[phase2]   -> {out_path.name}  shape {out_df.shape}")

    return out_path


def enrich_tree(
    phase1_root: str | Path,
    output_root: str | Path,
    reservoir: QuantumReservoir,
    baseline_rho: Optional[DensityMatrix] = None,
) -> list[Path]:
    phase1_root = Path(phase1_root)
    output_root = Path(output_root)

    windows_files = sorted(phase1_root.rglob("*_windows.csv"))
    print(f"[phase2] Found {len(windows_files)} dataset(s) under {phase1_root}")

    out_paths: list[Path] = []
    for win in windows_files:
        ipd = win.with_name(win.name.replace("_windows.csv", "_ip_distributions.csv"))
        if not ipd.exists():
            print(f"[phase2]  ! ip_distributions missing for {win.name}, skipping")
            continue

        rel_dir = win.parent.relative_to(phase1_root)
        out_dir = output_root / rel_dir
        out = enrich_file(win, ipd, out_dir, reservoir, baseline_rho, verbose=True)
        out_paths.append(out)

    print(f"\n[phase2] Done. {len(out_paths)} datasets enriched -> {output_root}")
    return out_paths


# Baseline I/O helpers

def save_baseline(rho: DensityMatrix, path: str | Path) -> None:
    """Save baseline density matrix to disk as .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, rho.data)
    print(f"[phase2] Saved baseline -> {path}  shape {rho.data.shape}")


def load_baseline(path: str | Path) -> DensityMatrix:
    arr = np.load(path)
    return validate_density_matrix(DensityMatrix(arr))


if __name__ == "__main__":
    # Enrich all full Option 2 window files using density-matrix reservoir input.
    FEATURE_SET = "full"
    N_QUBITS = 10
    DEPTH = 4
    SEED = 42

    reservoir = QuantumReservoir(
        n_qubits=N_QUBITS,
        depth=DEPTH,
        seed=SEED,
        shots=None,
        plot_circuit=False,
    )

    baseline = load_baseline("outputs/option_2/baseline/baseline_rho.npy")

    enrich_tree(
        phase1_root="outputs/option_2/full",
        output_root="outputs/option_2/full/enriched",
        reservoir=reservoir,
        baseline_rho=baseline,
    )
