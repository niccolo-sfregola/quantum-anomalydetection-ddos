import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    SparsePauliOp,
    entropy,
    partial_trace,
)
from qiskit_aer import AerSimulator


#  Density Matrix utilities 

def next_power_of_2(n: int) -> int:
    """Smallest 2^k >= n. Returns 1 for n <= 1."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def build_density_matrix(ip_freq: dict[str, float],
                         dim: Optional[int] = None) -> DensityMatrix:
    """
    Build a diagonal mixed state from ip frequency counts.
    """
    freqs = np.array(list(ip_freq.values()), dtype=float)
    freqs = freqs / freqs.sum()
    n_ips = len(freqs)
 
    if dim is None:
        dim = next_power_of_2(n_ips)
    elif dim < n_ips:
        raise ValueError(f"dim={dim} too small for {n_ips} IPs")
    elif dim & (dim - 1):
        raise ValueError(f"dim={dim} must be a power of 2")
 
    diag = np.zeros(dim)
    diag[:n_ips] = freqs
    return DensityMatrix(np.diag(diag))


def von_neumann_entropy(rho: DensityMatrix) -> float:
    return float(entropy(rho, base=2))


def trace_distance(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Quantum anomaly score."""
    if rho.data.shape != sigma.data.shape:
        raise ValueError(
            f"Dimension mismatch: rho.shape={rho.data.shape}, "
            f"sigma.shape={sigma.data.shape}"
        )
    diff = rho.data - sigma.data
    sv = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(np.abs(sv)))
 
 
def pad_density_matrix(rho: DensityMatrix, target_dim: int) -> DensityMatrix:
    """(block-diagonal padding with zeros)."""
    n = rho.data.shape[0]
    if n == target_dim:
        return rho
    if n > target_dim:
        raise ValueError(f"Cannot shrink rho (dim={n}) to {target_dim}")
    padded = np.zeros((target_dim, target_dim), dtype=complex)
    padded[:n, :n] = rho.data
    return DensityMatrix(padded)
 

######## end of helper functions ###############

class QuantumReservoir:


    def __init__(
        self,
        n_qubits: int,
        depth: int = 4,
        seed: int = 42,
        feedback_scale: float = 0.1,
        shots: Optional[int] = None,
        plot_circuit: bool = True
    ):
        self.n_qubits       = n_qubits
        self.depth          = depth
        self.seed           = seed
        self.feedback_scale = feedback_scale
        self.shots          = shots
        self.plot_circuit   = plot_circuit

        # Fixed reservoir random angles (NEVER updated after init)
        rng = np.random.default_rng(seed)
        self.reservoir_angles = rng.uniform(0, 2 * np.pi, (depth, n_qubits, 2))

        self._aer = AerSimulator() if shots is not None else None

        # Adaptive feedback memory: reset between datasets
        self._prev_z: Optional[np.ndarray] = None

    # Encoding with adaptive feedback

    def _adaptive_encoding_angles(self, features: np.ndarray) -> np.ndarray:
        """Map features -> angles in [0, pi], shifted by previous expectation values of Z if available."""


        #first: sanity check to see if I'm dividing by zero
        f_min, f_max = features.min(), features.max()
        if f_max - f_min < 1e-10:
            base = np.full(self.n_qubits, np.pi / 2)
        else:
            base = np.pi * (features - f_min) / (f_max - f_min)

        if self._prev_z is None or self.feedback_scale == 0.0:
            return base
        shift = self.feedback_scale * np.pi * self._prev_z

        #clip to prevent to exit the periodic bound of pi
        return np.clip(base + shift, 0, np.pi)

    #  Circuit construction

    def _build_circuit(self, encoding_angles: np.ndarray,
                       measure: bool = False) -> QuantumCircuit:
        """
        Build the reservoir circuit. If `measure=True`, append measurement
        operations (needed for shot-based simulation).
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits if measure else 0)

        # 1) Encoding
        for i, theta in enumerate(encoding_angles):
            qc.ry(theta, i)

        # 2) Reservoir: alternating RY/RZ + brick-wall CNOT
        for layer in range(self.depth):
            for i in range(self.n_qubits):
                qc.ry(self.reservoir_angles[layer, i, 0], i)
                qc.rz(self.reservoir_angles[layer, i, 1], i)
            offset = layer % 2
            for i in range(offset, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)

        if self.plot_circuit == True:
            qc.draw(output="mpl",filename="circuit_reservoir.png")

        if measure:
            qc.measure(range(self.n_qubits), range(self.n_qubits))

        return qc

    # Measurements 

    def _measure_z_expectations(self, qc_no_meas: QuantumCircuit) -> np.ndarray:
        """
            shots=None: exact via Statevector (deterministic, no noise)
            shots=N: AerSimulator with N shots (introduces shot noise)
        """
        if self.shots is None:
            sv = Statevector(qc_no_meas)
            z_vals = np.zeros(self.n_qubits)
            for i in range(self.n_qubits):
                pauli = "I" * (self.n_qubits - 1 - i) + "Z" + "I" * i
                z_vals[i] = sv.expectation_value(SparsePauliOp(pauli)).real
            return z_vals

        # Shot-based: build a circuit WITH measurements, run on AerSimulator,
        # then estimate expectation value from the bitstring counts.
        qc_meas = qc_no_meas.copy()
        qc_meas.measure_all()  # measures into a fresh classical register

        result = self._aer.run(qc_meas, shots=self.shots).result()
        counts = result.get_counts()

        # Qiskit returns bitstrings with qubit 0 on the right
        z_vals = np.zeros(self.n_qubits)
        total  = sum(counts.values())
        for bitstring, c in counts.items():
            # strip whitespace (Qiskit may insert spaces between registers)
            bits = bitstring.replace(" ", "")
            for i in range(self.n_qubits):
                bit = bits[-(i + 1)]   # qubit i is at position -(i+1)
                z_vals[i] += (1 if bit == "0" else -1) * c
        return z_vals / total

    def _entanglement_entropy(self, qc_no_meas: QuantumCircuit) -> float:
        """
        Only available in statevector mode (no shot noise on entropy).
        """
        if self.n_qubits < 2:
            return 0.0
        sv = Statevector(qc_no_meas)
        half = self.n_qubits // 2
        rho_B = partial_trace(sv, list(range(half)))
        return float(entropy(rho_B, base=2))

    # Public API 

    def reset_state(self):
        self._prev_z = None

    def process_window(
        self,
        features: np.ndarray,
        ip_freq: dict[str, float],
        baseline_rho: Optional[DensityMatrix] = None,
    ) -> dict:
        """
        Process one window
        """
        # Density matrix features
        rho   = build_density_matrix(ip_freq)
        s_rho = von_neumann_entropy(rho)
 
        if baseline_rho is not None:
            target = max(rho.data.shape[0], baseline_rho.data.shape[0])
            td = trace_distance(
                pad_density_matrix(rho,          target),
                pad_density_matrix(baseline_rho, target),
            )
        else:
            td = 0.0
 
        # Reservoir circuit (no measurements appended yet)
        angles = self._adaptive_encoding_angles(features)
        qc     = self._build_circuit(angles, measure=False)
 
        z_exp = self._measure_z_expectations(qc)
        s_e   = self._entanglement_entropy(qc)
 
        # Update adaptive memory
        self._prev_z = z_exp.copy()
 
        return {
            "z_expectations" : z_exp,
            "s_rho"          : s_rho,
            "trace_distance" : td,
            "s_entanglement" : s_e,
        }

    def process_dataset(
        self,
        window_features: np.ndarray,
        ip_distributions: pd.DataFrame,
        baseline_rho: Optional[DensityMatrix] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Process all 15 windows of a dataset
        """
        self.reset_state()
        n_windows = len(window_features)

        z_all = np.zeros((n_windows, self.n_qubits))
        s_rho = np.zeros(n_windows)
        td    = np.zeros(n_windows)
        s_ent = np.zeros(n_windows)

        for t in range(n_windows):
            sub     = ip_distributions[ip_distributions["window_id"] == t]
            ip_freq = dict(zip(sub["src_ip"], sub["freq"]))
            if not ip_freq:
                ip_freq = {"__empty__": 1.0}

            r = self.process_window(window_features[t], ip_freq, baseline_rho)
            z_all[t] = r["z_expectations"]
            s_rho[t] = r["s_rho"]
            td[t]    = r["trace_distance"]
            s_ent[t] = r["s_entanglement"]

            if verbose:
                print(f"[phase2]   window {t+1:02d}/{n_windows} done")

        return {
            "z_expectations" : z_all,
            "s_rho"          : s_rho,
            "trace_distance" : td,
            "s_entanglement" : s_ent,
        }


#  File-level entry points 

def enrich_file(
    windows_csv: str | Path,
    ip_dist_csv: str | Path,
    output_dir: str | Path,
    reservoir: QuantumReservoir,
    baseline_rho: Optional[DensityMatrix] = None,
    feature_columns: Optional[list[str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Enrich one dataset through the quantum reservoir.
    """
    windows_csv = Path(windows_csv)
    ip_dist_csv = Path(ip_dist_csv)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        mode = "statevector" if reservoir.shots is None else f"shots={reservoir.shots}"
        print(f"[phase2] {windows_csv.name}  ({mode})")

    win_df = pd.read_csv(windows_csv)
    ip_df  = pd.read_csv(ip_dist_csv)

    if feature_columns is None:
        feature_columns = [c for c in win_df.columns if c != "window_id"]

    if reservoir.n_qubits != len(feature_columns):
        raise ValueError(
            f"Reservoir has {reservoir.n_qubits} qubits but input has "
            f"{len(feature_columns)} features. Use n_qubits={len(feature_columns)}."
        )

    features = win_df[feature_columns].values.astype(float)

    result = reservoir.process_dataset(
        window_features  = features,
        ip_distributions = ip_df,
        baseline_rho     = baseline_rho,
        verbose          = verbose,
    )

    # Build output DataFrame
    out_df = win_df[["window_id"] + feature_columns].copy()
    for i in range(reservoir.n_qubits):
        out_df[f"z_qubit_{i}"] = result["z_expectations"][:, i]
    out_df["s_rho"]          = result["s_rho"]
    out_df["trace_distance"] = result["trace_distance"]
    out_df["s_entanglement"] = result["s_entanglement"]

    stem = windows_csv.stem.replace("_windows", "")
    out_path = output_dir / f"{stem}_enriched.csv"
    out_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[phase2]   → {out_path.name}  shape {out_df.shape}")

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

    print(f"\n[phase2] Done. {len(out_paths)} datasets enriched → {output_root}")
    return out_paths


#  Baseline I/O helpers 

def save_baseline(rho: DensityMatrix, path: str | Path) -> None:
    """Save baseline density matrix to disk as .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, rho.data)
    print(f"[phase2] Saved baseline → {path}  shape {rho.data.shape}")


def load_baseline(path: str | Path) -> DensityMatrix:
   
    arr = np.load(path)
    return DensityMatrix(arr)


if __name__ == "__main__":

    reservoir = QuantumReservoir(n_qubits=10, depth=4)
    baseline = load_baseline("outputs/option_1/baseline/baseline_rho.npy")

    enrich_tree(
        phase1_root  = "outputs_50000/option_2/full",
        output_root  = "outputs_50000/option_2/full/enriched",
        reservoir    = reservoir,
        baseline_rho = baseline,
)