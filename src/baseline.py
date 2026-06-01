import numpy as np
import pandas as pd
from pathlib import Path

from quantum_reservoir import (
    DensityMatrix,
    build_density_matrix,
    next_power_of_2,
    save_baseline,
    validate_density_matrix,
)



def build_baseline(
    phase1_normal_train_root: str | Path,
    output_path: str | Path,
    verbose: bool = True,
    *,
    n_qubits: int,
) -> DensityMatrix:
    """
    Walk the output of the normal training datasets and build sigma.

    The baseline must be built directly in the same Hilbert space as the
    reservoir. Do not build a compact baseline and pad it later, because that
    would not preserve the source-IP-to-basis mapping.
    """
    phase1_root = Path(phase1_normal_train_root)
    output_path = Path(output_path)

    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    global_dim = 2 ** n_qubits

    ip_dist_files = sorted(phase1_root.rglob("*_ip_distributions.csv"))
    if not ip_dist_files:
        raise FileNotFoundError(
            f"No *_ip_distributions.csv found under {phase1_root}. "
            f"Run Phase 1 on the normal training set first."
        )
    print(f"[baseline] Found {len(ip_dist_files)} normal training file(s)")

    if verbose:
        print(f"[baseline] Reservoir qubits: {n_qubits}")
        print(f"[baseline] Baseline density matrix dimension: {global_dim} × {global_dim}")

    accumulator = np.zeros((global_dim, global_dim), dtype=complex)
    n_windows   = 0

    for f in ip_dist_files:
        df = pd.read_csv(f)
        for wid, sub in df.groupby("window_id"):
            ip_freq = dict(zip(sub["src_ip"], sub["freq"]))
            if not ip_freq:
                continue
            rho = build_density_matrix(ip_freq, dim=global_dim)
            accumulator += rho.data
            n_windows   += 1
        if verbose:
            print(f"[baseline]   processed {f.name}")

    if n_windows == 0:
        raise ValueError("No non-empty IP-distribution windows found for baseline")

    sigma_data = accumulator / n_windows
    sigma      = validate_density_matrix(DensityMatrix(sigma_data), dim=global_dim)

    if verbose:
        trace_sigma = np.trace(sigma_data).real
        print(f"[baseline] Averaged over {n_windows} benign windows")
        print(f"[baseline] Tr(σ) = {trace_sigma:.6f}  (should be ≈ 1.0)")

    save_baseline(sigma, output_path)
    return sigma


if __name__ == "__main__":
    # Build benign baseline density matrix from full normal/train windows.
    PHASE1_NORMAL_TRAIN = "outputs/option_2/full/normal/train"
    OUTPUT_PATH         = "outputs/option_2/baseline/baseline_rho.npy"

    FEATURE_SET         = "full"
    N_QUBITS            = 10

    build_baseline(
        phase1_normal_train_root = PHASE1_NORMAL_TRAIN,
        output_path              = OUTPUT_PATH,
        n_qubits                 = N_QUBITS,
    )