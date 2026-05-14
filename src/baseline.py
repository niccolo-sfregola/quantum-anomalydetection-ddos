import numpy as np
import pandas as pd
from pathlib import Path

from niccolo_unsupervised.quantum_reservoir import (
    DensityMatrix,
    build_density_matrix,
    next_power_of_2,
    pad_density_matrix,
    save_baseline,
)


def build_baseline(
    phase1_normal_train_root: str | Path,
    output_path: str | Path,
    verbose: bool = True,
) -> DensityMatrix:
    """
    Walk the output of the normal training datasets and build sigma.
    """
    phase1_root = Path(phase1_normal_train_root)
    output_path = Path(output_path)

    ip_dist_files = sorted(phase1_root.rglob("*_ip_distributions.csv"))
    if not ip_dist_files:
        raise FileNotFoundError(
            f"No *_ip_distributions.csv found under {phase1_root}. "
            f"Run Phase 1 on the normal training set first."
        )
    print(f"[baseline] Found {len(ip_dist_files)} normal training file(s)")

    # ── Pass 1: find global dimension ────────────────────────────────────
    max_ips = 0
    for f in ip_dist_files:
        df = pd.read_csv(f)
        per_window_max = df.groupby("window_id").size().max()
        max_ips = max(max_ips, int(per_window_max))
    global_dim = next_power_of_2(max_ips)

    if verbose:
        print(f"[baseline] Max IPs in any training window: {max_ips}")
        print(f"[baseline] Global density matrix dimension: {global_dim} × {global_dim}")

    # ── Pass 2: accumulate ρ matrices, all padded to global_dim ──────────
    accumulator = np.zeros((global_dim, global_dim), dtype=complex)
    n_windows   = 0

    for f in ip_dist_files:
        df = pd.read_csv(f)
        for wid, sub in df.groupby("window_id"):
            ip_freq = dict(zip(sub["src_ip"], sub["freq"]))
            if not ip_freq:
                continue
            rho        = build_density_matrix(ip_freq)
            rho_padded = pad_density_matrix(rho, global_dim)
            accumulator += rho_padded.data
            n_windows   += 1
        if verbose:
            print(f"[baseline]   processed {f.name}")

    # Average → still a valid density matrix (convex combination of valid ρ_i)
    sigma_data = accumulator / n_windows
    sigma      = DensityMatrix(sigma_data)

    if verbose:
        # Sanity check
        trace_sigma = np.trace(sigma_data).real
        print(f"[baseline] Averaged over {n_windows} benign windows")
        print(f"[baseline] Tr(σ) = {trace_sigma:.6f}  (should be ≈ 1.0)")

    save_baseline(sigma, output_path)
    return sigma


if __name__ == "__main__":
    # Edit paths to match your local layout.
    PHASE1_NORMAL_TRAIN = "quantum-anomalydetection-ddos/outputs/option_2/full/normal/train"
    OUTPUT_PATH         = "quantum-anomalydetection-ddos/outputs/option_2/baseline/baseline_rho.npy"

    build_baseline(PHASE1_NORMAL_TRAIN, OUTPUT_PATH)