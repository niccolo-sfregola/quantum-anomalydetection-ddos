import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def new_window_ids_for_size(size: int, n_windows: int) -> np.ndarray:
    """
    Match the repository's existing windowing convention:
        window_id = row_in_window // (rows_per_dataset // N_WINDOWS)
    clipped to [0, N_WINDOWS - 1].

    For 1,000 rows and 15 windows, this gives:
      windows 0-13: 66 rows each
      window 14   : 76 rows
    """
    rows_per_window = size // n_windows
    return np.clip(np.arange(size) // rows_per_window, 0, n_windows - 1).astype(int)


def original_window_ids_from_audit(audit: pd.DataFrame) -> np.ndarray:
    """
    Infer original 15-window IDs from the original row_in_window values.

    Full challenge datasets have 100,000 rows and the repository's original
    convention uses:
        row_in_window // (100_000 // 15)
    clipped to [0, 14].
    """
    original_total_rows = int(audit["row_in_window"].max()) + 1
    rows_per_window = original_total_rows // N_WINDOWS

    if rows_per_window <= 0:
        raise ValueError("Invalid row_in_window values; cannot infer windows.")

    return np.clip(
        audit["row_in_window"].to_numpy() // rows_per_window,
        0,
        N_WINDOWS - 1,
    ).astype(int)


def choose_rows_for_file(clean: pd.DataFrame, audit: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Select TARGET_ROWS rows while preserving the original 15-window structure.

    For attack files:
      - if a source window contains seeded DDoS rows, include a proportional
        number of seeded rows in the reduced window;
      - fill the remaining quota with benign rows from the same source window.

    For normal files:
      - sample rows from each source window normally.

    This prevents the reduced attack datasets from accidentally becoming
    all-benign.
    """
    if len(clean) != len(audit):
        raise ValueError(f"clean/audit length mismatch: {len(clean)} vs {len(audit)}")

    audit = audit.copy()
    audit["_orig_window_id"] = original_window_ids_from_audit(audit)
    audit["_is_seeded"] = (
        pd.to_numeric(audit["is_seeded_ddos"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    target_new_wids = new_window_ids_for_size(TARGET_ROWS, N_WINDOWS)
    quotas = pd.Series(target_new_wids).value_counts().sort_index().to_dict()

    selected_indices: list[int] = []

    for wid in range(N_WINDOWS):
        quota = int(quotas.get(wid, 0))
        if quota == 0:
            continue

        window_idx = audit.index[audit["_orig_window_id"] == wid].to_numpy()
        if len(window_idx) == 0:
            raise ValueError(f"No rows found for original window {wid}")

        window_audit = audit.loc[window_idx]

        seeded_idx = window_audit.index[window_audit["_is_seeded"] == 1].to_numpy()
        benign_idx = window_audit.index[window_audit["_is_seeded"] == 0].to_numpy()

        if len(seeded_idx) > 0:
            # Preserve approximately the original seeded fraction inside this window,
            # but force at least 1 seeded row so the attack window remains labelled.
            seeded_fraction = len(seeded_idx) / len(window_idx)
            n_seeded_target = int(round(quota * seeded_fraction))
            n_seeded_target = max(1, n_seeded_target)
            n_seeded_target = min(n_seeded_target, quota, len(seeded_idx))

            chosen_seeded = rng.choice(seeded_idx, size=n_seeded_target, replace=False)

            remaining = quota - n_seeded_target
            if remaining > 0:
                if len(benign_idx) >= remaining:
                    chosen_benign = rng.choice(benign_idx, size=remaining, replace=False)
                else:
                    chosen_benign = rng.choice(window_idx, size=remaining, replace=True)
                chosen = np.concatenate([chosen_seeded, chosen_benign])
            else:
                chosen = chosen_seeded

        else:
            if len(window_idx) >= quota:
                chosen = rng.choice(window_idx, size=quota, replace=False)
            else:
                chosen = rng.choice(window_idx, size=quota, replace=True)

        rng.shuffle(chosen)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices, dtype=int)

    if len(selected_indices) != TARGET_ROWS:
        raise RuntimeError(
            f"Internal error: selected {len(selected_indices)} rows, expected {TARGET_ROWS}"
        )

    return selected_indices


def reduce_one_pair(clean_path: Path, source_root: Path, target_root: Path, rng: np.random.Generator) -> None:
    audit_path = clean_path.with_name(clean_path.name.replace("_clean.csv", "_audit.csv"))

    if not audit_path.exists():
        raise FileNotFoundError(f"Missing audit file for {clean_path}:\n  {audit_path}")

    rel_dir = clean_path.parent.relative_to(source_root)
    out_dir = target_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_clean = out_dir / clean_path.name
    out_audit = out_dir / audit_path.name

    print(f"[reduce] {clean_path}")

    clean = pd.read_csv(clean_path, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)

    selected = choose_rows_for_file(clean, audit, rng)

    reduced_clean = clean.iloc[selected].reset_index(drop=True)
    reduced_audit = audit.iloc[selected].reset_index(drop=True)

    # Recreate reduced row positions 0..TARGET_ROWS-1.
    reduced_audit["row_in_window"] = np.arange(TARGET_ROWS)

    # Recreate window_id in the cleaned feature file using the same convention
    # expected by the rest of the repository.
    reduced_clean["window_id"] = new_window_ids_for_size(TARGET_ROWS, N_WINDOWS)

    reduced_clean.to_csv(out_clean, index=False)
    reduced_audit.to_csv(out_audit, index=False)

    seeded_rows = int(
        pd.to_numeric(reduced_audit["is_seeded_ddos"], errors="coerce")
        .fillna(0)
        .astype(int)
        .sum()
    )

    attack_windows = int(
        reduced_audit.assign(
            window_id=new_window_ids_for_size(TARGET_ROWS, N_WINDOWS)
        )
        .groupby("window_id")["is_seeded_ddos"]
        .max()
        .sum()
    )

    print(
        f"[reduce]   wrote {out_clean} and {out_audit} "
        f"| seeded_rows={seeded_rows}, attack_windows={attack_windows}"
    )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"SOURCE_ROOT does not exist:\n"
            f"  {SOURCE_ROOT}\n\n"
            f"Expected the full cleaned dataset here. If yours is elsewhere, "
            f"edit SOURCE_ROOT at the top of this file."
        )

    if TARGET_ROOT.exists() and OVERWRITE:
        print(f"[reduce] Removing existing target: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)

    clean_files = sorted(SOURCE_ROOT.rglob("*_clean.csv"))

    if not clean_files:
        raise FileNotFoundError(
            f"No *_clean.csv files found under:\n"
            f"  {SOURCE_ROOT}"
        )

    print(f"[reduce] Found {len(clean_files)} clean files under {SOURCE_ROOT}")
    print(f"[reduce] Target rows per dataset: {TARGET_ROWS}")
    print(f"[reduce] Writing to: {TARGET_ROOT}")

    rng = np.random.default_rng(SEED)

    for clean_path in clean_files:
        reduce_one_pair(clean_path, SOURCE_ROOT, TARGET_ROOT, rng)

    print(f"\n[reduce] Done. Reduced cleaned dataset written to {TARGET_ROOT}")


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # User settings
    # -----------------------------------------------------------------------------

    TARGET_ROWS = 50_000
    OPTION = "option_2"
    N_WINDOWS = 15
    SEED = 42

    SOURCE_ROOT = Path(f"cleaned_dataset/{OPTION}")
    TARGET_ROOT = Path(f"cleaned_dataset_{TARGET_ROWS}/{OPTION}")

    OVERWRITE = True

    main()