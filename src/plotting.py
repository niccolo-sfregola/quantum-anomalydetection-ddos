import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi"       : 110,
    "savefig.dpi"      : 140,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

N_WINDOWS    = 15
ROWS_PER_WIN = 100_000 // N_WINDOWS    # ≈ 6667 — used for time axis on plot 3
COLOR_BENIGN = "#4A90C2"
COLOR_ATTACK = "#D85A30"

# Heavy-tailed features that benefit from log scale on histograms.
# (phase0 already applies log1p to these, so values are already in log scale.)
# Note: bytes_per_second, inter_packet_arrival_mean/std are dropped by phase0
# as redundant, so they are excluded from this list.
HEAVY_FEATURES = [
    "duration", "packets_per_second",
    "total_packets", "total_bytes",
    "packet_size_avg", "packet_size_std",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf and NaN with 0 in numeric columns, just in case."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = (
        df[num_cols]
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
    )
    return df


def _check_window_id(df: pd.DataFrame) -> pd.DataFrame:
    """phase0_preprocessing must add window_id; we just trust it."""
    if "window_id" not in df.columns:
        raise ValueError(
            "Cleaned CSV is missing 'window_id'. Run phase0_preprocessing first."
        )
    return df


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


# ── Plot 1: Class balance ─────────────────────────────────────────────────────

def plot_class_balance(audit: pd.DataFrame, out: Path):
    counts = audit["is_seeded_ddos"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Benign", "DDoS"], counts.values,
                  color=[COLOR_BENIGN, COLOR_ATTACK])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:,}\n({val/counts.sum()*100:.2f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Flow count")
    ax.set_title("Flow-level class balance")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, out / "01_class_balance.png")


# ── Plot 2: Attack density across windows ─────────────────────────────────────

def plot_attack_density_per_window(df: pd.DataFrame, audit: pd.DataFrame, out: Path):
    df = df.copy()
    df["is_attack"] = audit["is_seeded_ddos"].values
    grp = df.groupby("window_id").agg(
        total = ("is_attack", "size"),
        attack= ("is_attack", "sum"),
    )
    grp["rate"] = grp["attack"] / grp["total"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.bar(grp.index, grp["attack"], color=COLOR_ATTACK)
    ax1.set_xlabel("Window ID (1-min bin)")
    ax1.set_ylabel("Attack flow count")
    ax1.set_title("Attack flows per window")
    ax1.set_xticks(range(N_WINDOWS))

    ax2.bar(grp.index, grp["rate"], color=COLOR_ATTACK)
    ax2.set_xlabel("Window ID (1-min bin)")
    ax2.set_ylabel("Attack rate (%)")
    ax2.set_title("Attack density per window")
    ax2.set_xticks(range(N_WINDOWS))

    _save(fig, out / "02_attack_density_per_window.png")


# ── Plot 3: Burst structure (ramp-up / peak / taper) ──────────────────────────

def plot_burst_phases(audit: pd.DataFrame, out: Path):
    """
    Reduced schema has no millisecond timestamp — we use row_in_window
    (an integer 0..99999) as the time axis, since rows arrive in chronological
    order. Same shape as a real time axis.
    """
    if "burst_phase" not in audit.columns or "row_in_window" not in audit.columns:
        return
    mask = audit["is_seeded_ddos"] == 1
    if mask.sum() == 0:
        return

    sub = audit.loc[mask, ["row_in_window", "burst_id", "burst_phase"]].copy()

    phase_order = ["ramp_up", "peak", "taper"]
    phase_color = {"ramp_up": "#F2A623", "peak": "#D85A30", "taper": "#A04020"}

    fig, ax = plt.subplots(figsize=(11, 4))
    burst_ids = sorted(sub["burst_id"].dropna().unique())

    for burst in burst_ids:
        bsub = sub[sub["burst_id"] == burst]
        for phase in phase_order:
            psub = bsub[bsub["burst_phase"] == phase]
            if len(psub) == 0:
                continue
            ax.scatter(
                psub["row_in_window"],
                np.full(len(psub), int(burst)),
                color=phase_color[phase],
                s=12, alpha=0.6,
                label=phase if burst == burst_ids[0] else None,
            )

    ax.set_xlabel("row_in_window (proxy for time order)")
    ax.set_ylabel("Burst ID")
    ax.set_title("Burst phases over time")
    ax.set_yticks([int(b) for b in burst_ids])
    ax.legend(loc="upper right")

    for w in range(1, N_WINDOWS):
        ax.axvline(w * ROWS_PER_WIN, color="gray", lw=0.4, alpha=0.5)

    _save(fig, out / "03_burst_phases.png")


# ── Plot 4: Feature distributions, attack vs benign ───────────────────────────

def plot_feature_distributions(df: pd.DataFrame, audit: pd.DataFrame, out: Path):
    """
    Distribution of key flow-level features, benign vs DDoS.
    phase0 already applies log1p to heavy-tailed features.
    """
    available = [c for c in HEAVY_FEATURES if c in df.columns]
    n = len(available)
    if n == 0:
        return

    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3))
    axes = np.array(axes).flatten()

    is_attack = audit["is_seeded_ddos"].values == 1

    for i, feat in enumerate(available):
        ax     = axes[i]
        vals_b = df.loc[~is_attack, feat].values
        vals_a = df.loc[is_attack,  feat].values

        lo = float(np.nanmin([vals_b.min(), vals_a.min() if len(vals_a) else np.inf]))
        hi = float(np.nanmax([vals_b.max(), vals_a.max() if len(vals_a) else -np.inf]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            ax.axis("off")
            continue
        bins = np.linspace(lo, hi, 40)

        ax.hist(vals_b, bins=bins, alpha=0.6, color=COLOR_BENIGN,
                label="Benign", density=True)
        if len(vals_a) > 0:
            ax.hist(vals_a, bins=bins, alpha=0.6, color=COLOR_ATTACK,
                    label="DDoS", density=True)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("log1p(value)")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature distributions: benign vs DDoS (log1p scale)", y=1.02)
    _save(fig, out / "04_feature_distributions.png")


# ── Plot 5: Aggregated window-level features over time ────────────────────────

def plot_window_aggregates(df: pd.DataFrame, audit: pd.DataFrame, out: Path):
    """
    The view your LSTM will see: 15 window-level points per dataset.
    Aggregations are computed on the fly here from the cleaned CSV.
    """
    df = df.copy()
    df["is_attack"] = audit["is_seeded_ddos"].values

    def shannon_entropy(s):
        p = s.value_counts(normalize=True)
        return float(-(p * np.log2(p + 1e-12)).sum())

    agg = df.groupby("window_id").agg(
        flow_count          = ("is_attack",                "size"),
        attack_count        = ("is_attack",                "sum"),
        mean_total_bytes    = ("total_bytes",              "mean"),
        mean_duration       = ("duration",                 "mean"),
        mean_pkts_per_sec   = ("packets_per_second",       "mean"),
        mean_pkt_size       = ("packet_size_avg",          "mean"),
        mean_outbound_ratio = ("outbound_byte_ratio",      "mean"),
    )

    agg["unique_src_ip"] = df.groupby("window_id")["src_ip"].nunique()
    agg["unique_dst_ip"] = df.groupby("window_id")["dst_ip"].nunique()
    agg["entropy_src"]   = df.groupby("window_id")["src_ip"].apply(shannon_entropy)
    agg["entropy_dst"]   = df.groupby("window_id")["dst_ip"].apply(shannon_entropy)

    metrics = [
        "flow_count",
        "unique_src_ip", "unique_dst_ip",
        "entropy_src",   "entropy_dst",
        "mean_total_bytes", "mean_duration",
        "mean_pkts_per_sec", "mean_pkt_size",
        "mean_outbound_ratio",
    ]
    metrics = [m for m in metrics if m in agg.columns]

    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 2.8))
    axes = np.array(axes).flatten()

    attack_windows = agg.index[agg["attack_count"] > 0].tolist()

    for i, m in enumerate(metrics):
        ax = axes[i]
        ax.plot(agg.index, agg[m], "o-", color=COLOR_BENIGN, lw=1.5)
        for w in attack_windows:
            ax.axvspan(w - 0.4, w + 0.4, color=COLOR_ATTACK, alpha=0.2)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Window ID")
        ax.set_xticks(range(N_WINDOWS))

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Window-level aggregates (orange = attack windows)", y=1.02)
    _save(fig, out / "05_window_aggregates.png")


# ── Plot 6: Source IP distribution in attack windows ──────────────────────────

def plot_src_ip_distribution(df: pd.DataFrame, audit: pd.DataFrame, out: Path):
    if "src_ip" not in df.columns:
        return

    df = df.copy()
    df["is_attack"] = audit["is_seeded_ddos"].values

    attack_windows = (
        df.groupby("window_id")["is_attack"].sum()
        .pipe(lambda s: s[s > 0].index.tolist())
    )
    if not attack_windows:
        return

    n   = min(len(attack_windows), 4)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 3.8))
    if n == 1:
        axes = [axes]

    for ax, wid in zip(axes, attack_windows[:n]):
        sub        = df[df["window_id"] == wid]
        benign_ips = sub[sub["is_attack"] == 0]["src_ip"].value_counts().head(15)
        attack_ips = sub[sub["is_attack"] == 1]["src_ip"].value_counts().head(15)

        all_ips = pd.concat([benign_ips, attack_ips]).index.unique()
        bvals   = [benign_ips.get(ip, 0) for ip in all_ips]
        avals   = [attack_ips.get(ip, 0) for ip in all_ips]

        idx = np.arange(len(all_ips))
        ax.bar(idx, bvals, color=COLOR_BENIGN, label="benign flows")
        ax.bar(idx, avals, bottom=bvals, color=COLOR_ATTACK, label="attack flows")
        ax.set_title(f"Window {wid}", fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel("Flow count" if wid == attack_windows[0] else "")
        if wid == attack_windows[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Top source IPs in attack windows", y=1.02)
    _save(fig, out / "06_src_ip_distribution.png")


# ── Plot 7: Correlation matrix of numeric features ────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, out: Path):
    """
    Reduced schema is small (~14 features) so the correlation matrix is
    compact and very readable. Useful as a sanity check.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    drop = ["window_id", "src_port", "dst_port"]
    num = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    if num.shape[1] < 2:
        return

    corr = num.corr().fillna(0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, fontsize=9, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation matrix")
    _save(fig, out / "07_correlation_matrix.png")


# ── Plot 8: Protocol distribution ─────────────────────────────────────────────

def plot_protocol_breakdown(df: pd.DataFrame, audit: pd.DataFrame, out: Path):
    """
    Reduced schema stores `protocol` as a string (TCP / UDP / ICMP / ...).
    phase0 should one-hot encode it into proto_TCP, proto_UDP, proto_ICMP,
    proto_other. We support both forms here.
    """
    proto_cols = [c for c in df.columns if c.startswith("proto_")]
    is_attack  = audit["is_seeded_ddos"].values == 1

    if proto_cols:
        b = df.loc[~is_attack, proto_cols].sum() / max((~is_attack).sum(), 1)
        a = df.loc[ is_attack, proto_cols].sum() / max(is_attack.sum(),  1)
        labels = [c.replace("proto_", "") for c in proto_cols]
    elif "protocol" in df.columns:
        top = df["protocol"].value_counts().head(4).index.tolist()
        b = pd.Series({p: (df.loc[~is_attack, "protocol"] == p).mean() for p in top})
        a = pd.Series({p: (df.loc[ is_attack, "protocol"] == p).mean() for p in top})
        labels = [str(p) for p in top]
    else:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, b.values, width=0.4, color=COLOR_BENIGN, label="Benign")
    ax.bar(x + 0.2, a.values, width=0.4, color=COLOR_ATTACK, label="DDoS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of flows")
    ax.set_title("Protocol distribution")
    ax.legend()
    _save(fig, out / "08_protocol_breakdown.png")


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_all_plots(clean_csv: str | Path,
                  audit_csv: str | Path,
                  output_dir: str | Path) -> None:
    clean_csv  = Path(clean_csv)
    audit_csv  = Path(audit_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eda] Loading {clean_csv.name} + {audit_csv.name} ...")
    df    = pd.read_csv(clean_csv, low_memory=False)
    audit = pd.read_csv(audit_csv, low_memory=False)

    df = _check_window_id(df)
    df = _sanitize(df)

    print("[eda] Generating plots ...")
    plot_class_balance(audit, output_dir)
    plot_attack_density_per_window(df, audit, output_dir)
    plot_burst_phases(audit, output_dir)
    plot_feature_distributions(df, audit, output_dir)
    plot_window_aggregates(df, audit, output_dir)
    plot_src_ip_distribution(df, audit, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_protocol_breakdown(df, audit, output_dir)
    print(f"[eda] Done. Plots in {output_dir}")


def run_tree(cleaned_root: str | Path, output_root: str | Path) -> None:
    """Walk the cleaned-data tree, run all plots for every (clean, audit) pair."""
    cleaned_root = Path(cleaned_root)
    output_root  = Path(output_root)

    clean_files = sorted(cleaned_root.rglob("*_clean.csv"))
    print(f"[eda] Found {len(clean_files)} cleaned file(s) under {cleaned_root}")

    for clean in clean_files:
        audit = clean.with_name(clean.name.replace("_clean.csv", "_audit.csv"))
        if not audit.exists():
            print(f"[eda]  ! audit missing for {clean.name}, skipping")
            continue

        rel_dir = clean.parent.relative_to(cleaned_root)
        stem    = clean.name.replace("_clean.csv", "")
        out_dir = output_root / rel_dir / stem
        run_all_plots(clean, audit, out_dir)


if __name__ == "__main__":
    CLEANED_ROOT = "quantum-anomalydetection-ddos/cleaned_dataset/option_1/attack/train"
    OUTPUT_ROOT  = "quantum-anomalydetection-ddos/cleaned_dataset/option_1/attack/train/eda"
    run_tree(CLEANED_ROOT, OUTPUT_ROOT)