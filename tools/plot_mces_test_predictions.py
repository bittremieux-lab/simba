"""
Plot MCES prediction quality on the MSG official test set.

Two-panel figure:
  Left  — box plot: predicted MCES vs true MCES (rounded bins, whiskers 5–95 pct)
  Right — training pair MCES distribution (log-scale bar chart)

Usage:
  uv run python tools/plot_mces_test_predictions.py \
      --csv  /mnt/data2/.../msg_official_splits/test_predictions.csv \
      --train /mnt/data2/.../preprocessing_msg_official/ed_mces_indexes_tani_incremental_train_node0_chunk0.npy \
      --out  mces_analysis.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CSV = (
    "/mnt/data2/nkubrakov/experiments_3_dataset/training/"
    "msg_official_splits/test_predictions.csv"
)
DEFAULT_TRAIN_NPY = (
    "/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_official/"
    "ed_mces_indexes_tani_incremental_train_node0_chunk0.npy"
)
DEFAULT_OUT = "mces_analysis.png"


def load_test(csv_path: str):
    df = pd.read_csv(csv_path)
    df["bin"] = df["mces_true"].round().astype(int).clip(0, 20)
    return df


def load_train_dist(npy_path: str):
    arr = np.load(npy_path)
    mces = arr[:, 3]  # raw MCES column
    bins = np.round(mces).astype(int).clip(0, 20)
    counts = np.bincount(bins, minlength=21)
    return counts


def build_box_data(df: pd.DataFrame):
    """Return list of dicts with percentile stats per integer MCES bin."""
    rows = []
    for b in range(21):
        g = df.loc[df["bin"] == b, "mces_pred"].values
        if len(g) == 0:
            continue
        p5, q1, med, q3, p95 = np.percentile(g, [5, 25, 50, 75, 95])
        rows.append(
            {
                "bin": b,
                "n": len(g),
                "p5": p5,
                "q1": q1,
                "med": med,
                "q3": q3,
                "p95": p95,
            }
        )
    return rows


def draw_boxplot(ax, box_data, r, mae):
    BLUE = "#3b82f6"
    AMBER = "#f59e0b"
    SPARSE_BLUE = "#93c5fd"

    bins = [d["bin"] for d in box_data]

    ax.set_facecolor("#111827")
    ax.grid(axis="y", color="#1e293b", linewidth=0.8, zorder=0)

    # y = x reference
    ax.plot(
        [0, 20],
        [0, 20],
        color="#374151",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
        label="y = x",
    )

    BOX_W = 0.55
    for d in box_data:
        x = d["bin"]
        sparse = d["n"] < 500
        color = SPARSE_BLUE if sparse else BLUE
        alpha = 0.55 if sparse else 0.82

        # whisker
        ax.plot(
            [x, x],
            [d["p5"], d["p95"]],
            color="#94a3b8",
            linewidth=1.0,
            alpha=alpha * 0.8,
            zorder=2,
        )
        # caps
        for cap_y in (d["p5"], d["p95"]):
            ax.plot(
                [x - BOX_W * 0.3, x + BOX_W * 0.3],
                [cap_y, cap_y],
                color="#94a3b8",
                linewidth=1.0,
                alpha=alpha * 0.8,
                zorder=2,
            )
        # IQR box
        ax.bar(
            x,
            d["q3"] - d["q1"],
            bottom=d["q1"],
            width=BOX_W,
            color=color,
            alpha=alpha,
            zorder=3,
            linewidth=0.8 if sparse else 0,
            edgecolor=SPARSE_BLUE if sparse else "none",
        )
        # median
        ax.plot(
            [x - BOX_W / 2, x + BOX_W / 2],
            [d["med"], d["med"]],
            color=AMBER,
            linewidth=2.0,
            zorder=4,
        )

    # sparse / dense separator
    ax.axvline(9.5, color="#374151", linewidth=0.8, linestyle=":", zorder=1)

    # x-axis tick labels: bin + n=
    ax.set_xticks(bins)
    labels = []
    for d in box_data:
        n_str = f"{d['n'] / 1000:.0f}k" if d["n"] >= 1000 else str(d["n"])
        labels.append(f"{d['bin']}\nn={n_str}")
    ax.set_xticklabels(labels, fontsize=7, color="#94a3b8")

    ax.set_xlim(-0.8, 20.8)
    ax.set_ylim(-0.5, 22)
    ax.set_xlabel("True MCES (rounded)", color="#94a3b8", fontsize=9, labelpad=6)
    ax.set_ylabel("Predicted MCES", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#475569", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    ax.set_title(
        f"Predicted vs True MCES — test set\n"
        f"Spearman ρ = {r:.3f}   MAE = {mae:.2f} MCES units",
        color="#e2e8f0",
        fontsize=10,
        pad=10,
    )

    # legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=BLUE, alpha=0.82, label="IQR box (Q1–Q3)"),
        Line2D([0], [0], color=AMBER, linewidth=2, label="Median"),
        Line2D([0], [0], color="#374151", linewidth=1.2, linestyle="--", label="y = x"),
        Patch(
            facecolor=SPARSE_BLUE,
            alpha=0.55,
            edgecolor=SPARSE_BLUE,
            linewidth=0.8,
            label="Sparse bins (n < 500)",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=7.5,
        framealpha=0.15,
        facecolor="#1e293b",
        edgecolor="#374151",
        labelcolor="#94a3b8",
    )

    ax.annotate(
        "whiskers = 5th–95th pct · no outliers",
        xy=(0.99, 0.01),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#475569",
    )


def draw_train_dist(ax, train_counts):
    bins = np.arange(21)
    colors = ["#60a5fa" if c >= 500 else "#93c5fd" for c in train_counts]

    ax.set_facecolor("#111827")
    ax.grid(axis="y", color="#1e293b", linewidth=0.8, zorder=0)

    ax.bar(bins, train_counts, color=colors, alpha=0.80, zorder=3, width=0.7)
    ax.axvline(9.5, color="#374151", linewidth=0.8, linestyle=":", zorder=4)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: (
                f"{x / 1e6:.1f}M"
                if x >= 1e6
                else (f"{x / 1e3:.0f}k" if x >= 1e3 else str(int(x)))
            )
        )
    )

    ax.set_xticks(bins)
    ax.set_xticklabels([str(b) for b in bins], fontsize=7, color="#94a3b8")
    ax.set_xlabel("True MCES (rounded)", color="#94a3b8", fontsize=9, labelpad=6)
    ax.set_ylabel("Number of training pairs (log scale)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#475569", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    total = train_counts.sum()
    low = train_counts[:10].sum()
    high = train_counts[10:].sum()
    ax.set_title(
        f"Training pair MCES distribution\n"
        f"MCES 0–9: {low:,} pairs ({100 * low / total:.1f}%)   "
        f"MCES 10–20: {high:,} pairs ({100 * high / total:.1f}%)",
        color="#e2e8f0",
        fontsize=10,
        pad=10,
    )

    ax.annotate(
        "← sparse        dense →",
        xy=(9.5, ax.get_ylim()[0] * 3),
        xycoords="data",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#4b5563",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--train", default=DEFAULT_TRAIN_NPY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    print("Loading test predictions...")
    df = load_test(args.csv)
    r, _ = spearmanr(df["mces_true"], df["mces_pred"])
    mae = (df["mces_true"] - df["mces_pred"]).abs().mean()
    box_data = build_box_data(df)

    print("Loading training distribution...")
    train_counts = load_train_dist(args.train)

    plt.style.use("dark_background")
    fig, (ax_box, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        facecolor="#0d1117",
        gridspec_kw={"width_ratios": [3, 2]},
    )
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.16)

    draw_boxplot(ax_box, box_data, r, mae)
    draw_train_dist(ax_dist, train_counts)

    fig.suptitle(
        "SIMBA · MSG official splits · best_model-v1.ckpt",
        color="#64748b",
        fontsize=9,
        y=0.97,
    )

    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
