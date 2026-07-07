"""
Build a 4-panel balanced hexbin figure from val prediction CSVs.
Panels: rows = linear / log scale; cols = scaffold / official val set.
Balancing: equal number of pairs sampled per 2.5-unit GT-MCES bin.

Usage:
    uv run python tools/plot_val_hexbin_balanced.py \
        --val_dir /path/to/val_hexbin \
        --output   /path/to/val_hexbin/mces_hexbin_balanced.png
"""

import argparse
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BIN_STEP = 2.5
BINS = np.arange(0, 40 + BIN_STEP, BIN_STEP)  # 17 edges → 16 bins
GREEN = "#4E9A7A"


def balance_by_gt(gt_mces: np.ndarray, pred_mces: np.ndarray, rng: np.random.Generator):
    """Sample each bin to at most 3× the smallest non-empty bin count."""
    bin_ids = np.digitize(gt_mces, BINS) - 1
    bin_ids = np.clip(bin_ids, 0, len(BINS) - 2)
    counts = np.bincount(bin_ids, minlength=len(BINS) - 1)
    min_count = counts[counts > 0].min()
    cap = int(1.5 * min_count)
    idx = []
    for b in range(len(BINS) - 1):
        where = np.where(bin_ids == b)[0]
        if len(where) == 0:
            continue
        n = min(len(where), cap)
        chosen = rng.choice(where, size=n, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    return gt_mces[idx], pred_mces[idx], len(idx)


def one_panel(ax_main, ax_top, ax_right, gt, pred, title, scale):
    r, _ = spearmanr(gt, pred)
    mae = float(np.mean(np.abs(pred - gt)))

    bins_arg = "log" if scale == "log" else None
    ax_main.hexbin(
        gt,
        pred,
        gridsize=16,
        cmap="Greens",
        mincnt=1,
        bins=bins_arg,
        extent=[0, 40, 0, 40],
    )
    ax_main.plot([0, 40], [0, 40], "r--", lw=1)
    ax_main.set_xlim(0, 40)
    ax_main.set_ylim(0, 40)
    ax_main.grid(True, alpha=0.2)
    ax_main.set_xlabel("GT MCES", fontsize=9)
    ax_main.set_ylabel("Pred MCES", fontsize=9)
    ax_main.tick_params(labelsize=7)
    ax_main.set_title(
        f"{title}\nρ={r:.3f}  MAE={mae:.2f} MCES  [{'log' if scale == 'log' else 'linear'}]",
        fontsize=9,
    )

    ax_top.hist(gt, bins=BINS, color=GREEN, edgecolor="none")
    ax_top.set_xlim(0, 40)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(labelsize=6)
    ax_top.grid(True, alpha=0.2)

    ax_right.hist(
        pred, bins=BINS, orientation="horizontal", color=GREEN, edgecolor="none"
    )
    ax_right.set_ylim(0, 40)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(labelsize=6)
    ax_right.grid(True, alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = args.output or os.path.join(
        args.val_dir, "mces_hexbin_balanced_4panel.png"
    )

    available = []
    sets = {}
    for name in ("official", "test", "scaffold"):
        path = os.path.join(args.val_dir, f"val_predictions_{name}.csv")
        if not os.path.exists(path):
            print(f"  {name}: file not found, skipping")
            continue
        df = pd.read_csv(path)
        gt = df["mces_target_raw"].values.astype(np.float32)
        pred = df["mces_pred_raw"].values.astype(np.float32)
        gt_b, pred_b, n = balance_by_gt(gt, pred, rng)
        print(
            f"{name}: {len(gt):,} total → {n:,} balanced ({len(BINS) - 1} bins × {n // (len(BINS) - 1):,} each)"
        )
        sets[name] = (gt_b, pred_b)
        available.append(name)

    n_cols = len(available)
    # 2 rows (linear, log) × n_cols
    fig = plt.figure(figsize=(6.5 * n_cols, 12))
    outer = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.40, wspace=0.30)

    scales = ["linear", "log"]
    col_names = available

    for row, scale in enumerate(scales):
        for col, name in enumerate(col_names):
            gt, pred = sets[name]
            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer[row, col],
                width_ratios=[4, 1],
                height_ratios=[1, 4],
                hspace=0.04,
                wspace=0.04,
            )
            ax_top = fig.add_subplot(inner[0, 0])
            ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top)
            ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
            fig.add_subplot(inner[0, 1]).set_visible(False)

            one_panel(
                ax_main, ax_top, ax_right, gt, pred, title=f"{name} val", scale=scale
            )

    fig.suptitle(
        "MCES hexbins — step-61k checkpoint (GT-balanced, 2.5-unit bins)", fontsize=11
    )
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
