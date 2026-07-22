"""
9-panel hexbin figure: GT MCES vs cosine, pred MCES vs cosine, and error vs cosine
for the three splits (scaffold val, official val, test).

Uses the same GT-MCES-balanced sampling as plot_val_hexbin_balanced.py.
Each panel has top (cosine distribution) and right (y-axis distribution) marginals.

Usage:
    uv run python tools/plot_cosine_hexbins.py \
        --hexbin_dir /mnt/data/.../val_hexbin_step44k \
        --output results/cosine_hexbins_bs2048_v2_step44k.png
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = [
    ("scaffold", "Val scaffold"),
    ("official", "Val official"),
    ("test", "Test official"),
]

BIN_STEP = 2.5
BINS_MCES = np.arange(0, 40 + BIN_STEP, BIN_STEP)
BINS_COS = np.linspace(0, 1, 41)
GREEN = "#4E9A7A"


def balance_by_gt(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bin_ids = np.clip(np.digitize(gt, BINS_MCES) - 1, 0, len(BINS_MCES) - 2)
    counts = np.bincount(bin_ids, minlength=len(BINS_MCES) - 1)
    cap = int(1.5 * counts[counts > 0].min())
    idx = []
    for b in range(len(BINS_MCES) - 1):
        where = np.where(bin_ids == b)[0]
        if len(where):
            idx.append(rng.choice(where, size=min(len(where), cap), replace=False))
    return np.concatenate(idx)


def one_panel(
    fig,
    subplot_spec,
    x,
    y,
    split_label,
    col_title,
    cmap,
    x_bins,
    y_bins,
    y_lim,
    yline=None,
):
    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
    fig.add_subplot(inner[0, 1]).set_visible(False)

    # Main hexbin
    ax_main.hexbin(
        x, y, gridsize=35, cmap=cmap, mincnt=1, extent=[0, 1, y_lim[0], y_lim[1]]
    )
    if yline is not None:
        ax_main.axhline(yline, color="k", lw=0.9, ls="--", alpha=0.7)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(*y_lim)
    ax_main.set_xlabel("Spectral cosine similarity", fontsize=8)
    ax_main.set_ylabel(col_title, fontsize=8)
    ax_main.tick_params(labelsize=7)
    ax_main.grid(True, alpha=0.15)

    # Top marginal: cosine distribution
    ax_top.hist(x, bins=x_bins, color=GREEN, edgecolor="none")
    ax_top.set_xlim(0, 1)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(labelsize=6)
    ax_top.set_title(f"{split_label} · {col_title}", fontsize=7, fontweight="bold")
    ax_top.grid(True, alpha=0.15)

    # Right marginal: y distribution
    ax_right.hist(
        y, bins=y_bins, orientation="horizontal", color=GREEN, edgecolor="none"
    )
    ax_right.set_ylim(*y_lim)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(labelsize=6)
    ax_right.grid(True, alpha=0.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hexbin_dir", required=True)
    p.add_argument("--output", default="results/cosine_hexbins.png")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    hexbin_dir = Path(args.hexbin_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 19))
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    for row, (split_key, split_label) in enumerate(SPLITS):
        csv_path = hexbin_dir / f"val_predictions_{split_key}.csv"
        if not csv_path.exists():
            print(f"  SKIP {split_key}: not found")
            continue
        df = pd.read_csv(csv_path)
        if "cosine_spectral" not in df.columns:
            print(f"  SKIP {split_key}: cosine_spectral column missing")
            continue

        gt_raw = df["mces_target_raw"].values.astype(np.float32)
        pred_raw = df["mces_pred_raw"].values.astype(np.float32)
        cosine = df["cosine_spectral"].values.astype(np.float32)
        error = gt_raw - pred_raw

        idx = balance_by_gt(gt_raw, rng)
        gt_b, pred_b = gt_raw[idx], pred_raw[idx]
        cos_b, err_b = cosine[idx], error[idx]
        print(
            f"  {split_label}: {len(df):,} → {len(idx):,} balanced  "
            f"cos mean={cos_b.mean():.3f}  err mean={err_b.mean():.2f}"
        )

        err_lim = (float(np.percentile(err_b, 1)), float(np.percentile(err_b, 99)))

        one_panel(
            fig,
            outer[row, 0],
            cos_b,
            gt_b,
            split_label,
            "GT MCES",
            "Blues",
            BINS_COS,
            BINS_MCES,
            (0, 40),
        )

        one_panel(
            fig,
            outer[row, 1],
            cos_b,
            pred_b,
            split_label,
            "Pred MCES",
            "Purples",
            BINS_COS,
            BINS_MCES,
            (0, 40),
        )

        one_panel(
            fig,
            outer[row, 2],
            cos_b,
            err_b,
            split_label,
            "Error (GT−pred MCES)",
            "RdYlGn_r",
            BINS_COS,
            np.linspace(err_lim[0], err_lim[1], 41),
            err_lim,
            yline=0,
        )

    fig.suptitle(
        "Spectral cosine hexbins — bs2048_v2 · step 44k  (GT-MCES balanced)\n"
        "x = peak-based spectral cosine similarity (1 Da bins, sqrt intensities)",
        fontsize=11,
    )
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
