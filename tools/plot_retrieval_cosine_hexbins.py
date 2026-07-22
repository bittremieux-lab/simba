"""
3-panel hexbin: GT MCES / pred MCES / calibration error vs spectral cosine similarity
for SIMBA's retrieval picks (test spectrum vs SIMBA-picked training spectrum).

Reads retrieval_diagnostics_*.csv which must have simba_spectral_cos column
(added by the updated diagnose_retrieval.py).

Usage:
    uv run python tools/plot_retrieval_cosine_hexbins.py \
        --csv results/retrieval_diagnostics_bs2048_v2_step44k.csv \
        --output results/retrieval_cosine_hexbins_bs2048_v2_step44k.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


BINS_COS = np.linspace(0, 1, 41)
BINS_MCES = np.arange(0, 42.5, 2.5)
GREEN = "#E07B54"  # orange to distinguish from val hexbins


def one_panel(fig, subplot_spec, x, y, title, ylabel, cmap, y_bins, y_lim, yline=None):
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
    ax_main.set_xlabel("Spectral cosine (test vs SIMBA pick)", fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=9)
    ax_main.tick_params(labelsize=8)
    ax_main.grid(True, alpha=0.15)

    # Top: cosine distribution
    ax_top.hist(x, bins=BINS_COS, color=GREEN, edgecolor="none")
    ax_top.set_xlim(0, 1)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(labelsize=7)
    ax_top.set_title(title, fontsize=9, fontweight="bold")
    ax_top.grid(True, alpha=0.15)

    # Right: y distribution
    ax_right.hist(
        y, bins=y_bins, orientation="horizontal", color=GREEN, edgecolor="none"
    )
    ax_right.set_ylim(*y_lim)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(labelsize=7)
    ax_right.grid(True, alpha=0.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/retrieval_cosine_hexbins.png")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Loaded {len(rows):,} rows")

    covered = np.array([r["covered"] == "True" for r in rows])
    cos = np.array([float(r["simba_spectral_cos"]) for r in rows])[covered]
    gt = np.array([float(r["simba_gt_mces"]) for r in rows])[covered]
    pred = np.array([float(r["simba_pred_mces"]) for r in rows])[covered]
    err = gt - pred

    n = covered.sum()
    print(f"  covered: {n:,}  cos mean={cos.mean():.3f}  median={np.median(cos):.3f}")
    print(
        f"  gt_mces mean={gt.mean():.2f}  pred_mces mean={pred.mean():.2f}  err mean={err.mean():.2f}"
    )

    err_lim = (float(np.percentile(err, 1)), float(np.percentile(err, 99)))

    fig = plt.figure(figsize=(22, 7))
    outer = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.38)

    one_panel(
        fig,
        outer[0, 0],
        cos,
        gt,
        f"GT MCES vs spectral cosine  (n={n:,})",
        "GT MCES (SIMBA pick)",
        "Blues",
        BINS_MCES,
        (0, 40),
    )

    one_panel(
        fig,
        outer[0, 1],
        cos,
        pred,
        "Pred MCES vs spectral cosine",
        "Pred MCES (SIMBA output)",
        "Purples",
        BINS_MCES,
        (0, 40),
    )

    one_panel(
        fig,
        outer[0, 2],
        cos,
        err,
        "Calibration error vs spectral cosine",
        "Error = GT − pred MCES",
        "RdYlGn_r",
        np.linspace(err_lim[0], err_lim[1], 41),
        err_lim,
        yline=0,
    )

    fig.suptitle(
        "Retrieval pairs: test spectrum vs SIMBA pick — bs2048_v2 · step 44k\n"
        "x = spectral cosine similarity between test and SIMBA-picked training spectrum",
        fontsize=11,
    )
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
