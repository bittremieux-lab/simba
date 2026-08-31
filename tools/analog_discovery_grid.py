"""014_2 analog discovery: grid comparison across the CASMI-distance-
exclusion sweep (see NOTES_014_2_ANALOG_DISCOVERY.md), for ONE search (A or
B). One row per model (014_2 baseline with no exclusion, then increasing
exclusion thresholds), three columns:
  (a) boxplot (SIMBA-raw / SIMBA-CORN / Cosine / Oracle) -- same as
      analog_discovery_analyze.py's panel_boxplot
  (b) ROC at a single fixed threshold (default 10)
  (c) ranking-performance out to K=1000

All rows share the SAME exact_mces_dir -- the exact-MCES data doesn't
depend on which checkpoint scored the queries, only on the CASMI queries
and the reference library molecules, both fixed per search. Reuses
load_scores/build_gt_matrix from analog_discovery_analyze.py so the
underlying data/logic is identical to the single-model plots, just laid out
as a grid instead of separate files per model.

Usage:
    uv run python tools/analog_discovery_grid.py \\
        --search_label "Search A (NIST20 + MassSpecGym)" \\
        --exact_mces_dir data/analog_discovery/search_A_exact_mces \\
        --output data/analog_discovery/search_A_grid.png \\
        --row "014_2 (no exclusion)=data/analog_discovery/search_A_scores" \\
        --row "excl4=data/analog_discovery/search_A_scores_excl4" \\
        --row "excl6=data/analog_discovery/search_A_scores_excl6" \\
        ...
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analog_discovery_analyze import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    build_gt_matrix,
    load_scores,
)
from sklearn.metrics import auc, roc_curve


def row_boxplot(ax, scores: dict, gt: np.ndarray, top_k: int, row_label: str):
    n_q = gt.shape[0]
    data, labels = [], []
    for m in METHODS:
        best = []
        for i in range(n_q):
            top = np.argsort(scores[m][i])[:top_k]
            vals = gt[i, top]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                best.append(float(vals.min()))
        data.append(best)
        labels.append(METHOD_LABELS[m])

    oracle = []
    for i in range(n_q):
        row = gt[i]
        if not np.all(np.isnan(row)):
            oracle.append(float(np.nanmin(row)))
    data.append(oracle)
    labels.append("Oracle")

    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(f"{row_label}\nbest MCES (top-{top_k})", fontsize=9)
    ax.tick_params(labelsize=7)


def row_roc(ax, scores: dict, gt: np.ndarray, roc_threshold: float):
    valid = ~np.isnan(gt)
    labels = (gt[valid] < roc_threshold).astype(np.int32)
    pct_pos = 100 * labels.mean()
    for m in METHODS:
        pred = -scores[m][valid]
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{METHOD_LABELS[m]} (AUC={roc_auc:.3f})",
            color=METHOD_COLORS[m],
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"threshold={roc_threshold:g}  ({pct_pos:.2f}% label=1)", fontsize=9)
    ax.set_xlabel("FPR", fontsize=8)
    ax.set_ylabel("TPR", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")


def row_ranking(ax, scores: dict, gt: np.ndarray, max_k: int):
    n_q = gt.shape[0]
    ks = np.arange(1, max_k + 1)
    for m in METHODS:
        ranks = []
        for i in range(n_q):
            row = gt[i]
            if np.all(np.isnan(row)):
                continue
            best_j = int(np.nanargmin(row))
            order = np.argsort(scores[m][i])
            ranks.append(int(np.nonzero(order == best_j)[0][0]) + 1)
        ranks = np.asarray(ranks)
        cum_frac = [(ranks <= k).mean() for k in ks]
        ax.plot(ks, cum_frac, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
    ax.set_xlabel("K", fontsize=8)
    ax.set_ylabel("Fraction with true best in top-K", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")


def run(
    rows: list[tuple[str, str]],
    exact_mces_dir: str,
    output: str,
    search_label: str = "",
    top_k: int = 10,
    roc_threshold: float = 10.0,
    max_rank_k: int = 1000,
):
    exact_mces_dir = Path(exact_mces_dir)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for i, (label, scores_dir) in enumerate(rows):
        print(f"\n=== Row {i + 1}/{n_rows}: {label} ({scores_dir}) ===")
        q_smiles, lib_smiles, scores = load_scores(Path(scores_dir))
        gt = build_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)
        row_boxplot(axes[i, 0], scores, gt, top_k, label)
        row_roc(axes[i, 1], scores, gt, roc_threshold)
        row_ranking(axes[i, 2], scores, gt, max_rank_k)

    axes[0, 0].set_title("Retrieved-analog quality", fontsize=10)
    axes[0, 1].set_title(f"threshold={roc_threshold:g}", fontsize=10)
    axes[0, 2].set_title("Ranking performance", fontsize=10)

    fig.suptitle(search_label or "Analog discovery grid", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nWrote {output}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--row",
        action="append",
        required=True,
        dest="rows",
        metavar="LABEL=SCORES_DIR",
        help="Repeatable. One row per model, in top-to-bottom order.",
    )
    p.add_argument("--exact_mces_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--search_label", default="")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--roc_threshold", type=float, default=10.0)
    p.add_argument("--max_rank_k", type=int, default=1000)
    args = p.parse_args()

    rows = []
    for r in args.rows:
        label, _, scores_dir = r.partition("=")
        rows.append((label, scores_dir))

    run(
        rows=rows,
        exact_mces_dir=args.exact_mces_dir,
        output=args.output,
        search_label=args.search_label,
        top_k=args.top_k,
        roc_threshold=args.roc_threshold,
        max_rank_k=args.max_rank_k,
    )


if __name__ == "__main__":
    main()
