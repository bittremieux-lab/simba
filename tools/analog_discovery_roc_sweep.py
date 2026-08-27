"""014_2 analog discovery: ROC curves swept across raw-MCES "true analog"
thresholds 3..13 (one grid figure, one subplot per threshold), for ONE
search. Companion to the single-threshold ROC panel in
tools/analog_discovery_analyze.py -- reuses that script's load_scores/
build_gt_matrix so the underlying data is identical, just re-binarized at
each threshold instead of only --roc_threshold's single value.

Usage:
    uv run python tools/analog_discovery_roc_sweep.py \\
        --scores_dir data/analog_discovery/search_A_scores \\
        --exact_mces_dir data/analog_discovery/search_A_exact_mces \\
        --output_dir data/analog_discovery/search_A_results \\
        --search_label "Search A (NIST20 + MassSpecGym)"
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


THRESHOLDS = list(range(3, 14))  # 3..13 inclusive


def run(scores_dir: str, exact_mces_dir: str, output_dir: str, search_label: str = ""):
    scores_dir, exact_mces_dir, output_dir = (
        Path(scores_dir),
        Path(exact_mces_dir),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {scores_dir} ...")
    q_smiles, lib_smiles, scores = load_scores(scores_dir)
    print(f"Loading exact GT MCES from {exact_mces_dir} ...")
    gt = build_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)

    valid = ~np.isnan(gt)
    gt_valid = gt[valid]
    scores_valid = {m: scores[m][valid] for m in METHODS}
    print(f"  {valid.sum():,} resolved pairs pooled across all thresholds")

    n = len(THRESHOLDS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.ravel()

    for idx, thr in enumerate(THRESHOLDS):
        ax = axes[idx]
        labels = (gt_valid < thr).astype(np.int32)
        pct_pos = 100 * labels.mean()
        for m in METHODS:
            pred = -scores_valid[m]
            fpr, tpr, _ = roc_curve(labels, pred)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=f"{METHOD_LABELS[m]} (AUC={roc_auc:.3f})",
                color=METHOD_COLORS[m],
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(f"threshold={thr}  ({pct_pos:.2f}% label=1)", fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower right")
        print(f"  threshold={thr}: {pct_pos:.3f}% positive")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"ROC sweep -- {search_label}" if search_label else "ROC sweep")
    fig.tight_layout()
    out_path = output_dir / "analog_discovery_roc_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores_dir", required=True)
    p.add_argument("--exact_mces_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--search_label", default="")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
