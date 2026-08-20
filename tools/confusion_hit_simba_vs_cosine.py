"""Confusion matrix: does SIMBA's retrieval hit@k agree with cosine's, per
validation query, for the same self-pair retrieval benchmark (see
tools/benchmark_self_retrieval.py)?

For every query molecule (>=2 spectra, ~2,010 of them), computes each
method's own rank of the true match among its 255 GT-MCES-nearest decoys +
itself, then crosses "SIMBA hit@k" (rank<=k) against "cosine hit@k" into a
2x2 table: both worked, SIMBA-only, cosine-only, neither -- counts and
percentages, printed and rendered as an annotated heatmap image.

Usage:
    uv run python tools/confusion_hit_simba_vs_cosine.py \\
        --exp_dir /path/to/experiments/training/012_..._1gpu \\
        --cosine_parquet /path/to/preprocessing_dir/val_cosine_val.parquet \\
        --val_name val \\
        --k 1 \\
        --n_decoys 255
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_self_retrieval import (
    build_pool_and_queries,
    build_score_matrix,
    latest_pred_col,
    true_match_scores,
)


def per_query_rank(
    gt_matrix,
    score_matrix,
    true_scores,
    mol_to_local,
    query_mols,
    n_decoys,
    higher_is_better,
) -> dict:
    ranks = {}
    for q in query_mols:
        qi = mol_to_local[q]
        if q not in true_scores or np.isnan(true_scores[q]):
            continue
        gt_row = gt_matrix[qi].copy()
        gt_row[qi] = np.inf
        decoy_local = np.argsort(gt_row)[:n_decoys]
        decoy_scores = score_matrix[qi, decoy_local]
        decoy_scores = decoy_scores[~np.isnan(decoy_scores)]
        candidates = np.append(decoy_scores, true_scores[q])
        true_idx = len(candidates) - 1
        order = np.argsort(-candidates if higher_is_better else candidates)
        rank = int(np.nonzero(order == true_idx)[0][0]) + 1
        ranks[q] = rank
    return ranks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--cosine_parquet", required=True)
    parser.add_argument("--val_name", default="val")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n_decoys", type=int, default=255)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    ref_path = exp_dir / f"val_pairs_{args.val_name}_consolidated.parquet"
    ref_df = pd.read_parquet(
        ref_path,
        columns=[
            "mol_idx_0",
            "mol_idx_1",
            "gt_mces",
            "mces_bin",
            "spec_idx_0",
            "spec_idx_1",
        ],
    )
    pool_mols, query_mols = build_pool_and_queries(ref_df)
    mol_to_local = {m: i for i, m in enumerate(pool_mols)}
    print(f"Pool: {len(pool_mols):,} molecules; queries: {len(query_mols):,}")
    gt_matrix = build_score_matrix(ref_df, pool_mols, mol_to_local, "gt_mces")

    pred_col = latest_pred_col(ref_path)
    print(f"SIMBA: {exp_dir.name} ({pred_col})")
    simba_df = pd.read_parquet(
        ref_path,
        columns=[
            "mol_idx_0",
            "mol_idx_1",
            "mces_bin",
            "spec_idx_0",
            "spec_idx_1",
            pred_col,
        ],
    )
    simba_matrix = build_score_matrix(simba_df, pool_mols, mol_to_local, pred_col)
    simba_true = true_match_scores(simba_df, query_mols, pred_col)
    simba_ranks = per_query_rank(
        gt_matrix,
        simba_matrix,
        simba_true,
        mol_to_local,
        query_mols,
        args.n_decoys,
        higher_is_better=False,
    )

    print(f"Cosine: {args.cosine_parquet}")
    cos = pd.read_parquet(args.cosine_parquet)
    cos_df = ref_df[
        ["mol_idx_0", "mol_idx_1", "mces_bin", "spec_idx_0", "spec_idx_1"]
    ].merge(cos, on=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"], how="left")
    cos_matrix = build_score_matrix(cos_df, pool_mols, mol_to_local, "cosine")
    cos_true = true_match_scores(cos_df, query_mols, "cosine")
    cos_ranks = per_query_rank(
        gt_matrix,
        cos_matrix,
        cos_true,
        mol_to_local,
        query_mols,
        args.n_decoys,
        higher_is_better=True,
    )

    common = sorted(set(simba_ranks) & set(cos_ranks))
    dropped = len(query_mols) - len(common)
    if dropped:
        print(f"  ({dropped} queries missing a score for one method, excluded)")

    k = args.k
    simba_hit = np.array([simba_ranks[q] <= k for q in common])
    cos_hit = np.array([cos_ranks[q] <= k for q in common])
    n = len(common)

    both = int(np.sum(simba_hit & cos_hit))
    simba_only = int(np.sum(simba_hit & ~cos_hit))
    cos_only = int(np.sum(~simba_hit & cos_hit))
    neither = int(np.sum(~simba_hit & ~cos_hit))
    assert both + simba_only + cos_only + neither == n

    labels = [f"cosine hit@{k}", f"cosine miss@{k}"]
    rows = [f"SIMBA hit@{k}", f"SIMBA miss@{k}"]
    counts = np.array([[both, simba_only], [cos_only, neither]])
    pct = 100 * counts / n

    print(f"\n=== SIMBA ({exp_dir.name}) vs cosine, Hit@{k}, n={n:,} queries ===")
    header = f"{'':20s}{labels[0]:>18s}{labels[1]:>18s}"
    print(header)
    for r_label, row_counts, row_pct in zip(rows, counts, pct):
        cells = "  ".join(f"n={c:,} ({p:.1f}%)" for c, p in zip(row_counts, row_pct))
        print(f"{r_label:20s}{cells}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(rows)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"n={counts[i, j]:,}\n({pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if counts[i, j] > counts.max() / 2 else "black",
            )
    ax.set_title(f"{exp_dir.name}\nSIMBA vs cosine -- Hit@{k} agreement (n={n:,})")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()

    output = Path(args.output or exp_dir / f"confusion_hit{k}_simba_vs_cosine.png")
    fig.savefig(output, dpi=150)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
