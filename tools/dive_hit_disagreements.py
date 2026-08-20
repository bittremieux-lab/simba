"""Dive into where SIMBA and cosine disagree on hit@1 (see
tools/confusion_hit_simba_vs_cosine.py): for each disagreement query, what
rank did the *losing* method give the true match, and what did it rank #1
instead (which decoy molecule, and its GT MCES to the query)?

Two groups:
  - "cosine right, SIMBA wrong": SIMBA's own rank for the true match, and
    the GT MCES of whatever SIMBA ranked #1 instead.
  - "SIMBA right, cosine wrong": same, but for cosine.

Usage:
    uv run python tools/dive_hit_disagreements.py \\
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


def per_query_detail(
    gt_matrix,
    score_matrix,
    true_scores,
    mol_to_local,
    pool_mols,
    query_mols,
    n_decoys,
    higher_is_better,
) -> dict:
    """query_mol -> {true_rank, true_score, top1_mol, top1_gt_mces, top1_score}."""
    details = {}
    for q in query_mols:
        qi = mol_to_local[q]
        if q not in true_scores or np.isnan(true_scores[q]):
            continue
        gt_row = gt_matrix[qi].copy()
        gt_row[qi] = np.inf
        decoy_local = np.argsort(gt_row)[:n_decoys]
        decoy_scores = score_matrix[qi, decoy_local]
        valid = ~np.isnan(decoy_scores)
        decoy_local = decoy_local[valid]
        decoy_scores = decoy_scores[valid]
        decoy_gt = gt_row[decoy_local]

        cand_scores = np.append(decoy_scores, true_scores[q])
        cand_mol = np.append(pool_mols[decoy_local], q)
        cand_gt = np.append(decoy_gt, 0.0)  # true match: same molecule, GT MCES 0

        order = np.argsort(-cand_scores if higher_is_better else cand_scores)
        true_idx = len(cand_scores) - 1
        true_rank = int(np.nonzero(order == true_idx)[0][0]) + 1

        top1_pos = order[0]
        details[q] = {
            "true_rank": true_rank,
            "true_score": float(true_scores[q]),
            "top1_mol": int(cand_mol[top1_pos]),
            "top1_gt_mces": float(cand_gt[top1_pos]),
            "top1_score": float(cand_scores[top1_pos]),
        }
    return details


def summarize(label: str, ranks: np.ndarray, top1_gt: np.ndarray):
    print(f"\n--- {label} (n={len(ranks):,}) ---")
    print(
        f"  True-match rank:  median={np.median(ranks):.0f}  mean={ranks.mean():.1f}  "
        f"min={ranks.min()}  max={ranks.max()}"
    )
    for hi in (2, 5, 10, 20, 50, 100, 256):
        pct = 100 * np.mean(ranks <= hi)
        print(f"    rank <= {hi:>3d}: {pct:5.1f}%")
    print(
        f"  GT MCES of the wrongly-top-ranked molecule: median={np.median(top1_gt):.2f}  "
        f"mean={top1_gt.mean():.2f}  min={top1_gt.min():.1f}  max={top1_gt.max():.1f}"
    )
    for lo, hi in [(0, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 40)]:
        pct = 100 * np.mean((top1_gt > lo) & (top1_gt <= hi))
        print(f"    GT MCES in ({lo},{hi}]: {pct:5.1f}%")


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
    simba_detail = per_query_detail(
        gt_matrix,
        simba_matrix,
        simba_true,
        mol_to_local,
        pool_mols,
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
    cos_detail = per_query_detail(
        gt_matrix,
        cos_matrix,
        cos_true,
        mol_to_local,
        pool_mols,
        query_mols,
        args.n_decoys,
        higher_is_better=True,
    )

    common = sorted(set(simba_detail) & set(cos_detail))
    k = args.k
    simba_hit = {q: simba_detail[q]["true_rank"] <= k for q in common}
    cos_hit = {q: cos_detail[q]["true_rank"] <= k for q in common}

    cosine_right_simba_wrong = [q for q in common if cos_hit[q] and not simba_hit[q]]
    simba_right_cosine_wrong = [q for q in common if simba_hit[q] and not cos_hit[q]]
    print(
        f"\ncosine right, SIMBA wrong: n={len(cosine_right_simba_wrong):,} | "
        f"SIMBA right, cosine wrong: n={len(simba_right_cosine_wrong):,}"
    )

    simba_wrong_ranks = np.array(
        [simba_detail[q]["true_rank"] for q in cosine_right_simba_wrong]
    )
    simba_wrong_top1_gt = np.array(
        [simba_detail[q]["top1_gt_mces"] for q in cosine_right_simba_wrong]
    )
    summarize(
        "SIMBA, on queries cosine got right", simba_wrong_ranks, simba_wrong_top1_gt
    )

    cos_wrong_ranks = np.array(
        [cos_detail[q]["true_rank"] for q in simba_right_cosine_wrong]
    )
    cos_wrong_top1_gt = np.array(
        [cos_detail[q]["top1_gt_mces"] for q in simba_right_cosine_wrong]
    )
    summarize("cosine, on queries SIMBA got right", cos_wrong_ranks, cos_wrong_top1_gt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(simba_wrong_ranks, bins=40, color="#1f77b4")
    axes[0, 0].set_title(
        "SIMBA's rank for the true match\n(317-like group: cosine right, SIMBA wrong)"
    )
    axes[0, 0].set_xlabel("rank (1=best)")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(simba_wrong_top1_gt, bins=40, color="#d62728")
    axes[0, 1].set_title("GT MCES of what SIMBA ranked #1 instead")
    axes[0, 1].set_xlabel("GT MCES to query")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(cos_wrong_ranks, bins=40, color="#1f77b4")
    axes[1, 0].set_title(
        "cosine's rank for the true match\n(150-like group: SIMBA right, cosine wrong)"
    )
    axes[1, 0].set_xlabel("rank (1=best)")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(cos_wrong_top1_gt, bins=40, color="#d62728")
    axes[1, 1].set_title("GT MCES of what cosine ranked #1 instead")
    axes[1, 1].set_xlabel("GT MCES to query")
    axes[1, 1].set_ylabel("count")

    fig.suptitle(f"{exp_dir.name}\nHit@{k} disagreement dive")
    fig.tight_layout()
    output = Path(args.output or exp_dir / f"hit{k}_disagreement_dive.png")
    fig.savefig(output, dpi=150)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
