"""3e: per-query min/mean/max MCES-to-pool distribution plots.

For each query, gather the MCES distance to every other molecule in its
pool (candidates for test-to-candidate; the rest of the test fold for
test-to-test), then take that query's min/mean/max over its pool. The plot
shows how those three per-query statistics are distributed across all
queries — same style as the reference histogram (three overlaid histograms,
legend "Dist. stat.": min/means/max).

The query is each individual test SPECTRUM (not one averaged embedding per
test molecule) — matches how the real per-spectrum retrieval evaluation
actually ranks; averaging first was checked and found to inflate accuracy
(hit@1 16.3% averaged vs. 10.2% real), since it denoises the query side in a
way the real evaluation never benefits from. See ood_generalization_check.py's
module docstring for the same point in more detail.

NO AVERAGING ANYWHERE, on either side: test-to-test uses one dense matrix of
raw per-spectrum embeddings on BOTH sides (score_test_to_test_no_averaging);
test-to-candidate matches each candidate embedding to the query's own adduct
specifically, never blending a candidate's several per-adduct ICEBERG
embeddings into one (see ood_generalization_check.py's module docstring for
why that distinction matters).

EXCLUSION MODE (stated explicitly in every title/log line below — read this
before comparing numbers across plots):
  - test-to-candidate: self (the true candidate) is INCLUDED — GT=0, added
    back in via add_self_pairs, scored against the true candidate's own
    ICEBERG-predicted spectrum for the query's own adduct (dropped, not
    substituted, if no such embedding exists).
  - test-to-test: only the literal self SPECTRUM is excluded. Same-molecule
    *different*-spectrum pairs (GT=0, genuine spectrum-vs-spectrum) are
    INCLUDED alongside the usual cross-molecule pairs.

Two exceptions, same underlying reason: GT=0 for a trivial pair (self for
test-to-candidate; same-molecule-different-spectrum for test-to-test) makes
"min" exactly 0 for every query the instant that pair is included in the
pool — the same uninformative-spike problem that originally motivated
excluding self altogether. So:
  - test-to-candidate's GT plot computes "min" from the self-EXCLUDED pool.
  - test-to-test's GT plot computes "min" from the cross-molecule-only pool
    (same-molecule spectra excluded).
"mean"/"max" (here and everywhere else) use the fully-included pool in both
cases, since one more (non-extreme) 0 among many pool items barely moves
either.

Six plots, GT / SIMBA-predicted / |SIMBA-predicted - GT| kept as separate
plots (not overlaid) so they read cleanly against each other:
  - test-to-candidate, GT MCES (self included, except min)
  - test-to-candidate, SIMBA-predicted MCES (self included)
  - test-to-candidate, |SIMBA-predicted - GT| MCES (self included)
  - test-to-test, GT MCES (same-molecule spectra included, except min)
  - test-to-test, SIMBA-predicted MCES (self spectrum excluded, same-molecule spectra included)
  - test-to-test, |SIMBA-predicted - GT| MCES (self spectrum excluded, same-molecule spectra included)

Note: the "GT" plots aggregate over test SPECTRA — a molecule with 5 spectra
contributes its (identical, for the cross-molecule part) GT row 5 times —
so they weight by spectrum count rather than by molecule. That's
intentional: it keeps the population identical across all three plot types
per test-to-candidate / test-to-test, i.e. directly comparable to the
SIMBA/diff plots' populations.

Usage:
    uv run python tools/mces_pool_distribution_plots.py \\
        --intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --test_to_test_prepro_dir /path/to/preprocessing_msg_exact_mces_1020 \\
        --output_dir /path/to/output \\
        --mces_max_value 40
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ood_generalization_check import (
    add_self_pairs,
    build_candidate_embeddings_by_smi_adduct,
    expand_and_score_ragged,
    score_test_to_test_no_averaging,
)


def per_query_stats_ragged(
    pair_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group by query (column 0), aggregate the value column (last column) —
    for a ragged pool (variable-size candidate lists), e.g. test-to-candidate."""
    df = pd.DataFrame(
        {"q": pair_arr[:, 0].astype(int), "v": pair_arr[:, -1].astype(float)}
    )
    agg = df.groupby("q")["v"].agg(["min", "mean", "max"])
    return agg["min"].to_numpy(), agg["mean"].to_numpy(), agg["max"].to_numpy()


def row_stats(
    matrix: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row nanmin/nanmean/nanmax over a dense matrix + boolean mask — the
    dense-matrix counterpart of per_query_stats_ragged, for a single source
    (no combining step needed since test-to-test is now one unified matrix,
    not a dense part + a separate ragged same-molecule part)."""
    masked = np.where(mask, matrix, np.nan)
    with np.errstate(all="ignore"):
        mins = np.nanmin(masked, axis=1)
        means = np.nanmean(masked, axis=1)
        maxs = np.nanmax(masked, axis=1)
    return mins, means, maxs


def plot_min_mean_max(
    mins: np.ndarray,
    means: np.ndarray,
    maxs: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 80,
    x_clip_max: float | None = None,
) -> None:
    if x_clip_max is not None:
        mins = mins[mins <= x_clip_max]
        means = means[means <= x_clip_max]
        maxs = maxs[maxs <= x_clip_max]
        xmax = x_clip_max
    else:
        xmax = max(np.nanmax(mins), np.nanmax(means), np.nanmax(maxs))
    bin_edges = np.linspace(0, xmax, bins + 1)

    plt.figure(figsize=(10, 5.5))
    for name, vals in [("min", mins), ("means", means), ("max", maxs)]:
        plt.hist(vals, bins=bin_edges, alpha=0.5, label=name)
    plt.xlabel("MCES distance")
    plt.ylabel("Count")
    plt.title(title, fontsize=11, wrap=True)
    plt.legend(title="Dist. stat.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(
        f"Saved {out_path}  (n={len(mins)}, min-of-min={np.nanmin(mins):.1f}, max-of-max={np.nanmax(maxs):.1f})"
    )


def run(
    intermediates_dir: str,
    gt_mces_dir: str,
    test_to_test_prepro_dir: str,
    output_dir: str,
    mces_max_value: float = 40.0,
) -> None:
    inter = Path(intermediates_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved embeddings + SMILES + adducts ...")
    test_embeddings = torch.load(inter / "test_embeddings.pt", map_location="cpu")
    test_smiles = json.loads((inter / "test_smiles.json").read_text())
    test_adducts = json.loads((inter / "test_adducts.json").read_text())
    candidate_embeddings = torch.load(
        inter / "candidate_embeddings.pt", map_location="cpu"
    )
    candidate_smiles = json.loads((inter / "candidate_smiles.json").read_text())
    candidate_adducts = json.loads((inter / "candidate_adducts.json").read_text())

    cand_smi_adduct_to_emb = build_candidate_embeddings_by_smi_adduct(
        candidate_smiles, candidate_adducts, candidate_embeddings
    )

    print(
        "\n--- test-to-candidate (per-spectrum query, own-adduct candidate match; SELF INCLUDED) ---"
    )
    gt_dir = Path(gt_mces_dir)
    tc_idx_to_smiles = gt_dir.joinpath("smiles.txt").read_text().splitlines()
    tc_gt_pairs_raw = np.load(gt_dir / "mces_exact.npy")
    valid = (tc_gt_pairs_raw[:, 2] >= 0) & ~np.isnan(tc_gt_pairs_raw[:, 2])
    tc_gt_pairs_raw = tc_gt_pairs_raw[valid]
    tc_gt_pairs = add_self_pairs(tc_gt_pairs_raw)
    print(
        f"  {len(tc_gt_pairs_raw):,} GT molecule pairs (+{len(tc_gt_pairs) - len(tc_gt_pairs_raw):,} self)"
    )

    tc_spec_idx, tc_other_idx, tc_gt, tc_pred = expand_and_score_ragged(
        tc_gt_pairs,
        tc_idx_to_smiles,
        test_smiles,
        test_adducts,
        test_embeddings,
        cand_smi_adduct_to_emb,
        mces_max_value,
    )
    print(f"  {len(tc_spec_idx):,} scored (spectrum, candidate) pairs after expansion")

    # GT's own "min" is trivially 0 for every query once self (GT=0 by
    # definition) is included — uninformative, so it alone is computed from
    # the self-EXCLUDED pool. "mean"/"max" barely move by including one
    # more (non-extreme) 0 among ~185 avg pool items, so they use the
    # self-included pool like everything else, for a consistent population.
    tc_spec_idx_excl, tc_other_idx_excl, tc_gt_excl, _ = expand_and_score_ragged(
        tc_gt_pairs_raw,
        tc_idx_to_smiles,
        test_smiles,
        test_adducts,
        test_embeddings,
        cand_smi_adduct_to_emb,
        mces_max_value,
    )
    tc_gt_min, _, _ = per_query_stats_ragged(
        np.column_stack([tc_spec_idx_excl, tc_other_idx_excl, tc_gt_excl])
    )
    _, tc_gt_mean, tc_gt_max = per_query_stats_ragged(
        np.column_stack([tc_spec_idx, tc_other_idx, tc_gt])
    )
    plot_min_mean_max(
        tc_gt_min,
        tc_gt_mean,
        tc_gt_max,
        "Test-to-candidate: GT MCES to pool (self included, except min)",
        out_dir / "test_to_candidate_gt.png",
        x_clip_max=40.0,
    )

    tc_pred_min, tc_pred_mean, tc_pred_max = per_query_stats_ragged(
        np.column_stack([tc_spec_idx, tc_other_idx, tc_pred])
    )
    plot_min_mean_max(
        tc_pred_min,
        tc_pred_mean,
        tc_pred_max,
        "Test-to-candidate: SIMBA-predicted MCES to pool (self included)",
        out_dir / "test_to_candidate_simba.png",
    )

    tc_diff_min, tc_diff_mean, tc_diff_max = per_query_stats_ragged(
        np.column_stack([tc_spec_idx, tc_other_idx, np.abs(tc_pred - tc_gt)])
    )
    plot_min_mean_max(
        tc_diff_min,
        tc_diff_mean,
        tc_diff_max,
        "Test-to-candidate: |SIMBA - GT| MCES to pool (self included)",
        out_dir / "test_to_candidate_abs_dif_gt_simba.png",
    )

    print(
        "\n--- test-to-test (per-spectrum query, no averaging on either side; self SPECTRUM excluded, same-molecule spectra included) ---"
    )
    with open(Path(test_to_test_prepro_dir) / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    tt_idx_to_smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()
    tt_gt_pairs = np.load(
        Path(test_to_test_prepro_dir)
        / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy"
    )
    print(
        f"  {len(tt_gt_pairs):,} mined cross-molecule pairs, {len(tt_idx_to_smiles)} molecules"
    )

    tt_pred_matrix, tt_gt_matrix, tt_valid_mask, tt_spec_molidx = (
        score_test_to_test_no_averaging(
            tt_gt_pairs, tt_idx_to_smiles, test_smiles, test_embeddings, mces_max_value
        )
    )
    n_valid_spectra = tt_valid_mask.shape[0]
    print(
        f"  {n_valid_spectra:,} x {n_valid_spectra:,} individual-spectrum matrix "
        f"({int(tt_valid_mask.sum()):,} valid cells)"
    )

    # Same as test-to-candidate's GT plot: "min" alone is computed from the
    # cross-molecule-only pool — GT=0 for same-molecule pairs trivially makes
    # "min" exactly 0 for every query with >=2 spectra, which is pointless.
    # mean/max keep same-molecule pairs included, like everywhere else.
    same_molecule = tt_spec_molidx[:, None] == tt_spec_molidx[None, :]
    cross_molecule_mask = tt_valid_mask & ~same_molecule
    tt_gt_min, _, _ = row_stats(tt_gt_matrix, cross_molecule_mask)
    _, tt_gt_mean, tt_gt_max = row_stats(tt_gt_matrix, tt_valid_mask)
    plot_min_mean_max(
        tt_gt_min,
        tt_gt_mean,
        tt_gt_max,
        "Test-to-test: GT MCES to pool (same-molecule spectra included, except min)",
        out_dir / "test_to_test_gt.png",
    )

    tt_pred_min, tt_pred_mean, tt_pred_max = row_stats(tt_pred_matrix, tt_valid_mask)
    plot_min_mean_max(
        tt_pred_min,
        tt_pred_mean,
        tt_pred_max,
        "Test-to-test: SIMBA-predicted MCES to pool (self spectrum excluded, same-molecule spectra included, no averaging)",
        out_dir / "test_to_test_simba.png",
    )

    tt_diff_min, tt_diff_mean, tt_diff_max = row_stats(
        np.abs(tt_pred_matrix - tt_gt_matrix), tt_valid_mask
    )
    plot_min_mean_max(
        tt_diff_min,
        tt_diff_mean,
        tt_diff_max,
        "Test-to-test: |SIMBA - GT| MCES to pool (self spectrum excluded, same-molecule spectra included, no averaging)",
        out_dir / "test_to_test_abs_dif_gt_simba.png",
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--intermediates_dir", required=True)
    p.add_argument("--gt_mces_dir", required=True)
    p.add_argument("--test_to_test_prepro_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mces_max_value", type=float, default=40.0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
