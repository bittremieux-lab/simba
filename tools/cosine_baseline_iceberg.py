"""Cosine-similarity baseline for SIMBA+ICEBERG retrieval (3d).

Ranks ICEBERG-predicted candidate spectra against the real test spectrum by
plain binned spectral cosine similarity — no SIMBA involved — to see how
much of the retrieval signal SIMBA actually adds over raw spectral
similarity (Gaetan's suggested sanity-check baseline).

Reuses the exact same candidate-loading / hit-rate / GT-MCES machinery as
tools/simba_retrieval_iceberg.py; only the ranking step differs: spectra are
binned onto a fixed m/z grid (sum intensity per bin, sqrt-compress, L2-
normalize), so cosine similarity between two spectra is a single dot
product, and per-query ranking is one sparse matrix multiply restricted to
that query's own candidate list (<=256 rows) — no SIMBA embedding pass, no
GPU, so this should run fine directly on the login node (no SLURM job
needed unless it turns out to be too slow in practice).

--candidate_tsv / --iceberg_preds each accept one path or several (matched
1:1 by position) -- see simba_retrieval_iceberg.py's docstring for why
(scoring a query set whose candidates are split across the original files
plus a delta file of only the newly-generated pairs).

Usage:
    uv run python tools/cosine_baseline_iceberg.py \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --candidate_tsv /path/to/candidates_test_official.tsv [/path/to/delta.tsv ...] \\
        --iceberg_preds /path/to/preds.hdf5 [/path/to/delta_preds.hdf5 ...] \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates
"""

import argparse
import json

import numpy as np
import scipy.sparse as sp
from simba_retrieval import canonicalize, load_spectra
from simba_retrieval_iceberg import (
    build_candidate_spectra,
    compute_hit_rates_from_ranking,
    compute_mces_stats,
    load_candidate_index,
    load_gt_mces_lookup,
    load_iceberg_spectra,
)
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


def bin_spectra(spectra: list, bin_width: float, max_mz: float) -> sp.csr_matrix:
    """Bin (mz, intensity) peak lists onto a fixed grid, sqrt-compress,
    L2-normalize. Returns one sparse row per spectrum; a plain dot product
    between two rows is then exactly their cosine similarity.
    """
    n_bins = int(np.ceil(max_mz / bin_width)) + 1
    rows, cols, data = [], [], []
    for i, spec in enumerate(tqdm(spectra, desc="Binning spectra")):
        mz = np.asarray(spec.mz, dtype=np.float64)
        intensity = np.asarray(spec.intensity, dtype=np.float64)
        keep = mz <= max_mz
        if not keep.all():
            mz, intensity = mz[keep], intensity[keep]
        if len(mz) == 0:
            continue
        bins = np.clip(np.round(mz / bin_width).astype(np.int64), 0, n_bins - 1)
        rows.append(np.full(len(bins), i, dtype=np.int64))
        cols.append(bins)
        data.append(intensity)

    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    data = np.concatenate(data) if data else np.zeros(0, dtype=np.float64)

    # coo -> csr sums duplicate (row, col) entries automatically, i.e. multiple
    # peaks landing in the same bin are summed, not overwritten.
    mat = sp.coo_matrix((data, (rows, cols)), shape=(len(spectra), n_bins)).tocsr()
    mat.data = np.sqrt(
        mat.data
    )  # standard intensity compression (matches SIMBA's own preprocessing)
    mat = normalize(mat, norm="l2", axis=1, copy=False)
    return mat


def rank_candidates_cosine(
    test_smiles: list,
    test_adducts: list,
    query_candidates: dict,
    cand_smi_to_row: dict,
    test_mat: sp.csr_matrix,
    cand_mat: sp.csr_matrix,
    top_k: int = 20,
) -> list:
    """Same contract as simba_retrieval_iceberg.rank_candidates, but ranking by
    binned spectral cosine similarity (sparse dot product) instead of SIMBA
    embedding similarity."""
    per_query = []
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if not row_idxs:
            per_query.append(None)
            continue

        sims = np.asarray((test_mat[i] @ cand_mat[row_idxs].T).todense()).ravel()
        order = np.argsort(-sims)
        ranked = [cand_smis[j] for j in order]
        per_query.append(ranked[:top_k])
    return per_query


def run(
    mgf: str,
    candidates: str,
    candidate_tsv: str | list[str],
    iceberg_preds: str | list[str],
    split: str = "test",
    bin_width: float = 0.01,
    max_mz: float = 1100.0,
    output_tsv: str | None = None,
    gt_mces_dir: str | None = None,
    skip_mces: bool = False,
):
    print(f"\nLoading {split}-fold real spectra from {mgf} ...")
    test_smiles, test_spectra = load_spectra(mgf, split)
    test_adducts = [s.adduct for s in test_spectra]
    print(f"  {len(test_smiles)} real test spectra")

    print(f"\nLoading candidate pools from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)
    query_candidates = {canonicalize(k): v for k, v in candidate_json.items()}

    print(f"\nLoading candidate index from {candidate_tsv} ...")
    cand_index = load_candidate_index(candidate_tsv)

    print(f"\nLoading ICEBERG-predicted spectra from {iceberg_preds} ...")
    all_cand_ids = cand_index["spec"].tolist()
    iceberg_specs = load_iceberg_spectra(iceberg_preds, all_cand_ids)
    print(
        f"  {len(iceberg_specs)} / {len(all_cand_ids)} candidates have a predicted spectrum"
    )

    print("\nBuilding candidate SpectrumExt objects ...")
    cand_smiles, cand_spectra = build_candidate_spectra(cand_index, iceberg_specs)
    cand_smi_to_row = {
        (smi, spec.adduct): row_idx
        for row_idx, (smi, spec) in enumerate(zip(cand_smiles, cand_spectra))
    }
    print(f"  {len(cand_spectra)} candidate spectra ready")

    print(f"\nBinning spectra (bin_width={bin_width} Da, max_mz={max_mz} Da) ...")
    test_mat = bin_spectra(test_spectra, bin_width, max_mz)
    cand_mat = bin_spectra(cand_spectra, bin_width, max_mz)
    print(f"  test matrix: {test_mat.shape}, nnz={test_mat.nnz:,}")
    print(f"  candidate matrix: {cand_mat.shape}, nnz={cand_mat.nnz:,}")

    print("\nRanking candidates by binned spectral cosine similarity ...")
    per_query_ranked = rank_candidates_cosine(
        test_smiles,
        test_adducts,
        query_candidates,
        cand_smi_to_row,
        test_mat,
        cand_mat,
        top_k=20,
    )
    results, n_no_candidates = compute_hit_rates_from_ranking(
        test_smiles, per_query_ranked
    )
    if n_no_candidates:
        print(
            f"  Warning: {n_no_candidates}/{len(test_smiles)} queries had no usable candidates"
        )

    n_scored = len(test_smiles) - n_no_candidates
    print(f"\n=== Cosine-similarity baseline (no SIMBA), {split}, n={n_scored} ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if not skip_mces:
        print(f"\nLoading GT MCES lookup from {gt_mces_dir} ...")
        gt_lookup = load_gt_mces_lookup(gt_mces_dir)
        print(f"  {len(gt_lookup) // 2} unique (test, candidate) pairs with a GT value")

        print("\nComputing GT MCES between ranked candidates and the true molecule ...")
        mces_results = compute_mces_stats(test_smiles, per_query_ranked, gt_lookup)
        results.update(mces_results)
        print("\n=== GT MCES to true molecule (exact, threshold=20) ===")
        for k, v in mces_results.items():
            print(f"  {k}: {v}")

    if output_tsv:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "split": split,
                    "model": "cosine_baseline_no_simba",
                    "n": n_scored,
                    **results,
                }
            ]
        ).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results


def main():
    p = argparse.ArgumentParser(
        description="Cosine-similarity (no SIMBA) baseline for SIMBA+ICEBERG retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidates", required=True, help="Candidate JSON {smiles: [cands]}"
    )
    p.add_argument(
        "--candidate_tsv",
        required=True,
        nargs="+",
        help="ICEBERG candidate TSV(s) (smiles/ionization/precursor) -- "
        "one path, or several (e.g. the original plus a delta file) "
        "to be concatenated, matched 1:1 by position with --iceberg_preds",
    )
    p.add_argument(
        "--iceberg_preds",
        required=True,
        nargs="+",
        help="ICEBERG predictions HDF5(s) -- one path, or several to merge",
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bin_width", type=float, default=0.01, help="m/z bin width (Da)")
    p.add_argument(
        "--max_mz",
        type=float,
        default=1100.0,
        help="Max m/z covered by the binning grid (max precursor mass in this "
        "candidate set is ~997 Da)",
    )
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--gt_mces_dir",
        default=None,
        help=(
            "Dir with smiles.txt + combined mces_exact.npy from "
            "tools/prepare_gt_mces_retrieval.py (run its combine step first). "
            "Required unless --skip_mces."
        ),
    )
    p.add_argument(
        "--skip_mces",
        action="store_true",
        help="Skip GT MCES computation, report only hit@k",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
