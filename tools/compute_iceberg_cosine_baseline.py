"""Cosine-similarity (no SIMBA) baseline for the ICEBERG retrieval Hit@1/5/20
benchmark -- how much of the retrieval signal SIMBA actually adds over raw
spectral similarity. Logged as a flat, constant reference line into the same
"cosine" TensorBoard run tools/compute_val_cosine_baseline.py creates, under
both the raw and CORN-corrected tag families (cosine has no corrected
variant of its own, so the same value is mirrored into both -- see that
script for why).

Ranks ICEBERG-predicted candidate spectra against the real test spectrum by
plain binned spectral cosine similarity (sparse dot product on a fixed
m/z-binned grid, no model, no GPU needed).

Usage:
    uv run python tools/compute_iceberg_cosine_baseline.py \\
        --tb_logdir /path/to/experiment_checkpoint_dir
"""

import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm

from simba.core.training.iceberg_retrieval import (
    DEFAULT_CANDIDATE_TSV,
    DEFAULT_CANDIDATES,
    DEFAULT_ICEBERG_PREDS,
    DEFAULT_MGF,
    canonicalize,
    compute_hit_rates_from_ranking,
    load_all_iceberg_data,
)


BIN_WIDTH = 0.01
MAX_MZ = 1100.0


def bin_spectra(spectra: list, bin_width: float, max_mz: float) -> sp.csr_matrix:
    """Bin (mz, intensity) peak lists onto a fixed grid, sqrt-compress,
    L2-normalize. One sparse row per spectrum; a plain dot product between
    two rows is then exactly their cosine similarity."""
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

    mat = sp.coo_matrix((data, (rows, cols)), shape=(len(spectra), n_bins)).tocsr()
    mat.data = np.sqrt(mat.data)
    return normalize(mat, norm="l2", axis=1, copy=False)


def rank_candidates_cosine(
    test_smiles,
    test_adducts,
    query_candidates,
    cand_smi_to_row,
    test_mat,
    cand_mat,
    top_k=20,
):
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
        per_query.append([cand_smis[j] for j in order][:top_k])
    return per_query


def log_to_tensorboard(checkpoint_dir, hits: dict, max_step: int):
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(
        save_dir=str(checkpoint_dir), name="tb_logs", version="cosine"
    )
    for prefix in ("iceberg_hit_at", "iceberg_hit_at_corrected"):
        for k, v in hits.items():
            for step in (0, max_step):
                logger.experiment.add_scalar(f"{prefix}_{k}", v, global_step=step)
    logger.experiment.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tb_logdir", required=True)
    parser.add_argument("--mgf", default=DEFAULT_MGF)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate_tsv", nargs="+", default=DEFAULT_CANDIDATE_TSV)
    parser.add_argument("--iceberg_preds", nargs="+", default=DEFAULT_ICEBERG_PREDS)
    parser.add_argument("--max_step", type=int, default=20000)
    args = parser.parse_args()

    print("\nLoading test spectra + ICEBERG candidate data ...")
    data = load_all_iceberg_data(
        args.mgf, args.candidates, args.candidate_tsv, args.iceberg_preds
    )
    print(f"  {len(data['test_smiles'])} real test spectra")
    print(f"  {len(data['cand_spectra'])} candidate spectra ready")

    print(f"\nBinning spectra (bin_width={BIN_WIDTH} Da, max_mz={MAX_MZ} Da) ...")
    test_mat = bin_spectra(data["test_spectra"], BIN_WIDTH, MAX_MZ)
    cand_mat = bin_spectra(data["cand_spectra"], BIN_WIDTH, MAX_MZ)

    print("\nRanking candidates by binned spectral cosine similarity ...")
    ranked = rank_candidates_cosine(
        data["test_smiles"],
        data["test_adducts"],
        data["query_candidates"],
        data["cand_smi_to_row"],
        test_mat,
        cand_mat,
    )
    hits, n_no_cand = compute_hit_rates_from_ranking(data["test_smiles"], ranked)
    n_scored = len(data["test_smiles"]) - n_no_cand
    print(f"\n=== Cosine-similarity baseline (no SIMBA), n={n_scored} ===")
    for k, v in hits.items():
        print(f"  Hit@{k}: {v:.4f}")

    print(f"\nLogging to TensorBoard under {args.tb_logdir}/tb_logs/cosine ...")
    log_to_tensorboard(args.tb_logdir, hits, args.max_step)
    print("Done.")


if __name__ == "__main__":
    main()
