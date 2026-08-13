"""
Cosine retrieval baseline: binned-spectrum cosine NN transfer + Tanimoto
candidate ranking. The "cosine to train + FP of NN retrieval" row Gaetan's
sanity-check table calls for — the exact same mechanism as
tools/simba_retrieval.py, with the SIMBA-embedding NN-transfer step swapped
for plain binned-spectrum cosine similarity. Nothing else changes:

For each test spectrum:
  1. Find nearest train spectrum by cosine similarity of binned spectra
     (tools/cosine_baseline_iceberg.bin_spectra: fixed m/z grid, sqrt-
     compress, L2-normalize -- so a dot product IS cosine similarity).
  2. Transfer that train molecule's Morgan fingerprint as the predicted FP.
  3. Rank leaderboard candidates by Tanimoto similarity to the transferred FP.

canonicalize / morgan_fp / build_fp_lookup / tanimoto_scores are imported
unchanged from simba_retrieval.py -- they operate on fingerprints, not
embeddings, so there is nothing to adapt. load_spectra is unchanged too
(still returns raw, unprocessed peaks -- bin_spectra does its own binning
directly on those, no SIMBA preprocessing involved). compute_hit_rates is
NOT reused as-is: it calls `.numpy()` on each transferred fingerprint,
which assumes a torch tensor (true for simba_retrieval.py's dense
embeddings); here transferred fingerprints come out of numpy sparse-matrix
arithmetic already, so a local compute_hit_rates_np below does the exact
same scoring without that assumption. The NN-transfer step itself is new
too (sparse chunked cosine-similarity search instead of a dense torch
matmul, since a full (N_test, N_train) dense similarity matrix would be
too large to materialize at once for realistic train-fold sizes).

Usage:
    uv run python tools/cosine_retrieval.py \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from cosine_baseline_iceberg import bin_spectra
from simba_retrieval import (
    build_fp_lookup,
    canonicalize,
    load_spectra,
    tanimoto_scores,
)
from tqdm.auto import tqdm


# ── Nearest-neighbor transfer (sparse) ─────────────────────────────────────────


def nearest_neighbor_transfer_sparse(
    test_mat: sp.csr_matrix,
    train_mat: sp.csr_matrix,
    train_fps: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse-matrix analogue of simba_retrieval.nearest_neighbor_transfer:
    for each test spectrum's binned cosine vector, find the nearest train
    spectrum by cosine similarity (both already L2-normalized by
    bin_spectra, so a dot product is cosine similarity directly), transfer
    its Morgan fingerprint. Chunked over test rows so no more than
    chunk_size x N_train floats get densified at once.

    Returns (transferred_fps (N_test, nbits), nn_indices (N_test,))."""
    n_test = test_mat.shape[0]
    transferred = np.zeros((n_test, train_fps.shape[1]), dtype=train_fps.dtype)
    nn_indices = np.zeros(n_test, dtype=np.int64)
    train_mat_t = train_mat.T.tocsr()

    for start in tqdm(range(0, n_test, chunk_size), desc="NN transfer", unit="chunk"):
        chunk = test_mat[start : start + chunk_size]
        sims = np.asarray((chunk @ train_mat_t).todense())  # (C, N_train)
        nn_idx = sims.argmax(axis=1)
        transferred[start : start + len(nn_idx)] = train_fps[nn_idx]
        nn_indices[start : start + len(nn_idx)] = nn_idx

    return transferred, nn_indices


# ── Hit rates (numpy-native; compute_hit_rates in simba_retrieval.py assumes
#    torch tensors, calling .numpy() on each fingerprint) ─────────────────────


def compute_hit_rates_np(
    test_smiles: list,
    cand_lists: list,
    transferred_fps: np.ndarray,
    fp_lookup: dict,
    ks: tuple = (1, 5, 20),
) -> dict:
    hits = dict.fromkeys(ks, 0)
    n = 0

    for q_smi, cands, q_fp in zip(test_smiles, cand_lists, transferred_fps):
        if not cands:
            continue
        cand_fp_mat = np.stack([fp_lookup[c] for c in cands])
        scores = tanimoto_scores(q_fp.astype(np.uint8), cand_fp_mat)
        ranked_cands = [cands[i] for i in np.argsort(-scores)]

        q_canon = canonicalize(q_smi)
        for k in ks:
            if any(canonicalize(c) == q_canon for c in ranked_cands[:k]):
                hits[k] += 1
        n += 1

    return {f"hit@{k}": hits[k] / n if n > 0 else 0.0 for k in ks}


# ── Main ──────────────────────────────────────────────────────────────────────


def save_intermediates(
    out_dir: Path,
    split: str,
    train_smiles,
    test_smiles,
    train_mat,
    test_mat,
    nn_indices,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "train_mat.npz", train_mat)
    sp.save_npz(out_dir / f"{split}_mat.npz", test_mat)
    np.save(out_dir / f"{split}_nn_indices.npy", nn_indices)
    (out_dir / "train_smiles.json").write_text(json.dumps(train_smiles))
    (out_dir / f"{split}_smiles.json").write_text(json.dumps(test_smiles))
    print(f"Intermediates saved to {out_dir}/")


def run(
    mgf: str,
    candidates: str,
    split: str = "test",
    output_tsv: str | None = None,
    intermediates_dir: str | None = None,
    bin_width: float = 0.01,
    max_mz: float = 1100.0,
    chunk_size: int = 512,
):
    print(f"\nLoading candidate JSON from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)

    print(f"\nLoading spectra (split={split}, ref=train) ...")
    train_smiles, train_spectra = load_spectra(mgf, "train")
    test_smiles, test_spectra = load_spectra(mgf, split)
    print(f"  train: {len(train_smiles)}  {split}: {len(test_smiles)}")

    # Deduplicate training: keep first spectrum per unique canonical SMILES --
    # same reasoning as simba_retrieval.py: ensures each training molecule is
    # represented exactly once, so argmax cosine-sim maps unambiguously to a
    # unique molecule. Orthogonal to the embedding source, kept identical for
    # a fair comparison against the SIMBA variant of this same baseline.
    print("Deduplicating training spectra (first per unique molecule) ...")
    _seen: set[str] = set()
    _dedup_smi, _dedup_spec = [], []
    for _smi, _spec in zip(train_smiles, train_spectra):
        _c = canonicalize(_smi)
        if _c not in _seen:
            _seen.add(_c)
            _dedup_smi.append(_smi)
            _dedup_spec.append(_spec)
    train_smiles, train_spectra = _dedup_smi, _dedup_spec
    print(f"  After dedup: {len(train_smiles)} unique training molecules")

    # Build candidate lists aligned to test spectra, canonicalize keys
    print("\nBuilding candidate lists ...")
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}
    cand_lists = [cand_json_canon.get(canonicalize(s), []) for s in test_smiles]
    n_missing = sum(1 for c in cand_lists if not c)
    if n_missing:
        print(f"  Warning: {n_missing}/{len(test_smiles)} queries have no candidates")

    # Fingerprints for all unique SMILES
    print("\nComputing Morgan fingerprints ...")
    all_smiles = set(train_smiles) | {c for lst in cand_lists for c in lst}
    fp_lookup = build_fp_lookup(all_smiles)
    train_fps = np.stack([fp_lookup[s] for s in train_smiles])  # (N_train, 2048)

    print(f"\nBinning spectra (bin_width={bin_width} Da, max_mz={max_mz} Da) ...")
    train_mat = bin_spectra(train_spectra, bin_width, max_mz)
    test_mat = bin_spectra(test_spectra, bin_width, max_mz)
    print(f"  train matrix: {train_mat.shape}, nnz={train_mat.nnz:,}")
    print(f"  {split} matrix: {test_mat.shape}, nnz={test_mat.nnz:,}")

    # NN transfer
    print("\nNearest-neighbor transfer (sparse cosine) ...")
    transferred_fps, nn_indices = nearest_neighbor_transfer_sparse(
        test_mat, train_mat, train_fps, chunk_size=chunk_size
    )

    # Save intermediates before scoring so they're available even if scoring is re-run
    if intermediates_dir:
        save_intermediates(
            Path(intermediates_dir),
            split,
            train_smiles,
            test_smiles,
            train_mat,
            test_mat,
            nn_indices,
        )

    # Evaluate
    print("\nScoring candidates ...")
    results = compute_hit_rates_np(test_smiles, cand_lists, transferred_fps, fp_lookup)

    print(f"\n=== Cosine retrieval ({split}, n={len(test_smiles)}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if output_tsv:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "split": split,
                    "model": "cosine_nn_transfer",
                    "n": len(test_smiles),
                    **results,
                }
            ]
        ).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results


def main():
    p = argparse.ArgumentParser(
        description="Cosine retrieval: binned-spectrum NN transfer + Tanimoto ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidates", required=True, help="Candidate JSON {smiles: [cands]}"
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory to save binned-spectrum matrices and NN indices",
    )
    p.add_argument("--bin_width", type=float, default=0.01, help="m/z bin width (Da)")
    p.add_argument(
        "--max_mz",
        type=float,
        default=1100.0,
        help="Max m/z covered by the binning grid",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Test spectra per NN-transfer chunk (memory/speed tradeoff)",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
