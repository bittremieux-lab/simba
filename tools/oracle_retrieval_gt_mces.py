"""
GT-MCES oracle retrieval: an upper bound on the "old approach" (train-NN-
transfer) rows of Gaetan's sanity-check table. Same pipeline as
simba_retrieval.py / cosine_retrieval.py -- find each test spectrum's
nearest TRAINING molecule, transfer its Morgan fingerprint, Tanimoto-rank
candidates -- except the nearest-neighbor-selection step uses the TRUE
structural distance (GT MCES) instead of SIMBA-embedding or cosine
similarity: for each test molecule, the training molecule with the smallest
max(lb_matrix, all_smiles_mces.hdf5) MCES is picked as "nearest," exactly as
the training data preprocessing (prepare_msg_official_split_max_lb_hdf5.py)
computes distances.

NO new MCES computation: both sources are pre-existing all-vs-all matrices
already covering 100% of every split's train/test molecules used in this
repo (verified directly against every preprocessing_*_max_lb_hdf5/run.log):
  - data/massspecgym/lb_matrix.npy: 240,637-molecule condensed lower-bound
    MCES matrix (tight for MCES>=10, underestimate for <10).
  - data/massspecgym/data/auxiliary/all_smiles_mces.hdf5: 34,731-molecule
    condensed matrix, exact for MCES<10, weak lower bound for >=10.
This script is a lookup (argmin over a memory-mapped array) + Tanimoto
ranking, nothing more.

Adapted from tools/oracle_retrieval_max_lb_hdf5.py, which had:
  - A bug in the HDF5 condensed-index formula: it reused lb_matrix's
    hi*(hi-1)//2+lo convention for HDF5 too. HDF5 actually uses the
    scipy-pdist-style convention (i<j: n*i - i*(i+1)//2 + j-i-1) --
    prepare_msg_official_split_max_lb_hdf5.py's own comment documents this
    exact bug ("using the lb_matrix formula here silently returns wrong
    distances"). Fixed here (_hdf5_condensed_idx). Verified against
    lb_matrix on 1,000 random common pairs before adapting this script:
    Pearson r=0.886, no scatter -- confirms both index conventions are
    correct (a wrong formula would look up essentially unrelated values,
    giving ~0 correlation).
  - Hardcoded paths to a different machine/checkout -- now CLI args.
  - A "_mass.json" candidate pool -- now takes --candidates like every
    other retrieval script here (point it at the formula-matched
    MassSpecGym_retrieval_candidates_formula.json used everywhere else).
  - A per-test-molecule Python loop over the full train pool (~3,000 test
    molecules x ~25,000 train molecules = ~75M scattered mmap reads via
    thousands of small fancy-index gathers) -- replaced with two
    vectorized gathers over the whole (unique test molecule x train
    molecule) grid at once (still ~75M element reads total, but as one
    C-level numpy operation instead of thousands of Python-level ones).
  - Its own copies of morgan_fp/tanimoto_scores -- now imports
    simba_retrieval's versions (identical params: radius=2, nbits=2048)
    instead of duplicating them.
  - compute_hit_rates_np is reused unchanged from cosine_retrieval.py (its
    transferred fingerprints are numpy arrays, same as this script's).
  - Fingerprinting the ENTIRE train pool upfront, copied from
    simba_retrieval.py/cosine_retrieval.py where that's unavoidable (the
    nearest neighbor is only known after computing embeddings, so any train
    molecule could turn out to be needed). Here the oracle NN pick is a pure
    lb_matrix/hdf5 lookup that doesn't depend on fingerprints at all, so it's
    run FIRST -- only the train molecules that actually get selected as some
    test molecule's nearest neighbor (at most one per unique test molecule)
    get fingerprinted, not the whole pool.

Train pool is whatever "train"-labeled spectra --mgf/--split load, exactly
like simba_retrieval.py/cosine_retrieval.py -- so passing the combined
mini-MGFs built by extract_spectra_by_mgf_index.py (scaffold_val_as_test.mgf
/ gaetan_test.mgf) gives a byte-identical train pool to the existing
SIMBA-NN/cosine-NN rows for those splits, and the official MGF's own
FOLD=train for the official column.

Usage:
    uv run python tools/oracle_retrieval_gt_mces.py \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --lb_matrix /path/to/lb_matrix.npy \\
        --lb_smiles /path/to/lb_matrix.smiles.txt \\
        --hdf5_mces /path/to/all_smiles_mces.hdf5 \\
        --split test \\
        --output_tsv /path/to/retrieval_results.tsv
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from cosine_retrieval import compute_hit_rates_np
from simba_retrieval import build_fp_lookup, canonicalize, load_spectra


# ── lb_matrix / HDF5 loading ───────────────────────────────────────────────────


def load_lb_matrix(lb_matrix_path: str, lb_smiles_path: str) -> tuple[dict, np.ndarray]:
    print(f"Loading lb_matrix SMILES index from {lb_smiles_path} ...")
    with open(lb_smiles_path) as fh:
        lb_smiles_to_idx = {line.strip(): i for i, line in enumerate(fh)}
    print(
        f"  {len(lb_smiles_to_idx):,} molecules (already canonical, verified directly)"
    )
    # Full load, NOT mmap: the vectorized gather in oracle_nn_by_molecule needs
    # up to ~80M scattered elements from this 116 GB file -- as a memory-mapped
    # array that's ~80M random small disk reads (confirmed: >2h and still not
    # done on the smallest split). One sequential ~116 GB read is far cheaper
    # on these nodes (1.5 TB RAM) than that much random I/O; needs --mem set
    # high enough in the SLURM script (default per-cpu allocation isn't enough).
    print(f"Loading {lb_matrix_path} fully into RAM (~116 GB, one sequential read) ...")
    lb = np.load(lb_matrix_path)
    print(f"  shape={lb.shape}, dtype={lb.dtype}")
    return lb_smiles_to_idx, lb


def load_hdf5_mces(hdf5_path: str) -> tuple[dict, np.ndarray, int]:
    print(f"Loading HDF5 mces from {hdf5_path} ...")
    with h5py.File(hdf5_path, "r") as hf:
        raw = [
            s.decode() if isinstance(s, bytes) else s
            for s in hf["mces_smiles_order"][:]
        ]
        # HDF5 stores non-canonical SMILES; canonicalize to match train/test lookups.
        hdf5_smiles_to_idx: dict[str, int] = {}
        for i, s in enumerate(raw):
            c = canonicalize(s)
            if c:
                hdf5_smiles_to_idx[c] = i
        hdf5_mces = hf["mces"][:].astype(np.float32)
    print(
        f"  {len(hdf5_smiles_to_idx):,} molecules (of {len(raw):,} raw rows, "
        f"after canonicalization), mces shape={hdf5_mces.shape}"
    )
    return hdf5_smiles_to_idx, hdf5_mces, len(raw)


def _lb_condensed_idx(i, j):
    """lb_matrix's own convention: entry (hi, lo) with hi > lo sits at
    hi*(hi-1)//2 + lo."""
    hi = np.maximum(i, j)
    lo = np.minimum(i, j)
    return hi * (hi - 1) // 2 + lo


def _hdf5_condensed_idx(i, j, n):
    """all_smiles_mces.hdf5's own convention -- scipy-pdist-style, DIFFERENT
    from lb_matrix's (see module docstring): i < j, n*i - i*(i+1)//2 + j-i-1."""
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return n * lo - lo * (lo + 1) // 2 + hi - lo - 1


# ── Oracle nearest-neighbor transfer ───────────────────────────────────────────


def oracle_nn_by_molecule(
    unique_test_canon: list[str],
    train_canon: list[str],
    lb_smiles_to_idx: dict,
    lb: np.ndarray,
    hdf5_smiles_to_idx: dict,
    hdf5_mces: np.ndarray,
    hdf5_n: int,
    mces_cap: float = 40.0,
) -> tuple[dict[str, int], dict[str, float]]:
    """For each unique test molecule (canonical SMILES), the index (into
    train_canon) of the training molecule with the smallest true
    max(lb_matrix, hdf5) MCES -- an oracle stand-in for
    nearest_neighbor_transfer's embedding-similarity search. -1 for a test
    molecule present in neither source.

    Vectorized over the whole (unique test molecule x train molecule) grid
    in two gathers (one per source) rather than looping per test molecule --
    same total number of element reads, but one C-level numpy operation each
    instead of thousands of small Python-level ones."""
    n_test, n_train = len(unique_test_canon), len(train_canon)
    print(
        f"  oracle NN grid: {n_test} unique test molecules x {n_train} train molecules"
    )

    test_lb_idxs = np.array(
        [lb_smiles_to_idx.get(s, -1) for s in unique_test_canon], dtype=np.int64
    )
    test_hdf5_idxs = np.array(
        [hdf5_smiles_to_idx.get(s, -1) for s in unique_test_canon], dtype=np.int64
    )
    train_lb_idxs = np.array(
        [lb_smiles_to_idx.get(s, -1) for s in train_canon], dtype=np.int64
    )
    train_hdf5_idxs = np.array(
        [hdf5_smiles_to_idx.get(s, -1) for s in train_canon], dtype=np.int64
    )
    print(
        f"  test:  lb_missing={int((test_lb_idxs < 0).sum())}/{n_test}, "
        f"hdf5_missing={int((test_hdf5_idxs < 0).sum())}/{n_test}"
    )
    print(
        f"  train: lb_missing={int((train_lb_idxs < 0).sum())}/{n_train}, "
        f"hdf5_missing={int((train_hdf5_idxs < 0).sum())}/{n_train}"
    )

    vals = np.full((n_test, n_train), mces_cap, dtype=np.float32)

    test_lb_valid = test_lb_idxs >= 0
    train_lb_valid = train_lb_idxs >= 0
    if test_lb_valid.any() and train_lb_valid.any():
        print("  gathering lb_matrix values ...")
        ti = test_lb_idxs[test_lb_valid][:, None]
        tj = train_lb_idxs[train_lb_valid][None, :]
        fidx = _lb_condensed_idx(ti, tj)
        gathered = lb[fidx.ravel()].reshape(fidx.shape).astype(np.float32)
        vals[np.ix_(test_lb_valid, train_lb_valid)] = gathered

    test_hdf5_valid = test_hdf5_idxs >= 0
    train_hdf5_valid = train_hdf5_idxs >= 0
    if test_hdf5_valid.any() and train_hdf5_valid.any():
        print("  gathering hdf5 values (max with lb_matrix) ...")
        ti = test_hdf5_idxs[test_hdf5_valid][:, None]
        tj = train_hdf5_idxs[train_hdf5_valid][None, :]
        fidx2 = _hdf5_condensed_idx(ti, tj, hdf5_n)
        gathered2 = hdf5_mces[fidx2.ravel()].reshape(fidx2.shape)
        sub_rows = np.ix_(test_hdf5_valid, train_hdf5_valid)
        vals[sub_rows] = np.maximum(vals[sub_rows], gathered2)

    vals = np.clip(vals, 0.0, mces_cap)
    nn_idx = vals.argmin(axis=1)
    min_val = vals[np.arange(n_test), nn_idx]

    no_source = ~(test_lb_valid | test_hdf5_valid)
    n_no_source = int(no_source.sum())
    print(
        f"  {n_no_source}/{n_test} test molecules absent from BOTH lb_matrix and hdf5"
    )
    nn_idx = np.where(no_source, -1, nn_idx)
    min_val = np.where(no_source, np.nan, min_val)

    if (~no_source).any():
        found = min_val[~no_source]
        print(
            f"  oracle MCES to nearest train (found): mean={found.mean():.1f}, "
            f"median={np.median(found):.1f}, min={found.min():.1f}"
        )

    nn_idx_by_mol = dict(zip(unique_test_canon, nn_idx.tolist()))
    min_val_by_mol = dict(zip(unique_test_canon, min_val.tolist()))
    return nn_idx_by_mol, min_val_by_mol


# ── Main ──────────────────────────────────────────────────────────────────────


def save_intermediates(
    out_dir: Path,
    split: str,
    train_smiles,
    test_smiles,
    nn_indices: np.ndarray,
    nn_mces: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_nn_indices.npy", nn_indices)
    np.save(out_dir / f"{split}_nn_mces.npy", nn_mces)
    (out_dir / "train_smiles.json").write_text(json.dumps(train_smiles))
    (out_dir / f"{split}_smiles.json").write_text(json.dumps(test_smiles))
    print(f"Intermediates saved to {out_dir}/")


def run(
    mgf: str,
    candidates: str,
    lb_matrix: str,
    lb_smiles: str,
    hdf5_mces_path: str,
    split: str = "test",
    output_tsv: str | None = None,
    intermediates_dir: str | None = None,
    mces_cap: float = 40.0,
):
    print(f"\nLoading candidate JSON from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)

    print(f"\nLoading spectra (split={split}, ref=train) ...")
    train_smiles, _ = load_spectra(mgf, "train")
    test_smiles, _ = load_spectra(mgf, split)
    print(f"  train: {len(train_smiles)}  {split}: {len(test_smiles)}")

    # Deduplicate training: keep first spectrum per unique canonical SMILES --
    # same reasoning as simba_retrieval.py/cosine_retrieval.py: each training
    # molecule represented exactly once, so argmin GT-MCES maps unambiguously
    # to a unique molecule -- fair comparison against the other two NN-transfer
    # rows for the same split.
    print("Deduplicating training spectra (first per unique molecule) ...")
    _seen: set[str] = set()
    _dedup_smi = []
    for _smi in train_smiles:
        _c = canonicalize(_smi)
        if _c not in _seen:
            _seen.add(_c)
            _dedup_smi.append(_smi)
    train_smiles = _dedup_smi
    print(f"  After dedup: {len(train_smiles)} unique training molecules")
    train_canon = [canonicalize(s) for s in train_smiles]

    # Build candidate lists aligned to test spectra, canonicalize keys
    print("\nBuilding candidate lists ...")
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}
    cand_lists = [cand_json_canon.get(canonicalize(s), []) for s in test_smiles]
    n_missing = sum(1 for c in cand_lists if not c)
    if n_missing:
        print(f"  Warning: {n_missing}/{len(test_smiles)} queries have no candidates")

    # Oracle NN transfer (GT MCES), once per unique test molecule. Done BEFORE
    # fingerprinting -- unlike simba_retrieval.py/cosine_retrieval.py, where
    # the nearest train neighbor is only known after computing embeddings (so
    # the whole train pool must be fingerprinted upfront since any of them
    # could turn out to be someone's neighbor), this oracle's NN pick is a
    # pure lb_matrix/hdf5 lookup, independent of fingerprints. So we find out
    # which train molecules are actually used FIRST, and only fingerprint
    # those (at most one per unique test molecule, often fewer since several
    # test molecules can share the same nearest train neighbor) instead of
    # the entire (much larger) train pool.
    print("\nLoading GT-MCES sources ...")
    lb_smiles_to_idx, lb = load_lb_matrix(lb_matrix, lb_smiles)
    hdf5_smiles_to_idx, hdf5_mces, hdf5_n = load_hdf5_mces(hdf5_mces_path)

    print("\nOracle nearest-neighbor transfer (GT MCES) ...")
    unique_test_canon = list(dict.fromkeys(canonicalize(s) for s in test_smiles))
    nn_idx_by_mol, nn_mces_by_mol = oracle_nn_by_molecule(
        unique_test_canon,
        train_canon,
        lb_smiles_to_idx,
        lb,
        hdf5_smiles_to_idx,
        hdf5_mces,
        hdf5_n,
        mces_cap=mces_cap,
    )

    nn_indices = np.array(
        [nn_idx_by_mol[canonicalize(s)] for s in test_smiles], dtype=np.int64
    )
    nn_mces = np.array(
        [nn_mces_by_mol[canonicalize(s)] for s in test_smiles], dtype=np.float32
    )
    used_train_smiles = {train_smiles[i] for i in set(nn_indices.tolist()) if i >= 0}
    print(
        f"  {len(used_train_smiles):,}/{len(train_smiles):,} train molecules actually "
        "selected as a nearest neighbor (only these get fingerprinted, not the whole pool)"
    )

    # Fingerprints: only the train molecules actually used above, plus every
    # candidate molecule (unavoidable -- all of them get Tanimoto-scored).
    print("\nComputing Morgan fingerprints (used train molecules + all candidates) ...")
    all_smiles = used_train_smiles | {c for lst in cand_lists for c in lst}
    fp_lookup = build_fp_lookup(all_smiles)

    transferred_fps = np.zeros((len(test_smiles), 2048), dtype=np.uint8)
    valid = nn_indices >= 0
    for i in np.flatnonzero(valid):
        transferred_fps[i] = fp_lookup[train_smiles[nn_indices[i]]]

    if intermediates_dir:
        save_intermediates(
            Path(intermediates_dir),
            split,
            train_smiles,
            test_smiles,
            nn_indices,
            nn_mces,
        )

    # Evaluate
    print("\nScoring candidates ...")
    results = compute_hit_rates_np(test_smiles, cand_lists, transferred_fps, fp_lookup)

    print(f"\n=== Oracle GT-MCES retrieval ({split}, n={len(test_smiles)}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if output_tsv:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "split": split,
                    "model": "oracle_gt_mces_nn_transfer",
                    "n": len(test_smiles),
                    **results,
                }
            ]
        ).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results


def main():
    p = argparse.ArgumentParser(
        description="GT-MCES oracle retrieval: NN transfer by true structural "
        "distance + Tanimoto ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidates", required=True, help="Candidate JSON {smiles: [cands]}"
    )
    p.add_argument("--lb_matrix", required=True, help="lb_matrix.npy")
    p.add_argument("--lb_smiles", required=True, help="lb_matrix.smiles.txt")
    p.add_argument("--hdf5_mces_path", required=True, help="all_smiles_mces.hdf5")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory to save NN indices/MCES values",
    )
    p.add_argument(
        "--mces_cap",
        type=float,
        default=40.0,
        help="Fallback/clip value when a molecule is missing from both sources",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
