"""Build a GT-MCES lookup (smiles.txt + mces_exact.npy, same schema
load_gt_mces_lookup expects) for an arbitrary test-fold's test-to-candidate
pairs, from the two pre-existing all-vs-all MCES matrices already used by
oracle_retrieval_gt_mces.py -- no new exact-MCES computation, no asimov2
needed. Written to let mces_calibration_plots.py / mces_top1_diagnostics.py /
build_retrieval_comparison_table.py run on a split (e.g. Gaetan) that
prepare_gt_mces_retrieval.py's asimov2-only pipeline was never run for.

data/massspecgym/lb_matrix.npy + .smiles.txt: 240,637-molecule condensed
lower-bound matrix (tight for MCES>=10, underestimate for <10).
data/massspecgym/data/auxiliary/all_smiles_mces.hdf5: 34,731-molecule
condensed matrix (exact for MCES<10, weak lower bound for >=10).
Combined via max(lb, hdf5) -- same convention as
prepare_msg_*_split_max_lb_hdf5.py's own training-data MCES and
oracle_retrieval_gt_mces.py's NN-transfer oracle.

Unlike oracle_retrieval_gt_mces.py's oracle_nn_by_molecule (which defaults
every unresolved cell to mces_cap before taking an argmin -- a capped value
can never be picked as the minimum anyway, so that default is harmless
there), this script must NOT default unresolved pairs to any value:
load_gt_mces_lookup's downstream convention treats -1/NaN as "no reliable
GT value, drop it," and a calibration plot fed a fake "GT=40" for pairs
neither source actually covers would be silently wrong. A pair's MCES here
is max(lb, hdf5) if BOTH sources resolve it, whichever source resolves it
if only ONE does, or NaN (dropped downstream) if NEITHER does.

True-match (test molecule vs its own true candidate) pairs are excluded,
matching prepare_gt_mces_retrieval.py's own convention -- their GT MCES is
trivially 0 and every consumer of load_gt_mces_lookup already credits a
correct hit as MCES=0 directly rather than looking it up.

Memory: lb_matrix.npy is loaded fully into RAM (~116 GB) -- run on a
high-memory node with --mem set explicitly (see oracle_retrieval_gt_mces.py
for why: memory-mapping it turns ~80M scattered reads into a multi-hour
random-I/O job on this filesystem, vs one ~116GB sequential read).

Usage:
    uv run python tools/build_gt_mces_from_matrices.py \\
        --mgf /path/to/gaetan_test.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --split test \\
        --lb_matrix /path/to/lb_matrix.npy \\
        --lb_smiles /path/to/lb_matrix.smiles.txt \\
        --hdf5_mces_path /path/to/all_smiles_mces.hdf5 \\
        --output_dir /path/to/gt_mces_gaetan_test
"""

import argparse
import json
from pathlib import Path

import numpy as np
from oracle_retrieval_gt_mces import (
    _hdf5_condensed_idx,
    _lb_condensed_idx,
    load_hdf5_mces,
    load_lb_matrix,
)
from prepare_gt_mces_retrieval import _load_test_fold_smiles, canon
from simba_retrieval import canonicalize


def gather_pairwise_mces(
    a_canon: np.ndarray,
    b_canon: np.ndarray,
    lb_smiles_to_idx: dict,
    lb: np.ndarray,
    hdf5_smiles_to_idx: dict,
    hdf5_mces: np.ndarray,
    hdf5_n: int,
) -> np.ndarray:
    """Elementwise (not grid) gather: result[k] = MCES(a_canon[k], b_canon[k]).
    NaN where neither source resolves the pair -- see module docstring."""
    n = len(a_canon)
    result = np.full(n, np.nan, dtype=np.float64)

    a_lb = np.array([lb_smiles_to_idx.get(s, -1) for s in a_canon])
    b_lb = np.array([lb_smiles_to_idx.get(s, -1) for s in b_canon])
    lb_valid = (a_lb >= 0) & (b_lb >= 0)
    if lb_valid.any():
        idx = _lb_condensed_idx(a_lb[lb_valid], b_lb[lb_valid])
        result[lb_valid] = lb[idx]

    a_hdf5 = np.array([hdf5_smiles_to_idx.get(s, -1) for s in a_canon])
    b_hdf5 = np.array([hdf5_smiles_to_idx.get(s, -1) for s in b_canon])
    hdf5_valid = (a_hdf5 >= 0) & (b_hdf5 >= 0)
    if hdf5_valid.any():
        idx2 = _hdf5_condensed_idx(a_hdf5[hdf5_valid], b_hdf5[hdf5_valid], hdf5_n)
        gathered = hdf5_mces[idx2]
        existing = result[hdf5_valid]
        result[hdf5_valid] = np.where(
            np.isnan(existing), gathered, np.maximum(existing, gathered)
        )
    return result


def run(
    mgf: str,
    candidates: str,
    split: str,
    lb_matrix: str,
    lb_smiles: str,
    hdf5_mces_path: str,
    output_dir: str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {split}-fold query molecules from {mgf} (fast manual scan) ...")
    query_smis_raw = _load_test_fold_smiles(mgf) if split == "test" else None
    if query_smis_raw is None:
        # _load_test_fold_smiles hardcodes fold=="test" -- reimplement inline
        # for val without duplicating the whole function.
        query_smis_raw = set()
        fold_val, smi_val = None, None
        with open(mgf) as fh:
            for line in fh:
                line = line.strip()
                if line == "BEGIN IONS":
                    fold_val, smi_val = None, None
                elif line.upper().startswith("FOLD="):
                    fold_val = line.split("=", 1)[1].strip()
                elif line.upper().startswith("SMILES="):
                    smi_val = line.split("=", 1)[1].strip()
                elif line == "END IONS" and fold_val == split and smi_val:
                    query_smis_raw.add(smi_val)
    query_canon = {canon(s) for s in query_smis_raw} - {None}
    print(f"  {len(query_canon)} unique canonical {split}-fold molecules")

    print(f"\nLoading candidate pools from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}

    pairs_a, pairs_b = [], []
    n_no_candidates = 0
    for q in query_canon:
        cand_list = cand_json_canon.get(q, [])
        if not cand_list:
            n_no_candidates += 1
            continue
        for c in cand_list:
            c_canon = canonicalize(c)
            if c_canon == q:
                continue  # true-match pair, GT MCES=0 by definition, excluded
            pairs_a.append(q)
            pairs_b.append(c_canon)
    print(
        f"  {len(pairs_a):,} (query, candidate) pairs to resolve "
        f"({n_no_candidates} query molecules had no candidates)"
    )

    print("\nLoading GT-MCES sources ...")
    lb_smiles_to_idx, lb = load_lb_matrix(lb_matrix, lb_smiles)
    hdf5_smiles_to_idx, hdf5_mces, hdf5_n = load_hdf5_mces(hdf5_mces_path)

    print("\nGathering max(lb_matrix, hdf5) per pair ...")
    a_arr = np.array(pairs_a)
    b_arr = np.array(pairs_b)
    mces_vals = gather_pairwise_mces(
        a_arr, b_arr, lb_smiles_to_idx, lb, hdf5_smiles_to_idx, hdf5_mces, hdf5_n
    )
    n_resolved = int(np.isfinite(mces_vals).sum())
    print(
        f"  {n_resolved:,}/{len(mces_vals):,} pairs resolved from at least one source"
    )

    valid = np.isfinite(mces_vals)
    a_valid, b_valid, mces_valid = a_arr[valid], b_arr[valid], mces_vals[valid]

    print("\nBuilding unified smiles.txt index ...")
    smiles_list = sorted(set(a_valid.tolist()) | set(b_valid.tolist()))
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}
    idx_a = np.array([smi_to_idx[s] for s in a_valid], dtype=np.int64)
    idx_b = np.array([smi_to_idx[s] for s in b_valid], dtype=np.int64)

    out_arr = np.stack([idx_a, idx_b, mces_valid.astype(np.float32)], axis=1).astype(
        np.float32
    )
    np.save(out_dir / "mces_exact.npy", out_arr)
    (out_dir / "smiles.txt").write_text("\n".join(smiles_list))
    print(
        f"\nSaved {len(smiles_list):,} molecules to {out_dir / 'smiles.txt'}, "
        f"{len(out_arr):,} pairs to {out_dir / 'mces_exact.npy'}"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--lb_matrix", required=True)
    p.add_argument("--lb_smiles", required=True)
    p.add_argument("--hdf5_mces_path", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
