"""3e: does SIMBA's MCES prediction generalize from test-to-test (in-distribution,
MassSpecGym-native molecules) pairs to test-to-candidate (PubChem, formula-matched,
scored via ICEBERG-predicted spectra) pairs?

Compares SIMBA's predicted MCES (from cosine similarity of its own embeddings,
via predicted_mces = mces_max_value * (1 - sim)) against ground truth for both
populations, reusing embeddings + SMILES already saved from a prior
simba_retrieval_iceberg.py run (--intermediates_dir) — no re-embedding needed.

NO AVERAGING ANYWHERE, on either side, for either population — every
embedding used below is one specific spectrum's own individual embedding.
Two things this rules out, both found (and fixed) in earlier versions of
this script:
  - Averaging the query (test) side's several spectra of the same molecule
    into one embedding. Confirmed to inflate accuracy (hit@1 16.3% averaged
    vs. 10.2% real) by denoising the query in a way real evaluation never
    benefits from.
  - Averaging a candidate's several per-adduct ICEBERG-predicted embeddings
    into one per-molecule embedding. A candidate embedded under [M+H]+ and
    again under [M+Na]+ are two DIFFERENT (in silico) spectra — two separate
    instances, not two noisy measurements of the same thing — so blending
    them loses exactly the distinction simba_retrieval_iceberg.py's own
    rank_candidates relies on (it matches each query only to the candidate
    row embedded under that SAME query's adduct). This script now does the
    same, via test_adducts.json/candidate_adducts.json saved alongside the
    embeddings — a candidate with no embedding for a given query's adduct is
    dropped for that query, never substituted with a different adduct's row
    or an average across adducts.

EXCLUSION MODE (read this before trusting a number from either population):
  - test-to-candidate: self (the true candidate — the SAME molecule's own
    ICEBERG-predicted spectrum, matched on the query's OWN adduct) is
    INCLUDED (GT=0, added back in via add_self_pairs — prepare_gt_mces_retrieval.py
    never computed it since it's trivially known). A self pair whose
    (true_smiles, query_adduct) has no ICEBERG-predicted embedding is
    dropped, not substituted with a real-spectrum embedding — test-to-candidate
    is specifically about the ICEBERG-predicted modality, so a real spectrum
    is not a correct fallback. Label: "self included".
  - test-to-test: the true self-comparison (a spectrum against itself) is
    the only thing excluded. Same-molecule-*different*-spectrum pairs are
    INCLUDED (GT=0) alongside the usual cross-molecule pairs. Label: "self
    spectrum excluded, same-molecule spectra included".

  - test-to-test: one dense (n_valid_spectra, n_valid_spectra) matrix built
    directly from raw per-spectrum embeddings on BOTH sides (see
    score_test_to_test_no_averaging) — no per-molecule embedding of any kind
    involved — scored against every mined pair in
    data/massspecgym/preprocessing_msg_exact_mces_1020's official-test-fold
    pairs file.
  - test-to-candidate: ragged (ICEBERG-predicted, formula-matched candidate
    pools vary in size per query) — expanded to one scored row per test
    spectrum via expand_and_score_ragged, using the 584,340 molecule-level
    pairs + exact-MCES values from tools/prepare_gt_mces_retrieval.py, PLUS
    the true-candidate (self) pair added back in, each candidate matched to
    the query's own adduct.

Usage:
    uv run python tools/ood_generalization_check.py \\
        --intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --test_to_test_prepro_dir /path/to/preprocessing_msg_exact_mces_1020 \\
        --mces_max_value 40
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from load_test_to_test_gt_mces import (
    load_test_to_test_gt_lookup,  # noqa: F401 (kept for ad-hoc use)
)
from scipy.stats import spearmanr
from simba_retrieval import canonicalize


def build_candidate_embeddings_by_smi_adduct(
    smiles_list: list[str], adducts_list: list[str], embeddings: torch.Tensor
) -> dict[tuple[str, str], np.ndarray]:
    """One L2-normalized embedding per (canonical smiles, adduct) row — NO
    averaging across adducts, ever. A candidate embedded under [M+H]+ and
    again under [M+Na]+ are two different (in silico) spectra, matched to
    different queries by simba_retrieval_iceberg.py's own rank_candidates;
    collapsing them into one averaged vector (an earlier version of this
    script did that, keyed by smiles alone) blends predictions that were
    never meant to represent the same comparison.

    A handful of (canonical smiles, adduct) collisions come from a
    different, unrelated source: the SAME molecule + SAME adduct appearing
    twice in the candidate TSV under two different raw SMILES strings that
    happen to canonicalize to the same structure (an upstream candidate-list
    duplication, not a same-molecule-different-adduct case). Those are NOT
    averaged either — the first embedding seen for that (molecule, adduct)
    is kept and the rest dropped, logged via the returned dupe count so this
    stays visible rather than silent."""
    emb_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1).numpy()
    out: dict[tuple[str, str], np.ndarray] = {}
    n_dupes = 0
    for i, (s, adduct) in enumerate(zip(smiles_list, adducts_list)):
        key = (canonicalize(s), adduct)
        if key in out:
            n_dupes += 1
            continue
        out[key] = emb_norm[i]
    if n_dupes:
        print(
            f"  {n_dupes} duplicate (canonical smiles, adduct) candidate rows "
            "(same molecule + adduct, different raw SMILES spelling in the candidate "
            "list) — kept the first embedding seen for each, never averaged"
        )
    return out


def build_embedding_matrix(
    idx_to_smiles: list[str], smi_to_emb: dict[str, np.ndarray], emb_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Align an (n_mols, emb_dim) matrix + boolean mask to idx_to_smiles's own
    row order, so pair arrays indexed by mol_idx can gather with plain numpy
    fancy indexing (no per-pair Python loop). smi_to_emb here is expected to
    already be filtered to a single adduct — see expand_and_score_ragged."""
    n = len(idx_to_smiles)
    mat = np.zeros((n, emb_dim), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, smi in enumerate(idx_to_smiles):
        emb = smi_to_emb.get(smi)
        if emb is not None:
            mat[i] = emb
            has[i] = True
    return mat, has


def build_smi_to_spectrum_indices(smiles_list: list[str]) -> dict[str, np.ndarray]:
    """canonical smiles -> array of row indices in the parallel (raw, per-
    spectrum) embeddings tensor — every consumer of test spectra keeps every
    individual spectrum, never averages them together."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        groups.setdefault(canonicalize(s), []).append(i)
    return {k: np.array(v, dtype=int) for k, v in groups.items()}


def add_self_pairs(pair_arr: np.ndarray) -> np.ndarray:
    """Add back the trivial (query_mol_idx, query_mol_idx, 0.0) true-match
    rows that prepare_gt_mces_retrieval.py deliberately excludes (GT=0,
    trivially known, never computed) — for when self should be INCLUDED in
    the test-to-candidate pool. One row per unique query molecule already
    present in pair_arr's first column."""
    unique_q = np.unique(pair_arr[:, 0].astype(int))
    self_rows = np.zeros((len(unique_q), pair_arr.shape[1]), dtype=pair_arr.dtype)
    self_rows[:, 0] = unique_q
    self_rows[:, 1] = unique_q
    return np.vstack([pair_arr, self_rows])


def expand_and_score_ragged(
    pair_arr: np.ndarray,  # (N, >=3): [test_mol_idx, other_mol_idx, ..., gt]
    idx_to_smiles: list[str],
    test_smiles_raw: list[str],
    test_adducts_raw: list[str],
    test_embeddings: torch.Tensor,
    other_smi_adduct_to_emb: dict[tuple[str, str], np.ndarray],
    mces_max_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand a ragged test-to-X molecule-pair array into one scored row per
    test spectrum belonging to that molecule (the query side's own
    embedding). The OTHER side is matched to each query spectrum's OWN
    adduct specifically — other_smi_adduct_to_emb is keyed by (smiles,
    adduct), mirroring simba_retrieval_iceberg.py's rank_candidates, which
    ranks each query only against candidates embedded under that SAME
    adduct. A candidate molecule with no embedding for the query's adduct is
    dropped for that query — no averaging across its other adducts, no
    fallback to a different adduct.

    Partitioned by the (small) set of distinct adduct values rather than
    looped per pair row: for each adduct, build one (n_mols, emb_dim) "other
    side" matrix from only that adduct's rows, and score every query
    spectrum with that adduct against it in one vectorized batch.

    Returns (spec_idx, other_idx, gt, pred) — all the same length, fully
    aligned, so callers can either flatten (gt, pred) for MAE/Spearman or
    group by spec_idx for per-spectrum min/mean/max aggregation.
    """
    emb_dim = next(iter(other_smi_adduct_to_emb.values())).shape[0]
    smi_to_specs = build_smi_to_spectrum_indices(test_smiles_raw)
    test_emb_norm = torch.nn.functional.normalize(test_embeddings, p=2, dim=-1).numpy()
    test_adducts_arr = np.array(test_adducts_raw)

    a_idx_all = pair_arr[:, 0].astype(int)
    b_idx_all = pair_arr[:, 1].astype(int)
    gt_all = pair_arr[:, -1].astype(float)

    spec_parts, b_parts, gt_parts, pred_parts = [], [], [], []
    for adduct in sorted(set(test_adducts_raw)):
        smi_to_emb_this_adduct = {
            smi: emb
            for (smi, adct), emb in other_smi_adduct_to_emb.items()
            if adct == adduct
        }
        if not smi_to_emb_this_adduct:
            continue
        other_mat, other_has = build_embedding_matrix(
            idx_to_smiles, smi_to_emb_this_adduct, emb_dim
        )

        keep = other_has[b_idx_all]
        a_idx, b_idx, gt = a_idx_all[keep], b_idx_all[keep], gt_all[keep]

        unique_a, inverse = np.unique(a_idx, return_inverse=True)
        spec_lists = [
            smi_to_specs.get(idx_to_smiles[a], np.zeros(0, dtype=int)) for a in unique_a
        ]
        # only this adduct's spectra of each molecule belong in this batch
        spec_lists = [s[test_adducts_arr[s] == adduct] for s in spec_lists]
        counts = np.array([len(s) for s in spec_lists])
        counts_per_row = counts[inverse]

        keep2 = counts_per_row > 0
        b_idx2, gt2, inverse2 = b_idx[keep2], gt[keep2], inverse[keep2]
        counts_per_row2 = counts_per_row[keep2]

        expanded_spec = (
            np.concatenate([spec_lists[i] for i in inverse2])
            if len(inverse2)
            else np.zeros(0, dtype=int)
        )
        expanded_b = np.repeat(b_idx2, counts_per_row2)
        expanded_gt = np.repeat(gt2, counts_per_row2)

        ea = test_emb_norm[expanded_spec]
        eb = other_mat[expanded_b]
        sims = (ea * eb).sum(axis=1)
        pred = np.clip(mces_max_value * (1.0 - sims), 0.0, None)

        spec_parts.append(expanded_spec)
        b_parts.append(expanded_b)
        gt_parts.append(expanded_gt)
        pred_parts.append(pred)

    if not spec_parts:
        empty_i, empty_f = np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_f

    return (
        np.concatenate(spec_parts),
        np.concatenate(b_parts),
        np.concatenate(gt_parts),
        np.concatenate(pred_parts),
    )


def score_test_to_test_no_averaging(
    gt_pairs: np.ndarray,
    idx_to_smiles: list[str],
    test_smiles_raw: list[str],
    test_embeddings: torch.Tensor,
    mces_max_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full test-to-test scoring with NO embedding averaging anywhere — every
    one of the valid test spectra is compared against every OTHER individual
    test spectrum's own embedding directly, via a single dense
    (n_valid_spectra, n_valid_spectra) matrix built from raw per-spectrum
    embeddings on BOTH sides.

    GT per spectrum-pair: cross-molecule pairs look up the mined
    molecule-level GT (gt_pairs); same-molecule-different-spectrum pairs are
    GT=0 (trivially, same structure); only the literal self spectrum (i==j)
    is excluded.

    Returns (pred_matrix, gt_matrix, valid_mask, spec_molidx) — all
    (n_valid_spectra, n_valid_spectra) except spec_molidx. Callers flatten
    with valid_mask for MAE/Spearman, or reduce per-row (nanmin/nanmean/
    nanmax on np.where(valid_mask, matrix, nan)) for per-spectrum
    distribution plots.
    """
    n_mols = len(idx_to_smiles)
    smi_to_molidx = {smi: i for i, smi in enumerate(idx_to_smiles)}
    spec_molidx_all = np.array(
        [smi_to_molidx.get(canonicalize(s), -1) for s in test_smiles_raw]
    )
    valid_spec = spec_molidx_all >= 0
    spec_molidx = spec_molidx_all[valid_spec]

    gt_matrix_mol = np.full((n_mols, n_mols), np.nan, dtype=np.float32)
    a = gt_pairs[:, 0].astype(int)
    b = gt_pairs[:, 1].astype(int)
    v = gt_pairs[:, -1].astype(float)
    gt_matrix_mol[a, b] = v
    gt_matrix_mol[b, a] = v

    test_emb_norm = torch.nn.functional.normalize(test_embeddings, p=2, dim=-1).numpy()
    valid_emb = test_emb_norm[valid_spec]

    sims_matrix = valid_emb @ valid_emb.T
    pred_matrix = np.clip(mces_max_value * (1.0 - sims_matrix), 0.0, None).astype(
        np.float32
    )

    same_molecule = spec_molidx[:, None] == spec_molidx[None, :]
    gt_matrix = gt_matrix_mol[spec_molidx][:, spec_molidx].copy()
    gt_matrix[same_molecule] = 0.0  # same-molecule-different-spectrum -> GT=0

    rows = np.arange(len(spec_molidx))
    pred_matrix[rows, rows] = np.nan  # literal self spectrum excluded
    gt_matrix[rows, rows] = np.nan

    valid_mask = ~np.isnan(gt_matrix) & ~np.isnan(pred_matrix)

    return pred_matrix, gt_matrix, valid_mask, spec_molidx


def report(name: str, pred: np.ndarray, gt: np.ndarray) -> dict:
    mae = float(np.abs(pred - gt).mean())
    rho, _ = spearmanr(pred, gt)
    print(f"=== {name} (n={len(gt):,}) ===")
    print(f"  MAE:      {mae:.3f}")
    print(f"  Spearman: {rho:.3f}")
    return {"n": len(gt), "mae": mae, "spearman": float(rho)}


def run(
    intermediates_dir: str,
    gt_mces_dir: str,
    test_to_test_prepro_dir: str,
    mces_max_value: float = 40.0,
) -> dict:
    inter = Path(intermediates_dir)

    print("Loading saved embeddings + SMILES + adducts ...")
    test_embeddings = torch.load(inter / "test_embeddings.pt", map_location="cpu")
    test_smiles = json.loads((inter / "test_smiles.json").read_text())
    test_adducts = json.loads((inter / "test_adducts.json").read_text())
    candidate_embeddings = torch.load(
        inter / "candidate_embeddings.pt", map_location="cpu"
    )
    candidate_smiles = json.loads((inter / "candidate_smiles.json").read_text())
    candidate_adducts = json.loads((inter / "candidate_adducts.json").read_text())
    print(
        f"  {test_embeddings.shape[0]} test spectra, {candidate_embeddings.shape[0]} candidate spectra"
    )

    print(
        "Building per-(smiles, adduct) candidate embeddings (no averaging across adducts) ..."
    )
    cand_smi_adduct_to_emb = build_candidate_embeddings_by_smi_adduct(
        candidate_smiles, candidate_adducts, candidate_embeddings
    )
    print(
        f"  {len(cand_smi_adduct_to_emb)} unique (candidate molecule, adduct) embeddings"
    )

    print(
        "\n--- test-to-test (dense, individual spectra on BOTH sides; self SPECTRUM excluded) ---"
    )
    with open(Path(test_to_test_prepro_dir) / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    tt_idx_to_smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()
    tt_pairs = np.load(
        Path(test_to_test_prepro_dir)
        / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy"
    )
    print(
        f"  {len(tt_pairs):,} mined cross-molecule pairs, {len(tt_idx_to_smiles)} molecules"
    )

    tt_pred_matrix, tt_gt_matrix, tt_valid_mask, _ = score_test_to_test_no_averaging(
        tt_pairs, tt_idx_to_smiles, test_smiles, test_embeddings, mces_max_value
    )
    tt_pred = tt_pred_matrix[tt_valid_mask]
    tt_gt = tt_gt_matrix[tt_valid_mask]
    tt_result = report(
        "test-to-test (self spectrum excluded, no averaging)", tt_pred, tt_gt
    )

    print(
        "\n--- test-to-candidate (ragged, per-spectrum query, own-adduct candidate match; self included) ---"
    )
    gt_dir = Path(gt_mces_dir)
    tc_idx_to_smiles = gt_dir.joinpath("smiles.txt").read_text().splitlines()
    tc_pairs = np.load(gt_dir / "mces_exact.npy")
    valid = (tc_pairs[:, 2] >= 0) & ~np.isnan(tc_pairs[:, 2])
    tc_pairs = tc_pairs[valid]
    tc_pairs = add_self_pairs(tc_pairs)
    print(
        f"  {len(tc_pairs):,} test-to-candidate molecule pairs (incl. self), {len(tc_idx_to_smiles)} molecules"
    )

    _, _, tc_gt, tc_pred = expand_and_score_ragged(
        tc_pairs,
        tc_idx_to_smiles,
        test_smiles,
        test_adducts,
        test_embeddings,
        cand_smi_adduct_to_emb,
        mces_max_value,
    )
    tc_result = report(
        "test-to-candidate (self included, own-adduct match, no averaging)",
        tc_pred,
        tc_gt,
    )

    print("\n=== Summary ===")
    print(
        f"  test-to-test (self spectrum excluded, no averaging):               MAE {tt_result['mae']:.3f}, Spearman {tt_result['spearman']:.3f} (n={tt_result['n']:,})"
    )
    print(
        f"  test-to-candidate (self included, own-adduct match, no averaging): MAE {tc_result['mae']:.3f}, Spearman {tc_result['spearman']:.3f} (n={tc_result['n']:,})"
    )

    return {"test_to_test": tt_result, "test_to_candidate": tc_result}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--intermediates_dir",
        required=True,
        help="Dir with test_embeddings.pt/test_smiles.json/test_adducts.json/"
        "candidate_embeddings.pt/candidate_smiles.json/candidate_adducts.json from a "
        "prior simba_retrieval_iceberg.py --intermediates_dir run",
    )
    p.add_argument(
        "--gt_mces_dir",
        required=True,
        help="Dir with smiles.txt + mces_exact.npy from tools/prepare_gt_mces_retrieval.py",
    )
    p.add_argument(
        "--test_to_test_prepro_dir",
        required=True,
        help="Official-split preprocessing dir with the exact-refined test-fold pairs "
        "(data/massspecgym/preprocessing_msg_exact_mces_1020)",
    )
    p.add_argument("--mces_max_value", type=float, default=40.0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
