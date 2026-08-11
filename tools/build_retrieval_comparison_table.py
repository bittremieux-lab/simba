"""Build the "ultimate" per-(test spectrum, candidate) comparison table
(item 8b groundwork) — one row per test-to-candidate pair, with everything
needed to build any later plot without re-running SIMBA or the cosine
baseline again.

Starts from the FULL formula-matched candidate pool per query (candidates.json,
no pre-filtering) rather than silently dropping candidates lacking a usable
embedding — a missing embedding shows up in the table itself as a null
candidate_adduct + null ranks, so this run also directly verifies (rather
than assumes) whether that ever actually happens, and how often.

Columns:
  test_spec_idx     0..N-1, unique per test spectrum
  test_smiles       canonical SMILES of the query molecule
  test_adduct       this spectrum's own measured adduct
  candidate_smiles  canonical SMILES of the candidate (pool deduplicated by
                     canonical structure — a candidate listed twice under
                     different raw SMILES spellings is one row, not two)
  candidate_adduct  the adduct actually used to find this candidate's
                     embedding; equal to test_adduct when found, null when
                     no ICEBERG-predicted embedding exists for this
                     (candidate, test_adduct) pair at all
  simba_rank        rank by SIMBA-embedding cosine similarity (1 = most
                     similar); null if no SIMBA embedding
  cosine_rank       rank by raw binned-spectrum cosine similarity (1 = most
                     similar); null if no cosine embedding
  simba_similarity  raw SIMBA-embedding cosine similarity
  simba_mces        mces_max_value * (1 - simba_similarity) — a real,
                     calibrated quantity: SIMBA's cosine_no_head head is
                     trained so this approximates true MCES
  cosine_similarity raw binned-spectrum cosine similarity
  cosine_mces       mces_max_value * (1 - cosine_similarity) — NOT a
                     calibrated quantity (raw cosine similarity was never
                     trained to predict MCES); included as the same
                     convenience transform for side-by-side plotting only,
                     see ood_generalization_check.py's module docstring
  gt_mces           ground truth MCES(test molecule, candidate molecule);
                     0.0 for the true candidate, NaN if unresolved
  is_correct        1 for the true candidate (exactly one per
                     test_spec_idx), else 0
  n_peaks_test      raw peak count of the real test spectrum (straight from
                     the MGF, before any SIMBA preprocessing)
  n_peaks_candidate raw peak count of the candidate's own ICEBERG-predicted
                     spectrum for candidate_adduct (non-padding rows of the
                     top-100 sparse output); NaN if no ICEBERG embedding

SIMBA's and the cosine baseline's candidate embeddings are both built from
the exact same underlying candidate_tsv + preds.hdf5 rows (see
ood_generalization_check.py's module docstring), so a candidate missing
one method's embedding should be missing the other's too — this is checked
directly (not assumed) and reported as "SIMBA/cosine embedding-availability
mismatches" in the run log.

Usage:
    uv run python tools/build_retrieval_comparison_table.py \\
        --simba_intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --cosine_intermediates_dir /path/to/cosine_baseline_intermediates \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --output_csv /path/to/retrieval_comparison_table.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from ood_generalization_check import build_candidate_embeddings_by_smi_adduct
from simba_retrieval import canonicalize
from simba_retrieval_iceberg import load_gt_mces_lookup, load_iceberg_spectra
from tqdm.auto import tqdm


def count_test_peaks_in_order(mgf_path: str) -> list[int]:
    """Raw peak count per test-fold spectrum, in the same order
    load_spectra(mgf, "test") (and thus test_smiles.json/test_adducts.json)
    enumerates them. Manual MGF scan (fast — avoids matchms's slow
    per-spectrum object construction, same technique as
    plot_confusion_matrix_examples.py's extract_test_spectra_by_index)."""
    counts = []
    current_fold = None
    current_n = 0
    with open(mgf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current_fold = None
                current_n = 0
                continue
            if line == "END IONS":
                if current_fold == "test":
                    counts.append(current_n)
                continue
            if line.startswith("FOLD="):
                current_fold = line[len("FOLD=") :]
            elif current_fold == "test" and "=" not in line and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        current_n += 1
                    except ValueError:
                        pass
    return counts


def build_candidate_npeaks_lookup(
    candidate_tsv: str, iceberg_preds: str
) -> dict[tuple[str, str], int]:
    """(canonical smiles, adduct) -> raw peak count of that candidate's own
    ICEBERG-predicted spectrum. Same identity as
    build_candidate_embeddings_by_smi_adduct/build_candidate_row_lookup_by_smi_adduct
    above — first-seen-wins on a duplicate (smiles, adduct) row, never
    averaged."""
    print(f"Loading {candidate_tsv} for peak counts ...")
    cand_df = pd.read_csv(candidate_tsv, sep="\t")
    spec_ids = cand_df["spec"].tolist()
    print(
        f"  Reading peak counts for {len(spec_ids)} ICEBERG spectra from {iceberg_preds} ..."
    )
    iceberg_specs = load_iceberg_spectra(iceberg_preds, spec_ids)
    out: dict[tuple[str, str], int] = {}
    for smi, adduct, spec_id in zip(cand_df["smiles"], cand_df["ionization"], spec_ids):
        key = (canonicalize(smi), adduct)
        if key in out:
            continue
        spec = iceberg_specs.get(spec_id)
        if spec is not None:
            out[key] = len(spec[0])
    return out


def build_candidate_row_lookup_by_smi_adduct(
    smiles_list: list[str], adducts_list: list[str]
) -> dict[tuple[str, str], int]:
    """(canonical smiles, adduct) -> row index into the parallel sparse
    matrix — never averages; the first row seen for a given (molecule,
    adduct) wins on the rare raw-SMILES-spelling duplicate."""
    out: dict[tuple[str, str], int] = {}
    for i, (s, a) in enumerate(zip(smiles_list, adducts_list)):
        key = (canonicalize(s), a)
        if key not in out:
            out[key] = i
    return out


def rank_descending(sims: np.ndarray) -> np.ndarray:
    """1 = highest similarity. NaN stays NaN (no embedding -> no rank)."""
    ranks = np.full(len(sims), np.nan)
    valid = ~np.isnan(sims)
    order = np.argsort(-sims[valid])
    r = np.empty(order.shape, dtype=float)
    r[order] = np.arange(1, len(order) + 1)
    ranks[valid] = r
    return ranks


def run(
    simba_intermediates_dir: str,
    cosine_intermediates_dir: str,
    candidates: str,
    gt_mces_dir: str,
    mgf: str,
    candidate_tsv: str,
    iceberg_preds: str,
    output_csv: str,
    mces_max_value: float = 40.0,
) -> None:
    simba_inter = Path(simba_intermediates_dir)
    cosine_inter = Path(cosine_intermediates_dir)

    print("Loading SIMBA intermediates ...")
    test_embeddings = torch.load(simba_inter / "test_embeddings.pt", map_location="cpu")
    test_smiles = json.loads((simba_inter / "test_smiles.json").read_text())
    test_adducts = json.loads((simba_inter / "test_adducts.json").read_text())
    simba_cand_embeddings = torch.load(
        simba_inter / "candidate_embeddings.pt", map_location="cpu"
    )
    simba_cand_smiles = json.loads((simba_inter / "candidate_smiles.json").read_text())
    simba_cand_adducts = json.loads(
        (simba_inter / "candidate_adducts.json").read_text()
    )

    print("Loading cosine-baseline intermediates ...")
    cosine_test_mat = sp.load_npz(cosine_inter / "test_mat.npz")
    cosine_test_smiles = json.loads((cosine_inter / "test_smiles.json").read_text())
    cosine_test_adducts = json.loads((cosine_inter / "test_adducts.json").read_text())
    cosine_cand_mat = sp.load_npz(cosine_inter / "candidate_mat.npz")
    cosine_cand_smiles = json.loads(
        (cosine_inter / "candidate_smiles.json").read_text()
    )
    cosine_cand_adducts = json.loads(
        (cosine_inter / "candidate_adducts.json").read_text()
    )

    assert test_smiles == cosine_test_smiles, (
        "SIMBA and cosine intermediates disagree on test spectrum order/content — "
        "were they built from different MGF loads?"
    )
    assert test_adducts == cosine_test_adducts, (
        "SIMBA and cosine intermediates disagree on test spectrum adducts"
    )
    print(f"  {len(test_smiles)} test spectra (order verified identical across both)")

    simba_test_emb_norm = torch.nn.functional.normalize(
        test_embeddings, p=2, dim=-1
    ).numpy()
    simba_cand_smi_adduct_to_emb = build_candidate_embeddings_by_smi_adduct(
        simba_cand_smiles, simba_cand_adducts, simba_cand_embeddings
    )
    cosine_cand_row_lookup = build_candidate_row_lookup_by_smi_adduct(
        cosine_cand_smiles, cosine_cand_adducts
    )
    print(
        f"  {len(simba_cand_smi_adduct_to_emb)} SIMBA (candidate, adduct) embeddings, "
        f"{len(cosine_cand_row_lookup)} cosine (candidate, adduct) rows"
    )

    print(f"Loading candidate pools from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)
    query_candidates = {canonicalize(k): v for k, v in candidate_json.items()}

    print(f"Loading GT MCES lookup from {gt_mces_dir} ...")
    gt_lookup = load_gt_mces_lookup(gt_mces_dir)
    print(f"  {len(gt_lookup) // 2} unique (test, candidate) pairs with a GT value")

    print(f"Counting raw peaks per test spectrum from {mgf} ...")
    test_npeaks = count_test_peaks_in_order(mgf)
    assert len(test_npeaks) == len(test_smiles), (
        f"{len(test_npeaks)} test-fold spectra counted from the MGF vs "
        f"{len(test_smiles)} in the SIMBA intermediates — order/count mismatch"
    )
    cand_npeaks_lookup = build_candidate_npeaks_lookup(candidate_tsv, iceberg_preds)
    print(f"  {len(cand_npeaks_lookup)} (candidate, adduct) peak counts")

    cols = {
        "test_spec_idx": [],
        "test_smiles": [],
        "test_adduct": [],
        "candidate_smiles": [],
        "candidate_adduct": [],
        "simba_rank": [],
        "cosine_rank": [],
        "simba_similarity": [],
        "simba_mces": [],
        "cosine_similarity": [],
        "cosine_mces": [],
        "gt_mces": [],
        "is_correct": [],
        "n_peaks_test": [],
        "n_peaks_candidate": [],
    }

    n_no_pool = 0
    n_no_self_in_pool = 0
    n_embedding_availability_mismatch = 0
    n_neither_has_embedding = 0

    for spec_i, (smi, adduct) in enumerate(
        tqdm(list(zip(test_smiles, test_adducts)), desc="Building comparison rows")
    ):
        q_canon = canonicalize(smi)
        cand_list_raw = query_candidates.get(q_canon, [])
        if not cand_list_raw:
            n_no_pool += 1
            continue

        cand_canons, seen = [], set()
        for c in cand_list_raw:
            c_canon = canonicalize(c)
            if c_canon not in seen:
                seen.add(c_canon)
                cand_canons.append(c_canon)
        n_cand = len(cand_canons)
        if q_canon not in seen:
            n_no_self_in_pool += 1

        simba_has = np.zeros(n_cand, dtype=bool)
        simba_embs = []
        for c_canon in cand_canons:
            emb = simba_cand_smi_adduct_to_emb.get((c_canon, adduct))
            if emb is not None:
                simba_embs.append(emb)
        simba_sims = np.full(n_cand, np.nan)
        if simba_embs:
            idxs = [
                j
                for j, c_canon in enumerate(cand_canons)
                if (c_canon, adduct) in simba_cand_smi_adduct_to_emb
            ]
            simba_has[idxs] = True
            simba_sims[idxs] = np.stack(simba_embs) @ simba_test_emb_norm[spec_i]

        cosine_has = np.zeros(n_cand, dtype=bool)
        cosine_rows = [
            cosine_cand_row_lookup.get((c_canon, adduct)) for c_canon in cand_canons
        ]
        cosine_valid_idxs = [j for j, r in enumerate(cosine_rows) if r is not None]
        cosine_sims = np.full(n_cand, np.nan)
        if cosine_valid_idxs:
            cosine_has[cosine_valid_idxs] = True
            rows = [cosine_rows[j] for j in cosine_valid_idxs]
            sims_valid = np.asarray(
                cosine_cand_mat[rows].multiply(cosine_test_mat[spec_i]).sum(axis=1)
            ).ravel()
            cosine_sims[cosine_valid_idxs] = sims_valid

        mismatch = simba_has != cosine_has
        n_embedding_availability_mismatch += int(mismatch.sum())
        n_neither_has_embedding += int((~simba_has & ~cosine_has).sum())

        simba_ranks = rank_descending(simba_sims)
        cosine_ranks = rank_descending(cosine_sims)
        simba_mces = np.clip(mces_max_value * (1.0 - simba_sims), 0.0, None)
        cosine_mces = np.clip(mces_max_value * (1.0 - cosine_sims), 0.0, None)

        for j, c_canon in enumerate(cand_canons):
            is_correct = c_canon == q_canon
            gt = 0.0 if is_correct else gt_lookup.get((q_canon, c_canon), np.nan)
            cand_adduct = adduct if (simba_has[j] or cosine_has[j]) else None

            cols["test_spec_idx"].append(spec_i)
            cols["test_smiles"].append(q_canon)
            cols["test_adduct"].append(adduct)
            cols["candidate_smiles"].append(c_canon)
            cols["candidate_adduct"].append(cand_adduct)
            cols["simba_rank"].append(simba_ranks[j])
            cols["cosine_rank"].append(cosine_ranks[j])
            cols["simba_similarity"].append(simba_sims[j])
            cols["simba_mces"].append(simba_mces[j])
            cols["cosine_similarity"].append(cosine_sims[j])
            cols["cosine_mces"].append(cosine_mces[j])
            cols["gt_mces"].append(gt)
            cols["is_correct"].append(int(is_correct))
            cols["n_peaks_test"].append(test_npeaks[spec_i])
            cols["n_peaks_candidate"].append(
                cand_npeaks_lookup.get((c_canon, adduct), np.nan)
            )

    print(f"\n  {n_no_pool} test spectra had no formula-matched candidate pool at all")
    print(
        f"  {n_no_self_in_pool} test spectra whose true molecule wasn't in its own "
        "candidate pool"
    )
    print(
        f"  {n_embedding_availability_mismatch} (spectrum, candidate) pairs where "
        "SIMBA and cosine disagreed on embedding availability (expected: 0, since "
        "both are built from the same source rows)"
    )
    print(
        f"  {n_neither_has_embedding} (spectrum, candidate) pairs with NO embedding "
        "at all from either method (the real, measured failure count — see module "
        "docstring)"
    )

    df = pd.DataFrame(cols)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows to {out_path}")
    print(
        f"  is_correct sum: {df['is_correct'].sum():,} (should equal test spectra with a pool)"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--simba_intermediates_dir", required=True)
    p.add_argument("--cosine_intermediates_dir", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--gt_mces_dir", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidate_tsv", required=True)
    p.add_argument("--iceberg_preds", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--mces_max_value", type=float, default=40.0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
