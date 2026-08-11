"""8a: does raw cosine similarity's "max" (over the pool) cluster close to
1, the same way GT MCES's "min" (in test_to_candidate_gt.png /
test_to_test_gt.png) clusters close to 0? If even the ground truth says the
top-1/top-2 candidates are often nearly-indistinguishable structures, is
that ALSO true for a plain, no-SIMBA cosine-similarity baseline?

RAW cosine similarity only, no MCES-unit conversion anywhere — that
conversion (predicted_mces = mces_max_value * (1 - sim)) is only meaningful
for SIMBA's trained cosine_no_head head, not for an untrained similarity
score (see ood_generalization_check.py's module docstring for why). No
ranking, no ties, no argmax: this is a pure distribution-of-raw-similarity-
values question, nothing else.

Every spectrum in the pool is included in every statistic — no exclusions —
with exactly one necessary exception: test-to-test's "self spectrum" (a
spectrum compared with itself) is always excluded, since that comparison is
trivially similarity=1.0 by construction (not a real pool member, just the
query re-encountering itself) and would otherwise make "max" a meaningless
spike at 1.0 for every single query. Same-molecule-*different*-spectrum
pairs, and a test-to-candidate query's own true candidate, ARE genuine pool
members and are included in every statistic, including max.

Three plots, using cosine_baseline_intermediates.py's saved binned-spectrum
sparse matrices:
  - test_to_candidate_cosine_similarity.png: for each test spectrum, cosine
    similarity (min/mean/max) to every candidate in its formula-matched pool
    (own adduct; self included, nothing excluded).
  - test_to_candidate_cosine_similarity_top1_top2.png: same pool, but the
    top-1 (max) and top-2 (second-highest) similarity per query, overlaid —
    directly answers "how much of a gap is there between the best and
    second-best candidate."
  - test_to_test_cosine_similarity.png: for each test spectrum, cosine
    similarity (min/mean/max) to every OTHER individual test spectrum
    (literal self spectrum excluded; same-molecule-different-spectrum pairs
    included in every statistic). One dense matrix of raw per-spectrum
    binned vectors on both sides — no averaging anywhere.

Usage:
    uv run python tools/cosine_similarity_pool_distribution_plots.py \\
        --cosine_intermediates_dir /path/to/cosine_baseline_intermediates \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --test_to_test_prepro_dir /path/to/preprocessing_msg_exact_mces_1020 \\
        --output_dir /path/to/output
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
import scipy.sparse as sp
from simba_retrieval import canonicalize
from tqdm.auto import tqdm


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


def per_query_stats_ragged(
    spec_idx: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"q": spec_idx, "v": values})
    agg = df.groupby("q")["v"].agg(["min", "mean", "max"])
    return agg["min"].to_numpy(), agg["mean"].to_numpy(), agg["max"].to_numpy()


def per_query_top1_top2(
    spec_idx: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query top-1 (max) and top-2 (second-highest) value. top2 is NaN
    for a query whose pool has fewer than 2 members."""
    df = pd.DataFrame({"q": spec_idx, "v": values})
    df = df.sort_values(["q", "v"], ascending=[True, False])
    rank = df.groupby("q").cumcount()
    top1 = df.loc[rank == 0].set_index("q")["v"]
    top2 = df.loc[rank == 1].set_index("q")["v"].reindex(top1.index)
    return top1.to_numpy(), top2.to_numpy()


def row_stats(
    matrix: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = np.where(mask, matrix, np.nan)
    with np.errstate(all="ignore"):
        return (
            np.nanmin(masked, axis=1),
            np.nanmean(masked, axis=1),
            np.nanmax(masked, axis=1),
        )


def plot_min_mean_max(
    mins: np.ndarray,
    means: np.ndarray,
    maxs: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 80,
) -> None:
    xmax = max(np.nanmax(mins), np.nanmax(means), np.nanmax(maxs))
    xmin = min(np.nanmin(mins), np.nanmin(means), np.nanmin(maxs))
    bin_edges = np.linspace(xmin, xmax, bins + 1)

    plt.figure(figsize=(10, 5.5))
    for name, vals in [("min", mins), ("means", means), ("max", maxs)]:
        plt.hist(vals, bins=bin_edges, alpha=0.5, label=name)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title, fontsize=11, wrap=True)
    plt.legend(title="Dist. stat.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(
        f"Saved {out_path}  (n={len(mins)}, min-of-min={np.nanmin(mins):.3f}, max-of-max={np.nanmax(maxs):.3f})"
    )


def plot_top1_top2(
    top1: np.ndarray, top2: np.ndarray, title: str, out_path: Path, bins: int = 80
) -> None:
    top2_valid = top2[~np.isnan(top2)]
    xmax = max(np.nanmax(top1), np.nanmax(top2_valid))
    xmin = min(np.nanmin(top1), np.nanmin(top2_valid))
    bin_edges = np.linspace(xmin, xmax, bins + 1)

    plt.figure(figsize=(10, 5.5))
    plt.hist(top1, bins=bin_edges, alpha=0.5, label="top1 (max)")
    plt.hist(top2_valid, bins=bin_edges, alpha=0.5, label="top2 (2nd highest)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title, fontsize=11, wrap=True)
    plt.legend(title="Dist. stat.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(
        f"Saved {out_path}  (n={len(top1)}, {len(top2_valid)} with a top2; "
        f"top1 mean={np.nanmean(top1):.3f}, top2 mean={np.nanmean(top2_valid):.3f})"
    )


def expand_pool_similarity(
    query_candidates: dict[str, list[str]],
    test_smiles_raw: list[str],
    test_adducts_raw: list[str],
    test_mat: sp.csr_matrix,
    cand_row_lookup: dict[tuple[str, str], int],
    cand_mat: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """For each test spectrum, raw cosine similarity to every candidate in
    its formula-matched pool that has an embedding for this spectrum's own
    adduct — nothing excluded, self included like any other pool member. No
    MCES conversion, no ranking. Returns the flat (spec_idx, similarity)
    pairs, for per-query aggregation."""
    spec_parts, sim_parts = [], []
    for spec_i, (smi, adduct) in enumerate(
        tqdm(
            list(zip(test_smiles_raw, test_adducts_raw)),
            desc="Scoring test-to-candidate pools",
        )
    ):
        q_canon = canonicalize(smi)
        cand_list = query_candidates.get(q_canon, [])
        rows = [
            r
            for c in cand_list
            if (r := cand_row_lookup.get((canonicalize(c), adduct))) is not None
        ]
        if not rows:
            continue
        sims = np.asarray(cand_mat[rows].multiply(test_mat[spec_i]).sum(axis=1)).ravel()
        spec_parts.append(np.full(len(sims), spec_i))
        sim_parts.append(sims)

    if not spec_parts:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    return np.concatenate(spec_parts), np.concatenate(sim_parts)


def run(
    cosine_intermediates_dir: str,
    candidates: str,
    test_to_test_prepro_dir: str,
    output_dir: str,
) -> None:
    inter = Path(cosine_intermediates_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved binned-spectrum matrices + SMILES + adducts ...")
    test_mat = sp.load_npz(inter / "test_mat.npz")
    test_smiles = json.loads((inter / "test_smiles.json").read_text())
    test_adducts = json.loads((inter / "test_adducts.json").read_text())
    cand_mat = sp.load_npz(inter / "candidate_mat.npz")
    candidate_smiles = json.loads((inter / "candidate_smiles.json").read_text())
    candidate_adducts = json.loads((inter / "candidate_adducts.json").read_text())
    print(f"  {test_mat.shape[0]} test spectra, {cand_mat.shape[0]} candidate spectra")

    cand_row_lookup = build_candidate_row_lookup_by_smi_adduct(
        candidate_smiles, candidate_adducts
    )

    print(f"Loading candidate pools from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)
    query_candidates = {canonicalize(k): v for k, v in candidate_json.items()}

    print("\n--- test-to-candidate (cosine similarity to pool; nothing excluded) ---")
    spec_idx, sims = expand_pool_similarity(
        query_candidates, test_smiles, test_adducts, test_mat, cand_row_lookup, cand_mat
    )
    print(f"  {len(spec_idx):,} scored (spectrum, candidate) pairs")

    tc_min, tc_mean, tc_max = per_query_stats_ragged(spec_idx, sims)
    plot_min_mean_max(
        tc_min,
        tc_mean,
        tc_max,
        "Test-to-candidate: cosine similarity to pool (nothing excluded)",
        out_dir / "test_to_candidate_cosine_similarity.png",
    )

    tc_top1, tc_top2 = per_query_top1_top2(spec_idx, sims)
    plot_top1_top2(
        tc_top1,
        tc_top2,
        "Test-to-candidate: top-1 vs top-2 cosine similarity to pool (nothing excluded)",
        out_dir / "test_to_candidate_cosine_similarity_top1_top2.png",
    )

    print(
        "\n--- test-to-test (cosine similarity; literal self spectrum excluded, nothing else) ---"
    )
    with open(Path(test_to_test_prepro_dir) / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    tt_idx_to_smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()

    smi_to_molidx = {smi: i for i, smi in enumerate(tt_idx_to_smiles)}
    spec_molidx_all = np.array(
        [smi_to_molidx.get(canonicalize(s), -1) for s in test_smiles]
    )
    valid_spec = spec_molidx_all >= 0
    print(f"  {int(valid_spec.sum()):,} test spectra in the mined test-fold set")

    valid_mat = test_mat[np.where(valid_spec)[0]]
    sims_matrix = np.asarray((valid_mat @ valid_mat.T).todense())

    rows = np.arange(sims_matrix.shape[0])
    sims_matrix[rows, rows] = (
        np.nan
    )  # literal self spectrum excluded (see module docstring)
    valid_mask = ~np.isnan(sims_matrix)

    tt_min, tt_mean, tt_max = row_stats(sims_matrix, valid_mask)
    plot_min_mean_max(
        tt_min,
        tt_mean,
        tt_max,
        "Test-to-test: cosine similarity (literal self spectrum excluded, nothing else)",
        out_dir / "test_to_test_cosine_similarity.png",
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cosine_intermediates_dir",
        required=True,
        help="Dir with test_mat.npz/candidate_mat.npz/test_smiles.json/"
        "test_adducts.json/candidate_smiles.json/candidate_adducts.json from "
        "cosine_baseline_intermediates.py",
    )
    p.add_argument(
        "--candidates",
        required=True,
        help="Candidate JSON {smiles: [candidate smiles, ...]} "
        "(MassSpecGym_retrieval_candidates_formula.json)",
    )
    p.add_argument(
        "--test_to_test_prepro_dir",
        required=True,
        help="Official-split preprocessing dir with the exact-refined test-fold pairs "
        "(data/massspecgym/preprocessing_msg_exact_mces_1020)",
    )
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
