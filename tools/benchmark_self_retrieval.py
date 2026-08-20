"""Hit@1/5/20 retrieval benchmark among validation molecules with >=2
spectra, using the "self (MCES=0)" bucket's own molecule pool as GT-MCES-
nearest-neighbor decoys.

Fully buildable from data already saved by training (no fresh MCES
computation, no SIMBA re-inference): every pair among the ~2,731
self-bucket molecules already exists in the validation set (confirmed
directly -- 3,727,815 cross-molecule pairs among this pool == exactly
C(2731,2)), so both GT MCES and every experiment's own pred_mces for any
query-vs-decoy comparison are already logged.

For each molecule with >=2 validation spectra (2,010 of them -- excludes
the 721 singleton-spectrum molecules, which have no second spectrum to
retrieve):
  - query = that molecule's first spectrum (spec_idx_0 of its own
    "self (MCES=0)" pair row)
  - true match = the SAME molecule's other spectrum (spec_idx_1) -- scored
    directly from that same "self, different spectrum" row
  - decoys = the 255 OTHER pool molecules with the lowest GT MCES to the
    query molecule (decoy selection is GT-MCES-based only, hence identical
    across every method being compared -- only the ranking differs)
  - candidates = true match + 255 decoys = 256 total
  - rank by score: SIMBA (pred_mces) ascending -- lowest predicted MCES
    first; cosine descending -- highest similarity first
  - hit@k = 1 if the true match lands in the top k, else 0; averaged over
    all 2,010 queries

Caveat worth knowing: a query/decoy molecule's "first" vs "last" spectrum
(spec_idx_0 vs spec_idx_1, per CustomDatasetMultitasking's val-time index
selection) depends on which column (mol_idx_0 or mol_idx_1) it lands in for
a given saved pair row -- this script uses whatever score is already
logged for each query-decoy row as-is, rather than re-deriving a strictly
single-spectrum-consistent comparison (which would need fresh inference).

Usage:
    uv run python tools/benchmark_self_retrieval.py \\
        --exp_dirs /path/to/009_... /path/to/010_... /path/to/011_... /path/to/012_... \\
        --cosine_parquet /path/to/preprocessing_dir/val_cosine_val.parquet \\
        --val_name val \\
        --n_decoys 255 \\
        --ks 1 5 20
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_PARQUET_PRED_COL_RE = re.compile(r"^pred_mces_step(\d+)$")
SELF_BIN_LABEL = "self (MCES=0)"


def latest_pred_col(consolidated_path: Path) -> str:
    steps = [
        int(m.group(1))
        for name in pq.ParquetFile(consolidated_path).schema.names
        if (m := _PARQUET_PRED_COL_RE.match(name))
    ]
    if not steps:
        raise ValueError(f"No pred_mces_step* columns found in {consolidated_path}")
    return f"pred_mces_step{max(steps):06d}"


def build_pool_and_queries(ref_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(pool_mols, query_mols): the full self-bucket molecule pool, and the
    subset with >=2 spectra (i.e. spec_idx_0 != spec_idx_1 on their own
    self-pair row)."""
    self_df = ref_df[ref_df["mces_bin"] == SELF_BIN_LABEL]
    pool_mols = pd.unique(pd.concat([self_df["mol_idx_0"], self_df["mol_idx_1"]]))
    same_spec = self_df["spec_idx_0"] == self_df["spec_idx_1"]
    query_mols = self_df.loc[~same_spec, "mol_idx_0"].to_numpy()
    return pool_mols, query_mols


def build_score_matrix(
    ref_df: pd.DataFrame, pool_mols: np.ndarray, mol_to_local: dict, value_col: str
) -> np.ndarray:
    """Dense (n_pool, n_pool) matrix of `value_col` for every cross-molecule
    pair within the pool, symmetric, diagonal left as NaN (never selected --
    see build_pool_and_queries/rank_candidates)."""
    n = len(pool_mols)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    cross = ref_df[ref_df["mol_idx_0"] != ref_df["mol_idx_1"]]
    pool_set = set(pool_mols.tolist())
    in_pool = cross["mol_idx_0"].isin(pool_set) & cross["mol_idx_1"].isin(pool_set)
    within = cross[in_pool]
    i = within["mol_idx_0"].map(mol_to_local).to_numpy()
    j = within["mol_idx_1"].map(mol_to_local).to_numpy()
    v = within[value_col].to_numpy()
    mat[i, j] = v
    mat[j, i] = v
    return mat


def true_match_scores(
    ref_df: pd.DataFrame, query_mols: np.ndarray, value_col: str
) -> dict:
    """query_mol -> that method's score for the query's own "self, different
    spectrum" row (its true match: the same molecule's other spectrum)."""
    self_df = ref_df[ref_df["mces_bin"] == SELF_BIN_LABEL]
    diff_spec = self_df["spec_idx_0"] != self_df["spec_idx_1"]
    rows = self_df.loc[diff_spec]
    return dict(zip(rows["mol_idx_0"], rows[value_col]))


def hit_at_k(
    gt_matrix: np.ndarray,
    score_matrix: np.ndarray,
    true_scores: dict,
    pool_mols: np.ndarray,
    mol_to_local: dict,
    query_mols: np.ndarray,
    n_decoys: int,
    ks: list[int],
    higher_is_better: bool,
) -> dict[int, float]:
    hits = dict.fromkeys(ks, 0)
    n_scored = 0
    for q in query_mols:
        qi = mol_to_local[q]
        if q not in true_scores or np.isnan(true_scores[q]):
            continue
        gt_row = gt_matrix[qi].copy()
        gt_row[qi] = np.inf  # never pick the query molecule itself as a decoy
        decoy_local = np.argsort(gt_row)[:n_decoys]
        decoy_scores = score_matrix[qi, decoy_local]
        valid = ~np.isnan(decoy_scores)
        decoy_scores = decoy_scores[valid]
        candidates = np.append(decoy_scores, true_scores[q])
        true_idx = len(candidates) - 1
        order = np.argsort(-candidates if higher_is_better else candidates)
        rank = int(np.nonzero(order == true_idx)[0][0]) + 1
        n_scored += 1
        for k in ks:
            hits[k] += int(rank <= k)
    return {k: (hits[k] / n_scored if n_scored else float("nan")) for k in ks}, n_scored


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--val_name", default="val")
    parser.add_argument("--cosine_parquet", default=None)
    parser.add_argument("--n_decoys", type=int, default=255)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    args = parser.parse_args()

    exp_dirs = [Path(p) for p in args.exp_dirs]
    ref_path = exp_dirs[0] / f"val_pairs_{args.val_name}_consolidated.parquet"
    print(f"Reading pair identity/GT-MCES from {ref_path} ...")
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
    print(
        f"Pool size: {len(pool_mols):,} molecules; queries (>=2 spectra): {len(query_mols):,}"
    )
    mol_to_local = {m: i for i, m in enumerate(pool_mols)}

    gt_matrix = build_score_matrix(ref_df, pool_mols, mol_to_local, "gt_mces")

    results = {}
    for exp_dir in exp_dirs:
        path = exp_dir / f"val_pairs_{args.val_name}_consolidated.parquet"
        pred_col = latest_pred_col(path)
        print(f"\nScoring {exp_dir.name} (step column {pred_col}) ...")
        df = pd.read_parquet(
            path,
            columns=[
                "mol_idx_0",
                "mol_idx_1",
                "mces_bin",
                "spec_idx_0",
                "spec_idx_1",
                pred_col,
            ],
        )
        score_matrix = build_score_matrix(df, pool_mols, mol_to_local, pred_col)
        true_scores = true_match_scores(df, query_mols, pred_col)
        hits, n_scored = hit_at_k(
            gt_matrix,
            score_matrix,
            true_scores,
            pool_mols,
            mol_to_local,
            query_mols,
            args.n_decoys,
            args.ks,
            higher_is_better=False,
        )
        results[exp_dir.name] = hits
        print(f"  scored {n_scored:,}/{len(query_mols):,} queries")
        for k, v in hits.items():
            print(f"  Hit@{k}: {v:.4f}")

    if args.cosine_parquet:
        print(f"\nScoring cosine baseline ({args.cosine_parquet}) ...")
        cos = pd.read_parquet(args.cosine_parquet)
        df = ref_df[
            ["mol_idx_0", "mol_idx_1", "mces_bin", "spec_idx_0", "spec_idx_1"]
        ].merge(
            cos, on=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"], how="left"
        )
        score_matrix = build_score_matrix(df, pool_mols, mol_to_local, "cosine")
        true_scores = true_match_scores(df, query_mols, "cosine")
        hits, n_scored = hit_at_k(
            gt_matrix,
            score_matrix,
            true_scores,
            pool_mols,
            mol_to_local,
            query_mols,
            args.n_decoys,
            args.ks,
            higher_is_better=True,
        )
        results["cosine (raw spectral baseline)"] = hits
        print(f"  scored {n_scored:,}/{len(query_mols):,} queries")
        for k, v in hits.items():
            print(f"  Hit@{k}: {v:.4f}")

    print("\n=== Summary ===")
    summary = pd.DataFrame(results).T
    summary.columns = [f"Hit@{k}" for k in args.ks]
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
