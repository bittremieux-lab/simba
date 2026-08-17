"""Extends retrieval_comparison_table.csv (item 8b) with two columns needed
for precursor-mass-discrepancy analysis, so future work never has to re-pay
this lookup cost per-query the way plot_retrieval_comparison_checks.py's
run_precursor_boxplot used to (a full MGF scan + a per-row RDKit-
canonicalized candidate_tsv lookup, done freshly every single run):

  test_precursor_mz      the test spectrum's own measured PRECURSOR_MZ (one
                          single-pass MGF scan for every test-fold spectrum,
                          already cheap -- this was never the bottleneck)
  candidate_precursor_mz the calculated precursor m/z for
                          (candidate_smiles, test_adduct) -- note: test_adduct,
                          not candidate_adduct, to match run_precursor_boxplot's
                          original semantics exactly (does this candidate exist
                          in candidate_tsv under the QUERY's own measured
                          adduct, independent of whether an ICEBERG/SIMBA
                          embedding happened to resolve for it)

candidate_tsv already HAS the calculated precursor for every candidate --
no new mass computation happens here at all. The only real cost is a
ONE-TIME canonicalization of its 600,455 raw (non-canonical) SMILES so it
can be joined against the table's already-canonical candidate_smiles column
(the same 15-25 min cost every other script touching this candidate list
pays fresh -- see plot_confusion_matrix_examples.py's module docstring).
That canonical (smiles, adduct) -> precursor lookup is cached to
--precursor_cache so no future script has to pay it again either.

After this, extend candidate_smiles for ALL 2,909,708 rows (not just the
17,555 true-candidate ones run_precursor_boxplot used) -- a vectorized
merge costs nothing extra once the canonicalization is done, and it makes
the table useful for precursor-based checks beyond just the one boxplot.

Usage:
    uv run python tools/add_precursor_columns.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --precursor_cache /path/to/candidates_test_official_canonical_precursor.pkl
"""

import argparse
from pathlib import Path

import pandas as pd
from simba_retrieval import canonicalize
from tqdm.auto import tqdm


def get_all_test_precursor_mz(mgf_path: str) -> dict[int, float]:
    """Single-pass MGF scan, same technique as
    plot_confusion_matrix_examples.py's extract_test_spectra_by_index --
    every fold=='test' spectrum's PRECURSOR_MZ, keyed by its 0-indexed
    position within the test-fold sequence (same enumeration order
    load_spectra(mgf, 'test') uses). No target_indices filter: this is used
    for the WHOLE test fold, not a small subsample."""
    result: dict[int, float] = {}
    test_idx = -1
    current: dict | None = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current = {}
                continue
            if line == "END IONS":
                if current is not None and current.get("FOLD") == "test":
                    test_idx += 1
                    if "PRECURSOR_MZ" in current:
                        result[test_idx] = float(current["PRECURSOR_MZ"])
                current = None
                continue
            if current is None:
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                current[k] = v
    return result


def build_canonical_precursor_lookup(candidate_tsv: str, cache_path: str) -> pd.Series:
    """(canonical smiles, ionization) -> precursor, built ONCE and cached.
    First-seen-wins on a duplicate (canon_smiles, adduct) pair (same
    convention used throughout this project -- never averaged)."""
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached canonical precursor lookup from {cache} ...")
        return pd.read_pickle(cache)

    print(f"Loading {candidate_tsv} ...")
    cand_index = pd.read_csv(candidate_tsv, sep="\t")
    print(f"  {len(cand_index):,} rows -- canonicalizing SMILES (one-time cost) ...")
    cand_index["canon_smiles"] = [
        canonicalize(s) for s in tqdm(cand_index["smiles"], desc="canonicalize")
    ]
    lookup_df = cand_index.drop_duplicates(
        subset=["canon_smiles", "ionization"], keep="first"
    )
    lookup = lookup_df.set_index(["canon_smiles", "ionization"])["precursor"]
    print(f"  {len(lookup):,} unique (canon_smiles, adduct) -> precursor entries")

    cache.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_pickle(cache)
    print(f"Cached to {cache} -- future runs (any script) can reuse this directly")
    return lookup


def run(
    table_csv: str,
    mgf: str,
    candidate_tsv: str,
    precursor_cache: str,
) -> None:
    print(f"Loading {table_csv} ...")
    df = pd.read_csv(table_csv)
    n_before = len(df)
    print(f"  {n_before:,} rows")

    print(f"\nScanning {mgf} for measured precursor m/z (all test-fold spectra) ...")
    test_precursor_by_idx = get_all_test_precursor_mz(mgf)
    print(f"  {len(test_precursor_by_idx):,} test-fold spectra found")
    df["test_precursor_mz"] = df["test_spec_idx"].map(test_precursor_by_idx)
    n_missing_test = df["test_precursor_mz"].isna().sum()
    print(f"  {n_missing_test:,} / {len(df):,} rows missing test_precursor_mz")

    print()
    precursor_lookup = build_canonical_precursor_lookup(candidate_tsv, precursor_cache)

    print("\nJoining candidate_precursor_mz (key: candidate_smiles, test_adduct) ...")
    join_key = pd.MultiIndex.from_arrays([df["candidate_smiles"], df["test_adduct"]])
    df["candidate_precursor_mz"] = precursor_lookup.reindex(join_key).to_numpy()
    n_missing_cand = df["candidate_precursor_mz"].isna().sum()
    print(f"  {n_missing_cand:,} / {len(df):,} rows missing candidate_precursor_mz")

    assert len(df) == n_before, (
        f"row count changed during join: {n_before:,} -> {len(df):,} -- a join "
        "must have fanned out, refusing to write a corrupted table"
    )

    print(f"\nWriting extended table back to {table_csv} ...")
    df.to_csv(table_csv, index=False)
    print(f"Done: {len(df):,} rows, {len(df.columns)} columns")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidate_tsv", required=True)
    p.add_argument("--precursor_cache", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
