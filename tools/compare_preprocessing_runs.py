"""
Compare two preprocessing runs for consistency.

Checks that distances computed for molecules present in both runs are
identical, and reports molecule/pair overlap statistics per split.

Usage:
    python tools/compare_preprocessing_runs.py <run1_dir> <run2_dir>

For each split (train / val / test) it reports:
  - number of unique molecules in each run
  - number of pairs (ED + MCES) in each run
  - intersection of molecules between the two runs
  - intersection of pairs between the two runs
  - assertion: every pair whose both molecules appear in both runs must
    be present in both runs with identical ED and MCES values
"""

import sys
from pathlib import Path

import dill
import numpy as np


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def load_pickle(preprocessing_dir: str) -> dict:
    p = Path(preprocessing_dir)
    candidates = list(p.glob("mapping_unique_smiles.pkl"))
    if not candidates:
        candidates = list(p.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No pickle found in {preprocessing_dir}")
    path = candidates[0]
    print(f"  Loading {path}")
    with open(path, "rb") as f:
        return dill.load(f)


def get_molecules(data: dict, split: str) -> set[str]:
    """Return set of canonical SMILES for a split."""
    for key in [f"molecule_pairs_{split}", f"df_smiles_{split}"]:
        if key not in data or data[key] is None:
            continue
        obj = data[key]
        if hasattr(obj, "df_smiles"):  # MoleculePairsOpt
            df = obj.df_smiles
        else:  # plain DataFrame
            df = obj
        if df is None or df.empty:
            continue
        col = "canon_smiles" if "canon_smiles" in df.columns else df.columns[0]
        return set(df[col].tolist())
    return set()


def get_pairs(
    data: dict, split: str, preprocessing_dir: str
) -> dict[frozenset, tuple[float, float]]:
    """
    Return dict {frozenset({smiles_a, smiles_b}): (ed, mces)} for a split.
    Loads from ed_mces_indexes_tani_incremental_{split}_*.npy (4 cols: i, j, ed, mces).
    SMILES index mapping is taken from the pickle.
    """
    # Build index → smiles mapping from pickle
    idx_to_smiles = {}
    for key in [f"molecule_pairs_{split}", f"df_smiles_{split}"]:
        if key not in data or data[key] is None:
            continue
        obj = data[key]
        df = obj.df_smiles if hasattr(obj, "df_smiles") else obj
        if df is None or df.empty:
            continue
        col = "canon_smiles" if "canon_smiles" in df.columns else df.columns[0]
        idx_to_smiles = df[col].to_dict()
        break

    if not idx_to_smiles:
        return {}

    # Load all matching ed_mces npy files for this split
    p = Path(preprocessing_dir)
    npy_files = sorted(p.glob(f"ed_mces_indexes_tani_incremental_{split}_*.npy"))
    if not npy_files:
        return {}

    all_rows = np.concatenate([np.load(f) for f in npy_files], axis=0)  # shape (N, 4)

    pairs = {}
    for row in all_rows:
        i, j = int(row[0]), int(row[1])
        s0 = idx_to_smiles.get(i)
        s1 = idx_to_smiles.get(j)
        if s0 is None or s1 is None:
            continue
        pairs[frozenset([s0, s1])] = (float(row[2]), float(row[3]))
    return pairs


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #


def main(dir1: str, dir2: str):
    section("Loading pickles")
    d1 = load_pickle(dir1)
    d2 = load_pickle(dir2)

    any_assertion_failed = False

    for split in ["train", "val", "test"]:
        section(f"Split: {split.upper()}")

        mols1 = get_molecules(d1, split)
        mols2 = get_molecules(d2, split)
        intersection_mols = mols1 & mols2

        pairs1 = get_pairs(d1, split, dir1)
        pairs2 = get_pairs(d2, split, dir2)

        print(f"  Run 1 – unique molecules : {len(mols1):>6}")
        print(f"  Run 2 – unique molecules : {len(mols2):>6}")
        print(f"  Intersection (molecules) : {len(intersection_mols):>6}")
        print(f"  Only in run 1            : {len(mols1 - mols2):>6}")
        print(f"  Only in run 2            : {len(mols2 - mols1):>6}")

        print(f"\n  Run 1 – pairs            : {len(pairs1):>6}")
        print(f"  Run 2 – pairs            : {len(pairs2):>6}")

        # Pairs whose both molecules are in the intersection
        cross_pairs1 = {k for k in pairs1 if k <= intersection_mols}
        cross_pairs2 = {k for k in pairs2 if k <= intersection_mols}
        common_pairs = cross_pairs1 & cross_pairs2
        only_in_r1 = cross_pairs1 - cross_pairs2
        only_in_r2 = cross_pairs2 - cross_pairs1

        print("\n  Pairs between shared molecules:")
        print(f"    Both runs              : {len(common_pairs):>6}")
        print(f"    Only in run 1          : {len(only_in_r1):>6}")
        print(f"    Only in run 2          : {len(only_in_r2):>6}")

        # Assertion: a pair of shared molecules must be in both or neither
        if only_in_r1 or only_in_r2:
            any_assertion_failed = True
            print("\n  ❌ ASSERTION FAILED: inconsistent pairs for shared molecules!")
            if only_in_r1:
                example = next(iter(only_in_r1))
                ed, mc = pairs1[example]
                print(
                    f"     Example only in run1: {sorted(example)}  ED={ed:.1f} MCES={mc:.1f}"
                )
            if only_in_r2:
                example = next(iter(only_in_r2))
                ed, mc = pairs2[example]
                print(
                    f"     Example only in run2: {sorted(example)}  ED={ed:.1f} MCES={mc:.1f}"
                )
        else:
            print("\n  ✅ Assertion passed: all shared-molecule pairs are consistent.")

        # Show a few pairs present in both for sanity check
        if common_pairs:
            print("\n  Sample common pairs (first 3):")
            for p in list(common_pairs)[:3]:
                ed1, mc1 = pairs1[p]
                ed2, mc2 = pairs2[p]
                smiles = sorted(p)
                match = (
                    "✅" if abs(ed1 - ed2) < 1e-6 and abs(mc1 - mc2) < 1e-6 else "⚠️ "
                )
                print(
                    f"    {match}  {smiles[0][:40]}..."
                    f"  ED={ed1:.1f}/{ed2:.1f}  MCES={mc1:.1f}/{mc2:.1f}"
                )

    section("Summary")
    if any_assertion_failed:
        print("  ❌ At least one split failed the pair-consistency assertion.")
        sys.exit(1)
    else:
        print("  ✅ All splits passed pair-consistency assertions.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/compare_preprocessing_runs.py <run1_dir> <run2_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
