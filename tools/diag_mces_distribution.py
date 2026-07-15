"""
Diagnostic: print raw MCES distribution across all splits without any normalization or training.
Reads .npy pair files directly, column 3 = raw MCES (0-40).
"""

import glob
import sys

import numpy as np


MCES_COL = 3  # same as COLUMN_MCES20 in load_mces.py
BINS = [0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40]
SPLITS = [
    ("train", "ed_mces_indexes_tani_incremental_train"),
    ("val_scaffold", "ed_mces_indexes_tani_incremental_val"),
    ("val_official", "ed_mces_indexes_tani_incremental_val_official"),
]


def load_raw_mces(prepro_dir, prefix):
    pattern = f"{prepro_dir}/{prefix}*.npy"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    chunks = [np.load(f)[:, MCES_COL] for f in files]
    print(f"  loaded {len(files)} file(s): {[f.split('/')[-1] for f in files]}")
    return np.concatenate(chunks)


def print_distribution(name, mces):
    counts, edges = np.histogram(mces, bins=BINS)
    total = counts.sum()
    print(f"\n=== {name}  (total pairs: {total:,}) ===")
    print(f"  {'bin':>14}  {'count':>10}  {'%':>7}")
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        pct = 100 * counts[i] / total if total > 0 else 0
        print(f"  [{lo:5.1f}, {hi:5.1f})  {counts[i]:>10,}  {pct:>6.2f}%")
    print(
        f"  min={mces.min():.3f}  max={mces.max():.3f}  mean={mces.mean():.3f}  median={np.median(mces):.3f}"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/diag_mces_distribution.py <prepro_dir>")
        sys.exit(1)
    prepro_dir = sys.argv[1]
    print(f"Prepro dir: {prepro_dir}\n")
    for name, prefix in SPLITS:
        print(f"Loading {name} ...")
        mces = load_raw_mces(prepro_dir, prefix)
        if mces is None:
            print(f"  [skipped — no files found for prefix '{prefix}']")
            continue
        print_distribution(name, mces)
    print("\nDone.")


if __name__ == "__main__":
    main()
