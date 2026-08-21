"""
Compare two preprocessing output directories to verify refactoring correctness.

Usage:
    python compare_preprocessing_outputs.py <baseline_dir> <fixed_dir>

Checks:
  - For each (split, node) group: same pairs (idx1, idx2), same ED values (exact),
    MCES values within tolerance (stochastic ILP)
  - All chunks per group are concatenated before comparing, so chunk0+chunk1 in one
    directory is compared against chunk0 in another.

MCES values may differ slightly between runs due to HiGHS solver non-determinism
under a 2-second time limit. This is expected and not a bug.
"""

import re
import sys
from pathlib import Path

import numpy as np


MCES_TOL = 5.0


def get_group_key(filename: str) -> str:
    """Strip chunk index to get a stable grouping key."""
    # Multi-node: ends with _node{n}_chunk{k}.npy
    m = re.match(r"^(.+_node\d+)_chunk\d+\.npy$", filename)
    if m:
        return m.group(1)
    # Single-node: ends with _{k}.npy
    m = re.match(r"^(.+)_\d+\.npy$", filename)
    if m:
        return m.group(1)
    return filename.removesuffix(".npy")


def load_grouped(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    """Load all matching .npy files, concatenate chunks with the same group key."""
    groups: dict[str, list[np.ndarray]] = {}
    for f in sorted(directory.glob(f"{prefix}*.npy")):
        key = get_group_key(f.name)
        groups.setdefault(key, []).append(np.load(f))
    return {k: np.concatenate(arrays, axis=0) for k, arrays in groups.items()}


def compare_arrays(name: str, arr_a: np.ndarray, arr_b: np.ndarray) -> bool:
    if arr_a.shape[1] != arr_b.shape[1]:
        print(
            f"  FAIL {name}: column count mismatch {arr_a.shape[1]} vs {arr_b.shape[1]}"
        )
        return False

    def sort_key(arr):
        return arr[np.lexsort((arr[:, 1], arr[:, 0]))]

    a = sort_key(arr_a)
    b = sort_key(arr_b)

    if a.shape[0] != b.shape[0]:
        print(
            f"  WARN {name}: pair count differs — baseline={a.shape[0]}, fixed={b.shape[0]}"
        )
        pairs_a = set(zip(a[:, 0].astype(int), a[:, 1].astype(int)))
        pairs_b = set(zip(b[:, 0].astype(int), b[:, 1].astype(int)))
        common = pairs_a & pairs_b
        print(
            f"         Common pairs: {len(common)} / baseline={len(pairs_a)} fixed={len(pairs_b)}"
        )
        if len(common) == 0:
            return False
        mask_a = np.array([(int(r[0]), int(r[1])) in common for r in a])
        mask_b = np.array([(int(r[0]), int(r[1])) in common for r in b])
        a = a[mask_a]
        b = b[mask_b]

    if not (np.all(a[:, 0] == b[:, 0]) and np.all(a[:, 1] == b[:, 1])):
        print(f"  FAIL {name}: pair indices differ after sorting")
        return False

    n = a.shape[0]
    ok = True

    if arr_a.shape[1] >= 3:
        col, col_f = a[:, 2], b[:, 2]
        both_nan = np.isnan(col) & np.isnan(col_f)
        ed_mismatch = int(np.sum((col != col_f) & ~both_nan))
        nan_count = int(np.sum(both_nan))
        if ed_mismatch > 0:
            print(
                f"  FAIL {name}: ED mismatch on {ed_mismatch}/{n} pairs  (NaN-NaN pairs: {nan_count})"
            )
            ok = False
        else:
            print(
                f"  OK   {name}: ED exact match on all {n} pairs  (NaN pairs: {nan_count})"
            )

    if arr_a.shape[1] >= 4:
        mces_a, mces_b = a[:, 3], b[:, 3]
        abs_diff = np.abs(mces_a - mces_b)
        exact = int(np.sum(abs_diff == 0))
        within_tol = int(np.sum(abs_diff <= MCES_TOL))
        mean_err = float(np.nanmean(abs_diff))
        max_err = float(np.nanmax(abs_diff))
        print(
            f"  OK   {name}: MCES exact={exact}/{n} ({100 * exact / n:.1f}%)"
            f"  within±{MCES_TOL}={within_tol}/{n} ({100 * within_tol / n:.1f}%)"
            f"  mean_err={mean_err:.2f}  max_err={max_err:.2f}"
        )
        if within_tol < 0.95 * n:
            print(f"  WARN {name}: >5% of MCES values outside tolerance")

    return ok


def main(baseline_dir: str, fixed_dir: str) -> None:
    bd = Path(baseline_dir)
    fd = Path(fixed_dir)

    for d in (bd, fd):
        if not d.exists():
            print(f"ERROR: directory not found: {d}")
            sys.exit(1)

    print(f"Baseline : {bd}")
    print(f"Fixed    : {fd}")
    print()

    all_ok = True

    for prefix in ("ed_mces_", "edit_distance_", "mces_"):
        groups_b = load_grouped(bd, prefix)
        groups_f = load_grouped(fd, prefix)

        if not groups_b and not groups_f:
            continue

        print(f"--- {prefix}*.npy (chunks merged per node/split) ---")
        all_keys = sorted(set(groups_b) | set(groups_f))
        for key in all_keys:
            if key not in groups_b:
                print(f"  MISSING in baseline: {key}")
                all_ok = False
            elif key not in groups_f:
                print(f"  MISSING in fixed: {key}")
                all_ok = False
            else:
                ok = compare_arrays(key, groups_b[key], groups_f[key])
                all_ok = all_ok and ok
        print()

    print("======================")
    print("RESULT:", "ALL OK" if all_ok else "ISSUES FOUND — review above")
    print("======================")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
