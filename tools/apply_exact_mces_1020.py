"""Replace [10, 20] lower-bound distances with exact MCES in the preprocessing pairs files.

For each split, the script:
  1. Loads the original ed_mces_indexes_tani_incremental_{split}_*.npy (base: max(Gaetan, HDF5)).
  2. Loads mces_exact_10_20.npy produced by compute_mces_exact_1020.py combine.
  3. Matches pairs by (mol_idx_a, mol_idx_b) via sorted-key binary search.
  4. Sanity-checks that ≥ 99 % of valid exact values are ≥ the original lower bound.
     (Exact MCES must never be below a true lower bound; 1 % slack for numerical noise.)
  5. Replaces the distance column with the exact value, EXCEPT when exact == -1.0
     (solver could not resolve within the watchdog deadline — keep original lb).
  6. Writes the updated pairs array to OUTPUT_DIR.

OUTPUT_DIR mirrors the original PREPRO_DIR: same file names, updated distance columns,
mapping.pkl symlinked from the original.

Usage:
    uv run python tools/apply_exact_mces_1020.py

    # Dry-run (print stats only, write nothing):
    uv run python tools/apply_exact_mces_1020.py --dry_run

Memory note: the train pairs file is ~7.6 GB (238 M × 4 × float64).
Run on asimov / asimov2 which have sufficient RAM; the login node may not.
"""

import argparse
from pathlib import Path

import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────
PREPRO_DIR = Path("/mnt/data/nkubrakov/massspecgym/preprocessing_msg_max_lb_hdf5")
EXACT_DIR = Path("/mnt/data2/nkubrakov/mces_exact_1020")
OUTPUT_DIR = Path("/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020")

SPLITS = {
    "train": {
        "pairs_npy": PREPRO_DIR
        / "ed_mces_indexes_tani_incremental_train_node0_chunk0.npy",
        "exact_npy": EXACT_DIR / "mces_exact_10_20.npy",
        "out_npy": OUTPUT_DIR
        / "ed_mces_indexes_tani_incremental_train_node0_chunk0.npy",
    },
    "val": {
        "pairs_npy": PREPRO_DIR
        / "ed_mces_indexes_tani_incremental_val_node0_chunk0.npy",
        "exact_npy": EXACT_DIR / "val" / "mces_exact_10_20.npy",
        "out_npy": OUTPUT_DIR / "ed_mces_indexes_tani_incremental_val_node0_chunk0.npy",
    },
    "val_official": {
        "pairs_npy": PREPRO_DIR
        / "ed_mces_indexes_tani_incremental_val_official_node0_chunk0.npy",
        "exact_npy": EXACT_DIR / "val_official" / "mces_exact_10_20.npy",
        "out_npy": OUTPUT_DIR
        / "ed_mces_indexes_tani_incremental_val_official_node0_chunk0.npy",
    },
    "test": {
        "pairs_npy": PREPRO_DIR
        / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy",
        "exact_npy": EXACT_DIR / "test" / "mces_exact_10_20.npy",
        "out_npy": OUTPUT_DIR
        / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy",
    },
}

LB_MIN, LB_MAX = 10.0, 20.0
SANITY_THRESHOLD = 0.99  # fraction of exact values that must be ≥ original lb


def pack_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack (mol_idx_a, mol_idx_b) int32 pairs into a single int64 sort key."""
    return a.astype(np.int64) * 1_000_000 + b.astype(np.int64)


def process_split(name: str, cfg: dict, dry_run: bool) -> None:
    print(f"\n{'=' * 60}")
    print(f"Split: {name}")
    print(f"{'=' * 60}")

    # ── Load exact MCES ────────────────────────────────────────────────────
    print(f"Loading exact MCES from {cfg['exact_npy'].name} ...")
    exact = np.load(cfg["exact_npy"])  # (M, 3) float32 [mol_a, mol_b, mces]
    exact_a = exact[:, 0].astype(np.int32)
    exact_b = exact[:, 1].astype(np.int32)
    exact_mces = exact[:, 2]  # float32, -1.0 = unresolved

    # Sort by packed key for binary search
    exact_keys = pack_key(exact_a, exact_b)
    sort_idx = np.argsort(exact_keys, kind="stable")
    exact_keys = exact_keys[sort_idx]
    exact_mces = exact_mces[sort_idx]
    print(f"  {len(exact_keys):,} exact pairs loaded and sorted")
    n_unresolved = (exact_mces == -1.0).sum()
    print(
        f"  Unresolved (−1): {n_unresolved:,}  ({100 * n_unresolved / len(exact_mces):.2f}%)"
    )

    # ── Load original pairs ────────────────────────────────────────────────
    print(f"Loading original pairs from {cfg['pairs_npy'].name} ...")
    orig = np.load(cfg["pairs_npy"])  # (N, 4) float64 [a, b, ?, lb]
    n_total = len(orig)
    print(f"  {n_total:,} total pairs")

    mask_1020 = (orig[:, 3] >= LB_MIN) & (orig[:, 3] <= LB_MAX)
    n_1020 = mask_1020.sum()
    print(f"  {n_1020:,} pairs in [{LB_MIN}, {LB_MAX}]")

    # ── Match pairs ────────────────────────────────────────────────────────
    idx_1020 = np.where(mask_1020)[0]
    a_1020 = orig[idx_1020, 0].astype(np.int32)
    b_1020 = orig[idx_1020, 1].astype(np.int32)
    query_keys = pack_key(a_1020, b_1020)

    pos = np.searchsorted(exact_keys, query_keys)
    # Clamp to valid range before indexing
    pos = np.minimum(pos, len(exact_keys) - 1)
    matched = exact_keys[pos] == query_keys

    n_matched = matched.sum()
    print(f"  Matched in exact: {n_matched:,} / {n_1020:,}")
    if n_matched < n_1020:
        print(
            f"  WARNING: {n_1020 - n_matched:,} pairs in [10,20] not found in exact data"
        )

    # ── Sanity check: exact ≥ lb for ≥ 99% of valid (non -1) pairs ────────
    exact_vals_matched = exact_mces[pos[matched]]
    old_vals_matched = orig[idx_1020[matched], 3].astype(np.float32)

    valid_mask = exact_vals_matched != -1.0
    n_valid = valid_mask.sum()
    n_higher = (exact_vals_matched[valid_mask] >= old_vals_matched[valid_mask]).sum()
    frac_higher = n_valid and n_higher / n_valid
    print("\nSanity check (exact ≥ lb):")
    print(f"  Valid (non-−1): {n_valid:,} / {n_matched:,}")
    print(f"  Exact ≥ lb:     {n_higher:,} / {n_valid:,}  ({100 * frac_higher:.3f}%)")

    if n_valid > 0 and frac_higher < SANITY_THRESHOLD:
        raise RuntimeError(
            f"SANITY FAIL: only {100 * frac_higher:.1f}% of exact values ≥ lb "
            f"(need ≥ {100 * SANITY_THRESHOLD:.0f}%). "
            "Pair matching is likely wrong — check mol_idx alignment."
        )
    print("  ✓ Sanity check passed")

    # ── Apply replacement ──────────────────────────────────────────────────
    # Replace only where matched AND exact != -1.0
    replace_mask_local = matched.copy()
    replace_mask_local[matched] &= exact_vals_matched != -1.0
    n_replaced = replace_mask_local.sum()
    n_kept_unresolved = matched.sum() - n_replaced

    print("\nReplacement:")
    print(f"  Replaced with exact:   {n_replaced:,}")
    print(f"  Kept (unresolved −1):  {n_kept_unresolved:,}")
    print(f"  Not in exact (kept):   {n_1020 - matched.sum():,}")

    if dry_run:
        print("\n  [dry-run] Nothing written.")
        return

    # ── Write output ───────────────────────────────────────────────────────
    updated = orig.copy()
    replace_idxs = idx_1020[replace_mask_local]
    new_vals = exact_mces[pos[replace_mask_local]].astype(np.float64)
    updated[replace_idxs, 3] = new_vals

    cfg["out_npy"].parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg["out_npy"], updated)
    size_gb = cfg["out_npy"].stat().st_size / 1e9
    print(f"\n  Wrote {cfg['out_npy']}  ({size_gb:.2f} GB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Which splits to process (default: all four).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print stats and sanity check only; write nothing.",
    )
    a = p.parse_args()

    if not a.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        mapping_link = OUTPUT_DIR / "mapping.pkl"
        if not mapping_link.exists():
            mapping_link.symlink_to(PREPRO_DIR / "mapping.pkl")
            print(f"Symlinked mapping.pkl → {PREPRO_DIR / 'mapping.pkl'}")

    for split in a.splits:
        process_split(split, SPLITS[split], a.dry_run)

    if not a.dry_run:
        print(f"\nDone. Updated preprocessing at:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
