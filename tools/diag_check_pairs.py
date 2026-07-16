"""
Cross-reference logged pairs against the actual pickle + raw npy files.
Usage: uv run python tools/diag_check_pairs.py <prepro_dir>
"""

import glob
import pickle
import sys

import numpy as np


PAIRS_TO_CHECK = {
    "train": [
        (4967, 11854, 40.00),
        (16667, 17669, 37.00),
        (270, 12590, 38.00),
        (1, 4236, 19.50),
        (7287, 12234, 18.50),
    ],
    "val_scaffold": [
        (1272, 1319, 40.00),
        (41, 286, 37.00),
        (1135, 1375, 26.00),
        (1012, 1221, 22.00),
        (547, 749, 28.00),
    ],
    "val_official": [
        (470, 1511, 40.00),
        (583, 850, 40.00),
        (1111, 1864, 23.50),
        (2128, 2170, 26.00),
        (1504, 2890, 27.50),
    ],
}

PREFIXES = {
    "train": "ed_mces_indexes_tani_incremental_train",
    "val_scaffold": "ed_mces_indexes_tani_incremental_val",
    "val_official": "ed_mces_indexes_tani_incremental_val_official",
}

MCES_COL = 3


def load_mapping(prepro_dir):
    path = f"{prepro_dir}/mapping.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw_pairs(prepro_dir, prefix):
    files = sorted(glob.glob(f"{prepro_dir}/{prefix}*.npy"))
    chunks = [np.load(f) for f in files]
    arr = np.concatenate(chunks, axis=0)
    print(
        f"  Loaded {arr.shape[0]:,} pairs from {len(files)} file(s) for prefix '{prefix}'"
    )
    return arr


def find_pair(arr, idx0, idx1):
    """Find row where col0==idx0 and col1==idx1 (or swapped)."""
    mask = ((arr[:, 0].astype(int) == idx0) & (arr[:, 1].astype(int) == idx1)) | (
        (arr[:, 0].astype(int) == idx1) & (arr[:, 1].astype(int) == idx0)
    )
    hits = arr[mask]
    return hits


def get_smiles(mol_pairs, idx):
    try:
        return mol_pairs.spectra[idx].smiles
    except Exception as e:
        return f"ERROR: {e}"


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python tools/diag_check_pairs.py <prepro_dir>")
        sys.exit(1)
    prepro_dir = sys.argv[1]

    print(f"Loading mapping from {prepro_dir}/mapping.pkl ...")
    mapping = load_mapping(prepro_dir)

    mol_pairs_train = mapping.get("molecule_pairs_train")
    mol_pairs_val = mapping.get("molecule_pairs_val")
    mol_pairs_val_official = mapping.get("molecule_pairs_val_official")

    split_objs = {
        "train": mol_pairs_train,
        "val_scaffold": mol_pairs_val,
        "val_official": mol_pairs_val_official,
    }

    for split, pairs in PAIRS_TO_CHECK.items():
        mol_pairs = split_objs.get(split)
        if mol_pairs is None:
            print(f"\n[{split}] not in mapping, skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"Checking {split} ({len(mol_pairs.spectra)} unique molecules)")
        print("Loading raw npy pairs ...")
        raw = load_raw_pairs(prepro_dir, PREFIXES[split])

        for idx0, idx1, logged_mces in pairs:
            s0 = get_smiles(mol_pairs, idx0)
            s1 = get_smiles(mol_pairs, idx1)
            # raw MCES in npy: col 3 is raw (0-40), before normalization
            hits = find_pair(raw, idx0, idx1)
            if len(hits) == 0:
                npy_mces = "NOT FOUND in npy"
            else:
                npy_mces_raw = hits[0, MCES_COL]
                # normalize: sim = 1 - raw/40, so raw = (1-sim)*40
                # but raw npy may already be raw (0-40) before normalize_mces20
                npy_mces = f"{npy_mces_raw:.2f} (raw npy col3)"

            match = (
                "OK"
                if len(hits) > 0 and abs(hits[0, MCES_COL] - logged_mces) < 0.1
                else "MISMATCH"
            )
            print(
                f"\n  pair ({idx0}, {idx1})  logged_mces={logged_mces}  npy={npy_mces}  [{match}]"
            )
            print(f"    s0: {s0[:100]}")
            print(f"    s1: {s1[:100]}")


if __name__ == "__main__":
    main()
