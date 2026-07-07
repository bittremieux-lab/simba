"""
Prepare SIMBA preprocessing files using Murcko scaffold split.

Uses MSG official test fold as test (unchanged).
Keeps MSG official val fold as val_official (second validation set).
Splits MSG official train into new_train / new_val using Murcko scaffold split
(90% of scaffold groups -> train, 10% -> val, seed=42).

ED column is set to 0 (disabled via tasks.edit_distance.enabled=false at train time).
"""

import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import matchms
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


MGF_PATH = "/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf"
HDF5_PATH = "/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5"
OUT_DIR = Path("/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_scaffold_split")
MCES_THRESHOLD = 20.0

MIN_N_PEAKS = 6  # mirrors cfg.data.preprocessing.min_n_peaks default
PROTONIZED_ADDUCTS = {"M+", "[M+H]+", "M+H"}


def is_valid_for_simba(spec: matchms.Spectrum) -> bool:
    """Mirror is_valid_spectrum_janssen from the SIMBA training loader exactly.

    Checks: protonized adduct, precursor_mz, min peaks, no NaN, centroid
    (all intensities > 0), charge == 1 if present, ionmode == positive if
    present, smiles field present and not "N/A".
    """
    mz = spec.peaks.mz
    intensities = spec.peaks.intensities
    if mz is None or intensities is None or len(mz) < MIN_N_PEAKS:
        return False

    # is_centroid: all intensities strictly positive
    if not np.all(intensities > 0):
        return False

    # ion mode: positive only (if field present)
    ionmode = spec.get("ionmode")
    if ionmode and ionmode.lower() != "positive":
        return False

    # adduct: only protonized (if field present)
    adduct = spec.get("adduct")
    if adduct and adduct not in PROTONIZED_ADDUCTS:
        return False

    # smiles: not N/A (if field present)
    smiles = spec.get("smiles")
    if smiles is not None and smiles == "N/A":
        return False

    # NaN / finite / non-negative check (mirrors training.py loader NaN check)
    return not (
        np.isnan(mz).any()
        or np.isnan(intensities).any()
        or not np.isfinite(mz).all()
        or not np.isfinite(intensities).all()
        or not np.all(mz >= 0)
        or not np.all(intensities >= 0)
    )


def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def flat_idx(i, j, n):
    """Upper triangle (i < j) flat index."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + j - i - 1


def build_pairs(local_mol_indices, global_mol_indices, mces_flat, n):
    """Return (N,4) array: [local_i, local_j, ed=0, mces] for mces <= threshold."""
    rows = []
    m = len(local_mol_indices)
    for li in tqdm(range(m), desc="  pairing", leave=False):
        gi = global_mol_indices[li]
        # all j > li
        ljs = np.arange(li + 1, m)
        gjs = global_mol_indices[ljs]
        # ensure gi < gj in the flat index formula
        gi_arr = np.where(gi < gjs, gi, gjs)
        gj_arr = np.where(gi < gjs, gjs, gi)
        fidx = gi_arr * n - gi_arr * (gi_arr + 1) // 2 + gj_arr - gi_arr - 1
        mces_vals = mces_flat[fidx]
        mask = mces_vals <= MCES_THRESHOLD
        if mask.any():
            rows.append(
                np.column_stack(
                    [
                        np.full(mask.sum(), li, dtype=np.float64),
                        ljs[mask].astype(np.float64),
                        np.zeros(mask.sum(), dtype=np.float64),
                        mces_vals[mask].astype(np.float64),
                    ]
                )
            )
    return np.concatenate(rows) if rows else np.empty((0, 4), dtype=np.float64)


def build_split_spectra(df_subset, orig_spec_idxs):
    """
    df_subset: rows from original df_smiles (indexes point into orig_spec_idxs)
    orig_spec_idxs: original flat list of MGF indices
    Returns (new_df, new_spec_idxs) with df_smiles indexes re-pointed to new_spec_idxs
    """
    # Collect all old local positions referenced by this subset (preserve order)
    old_positions = []
    seen = set()
    for idxs in df_subset["indexes"]:
        for p in idxs:
            if p not in seen:
                old_positions.append(p)
                seen.add(p)
    old_positions.sort()

    # Build mapping: old local pos -> new local pos
    old_to_new = {old: new for new, old in enumerate(old_positions)}

    # new_spec_idxs: MGF indices in new local order
    new_spec_idxs = [orig_spec_idxs[p] for p in old_positions]

    # Rebuild df with new local indexes
    new_rows = []
    for _, row in df_subset.iterrows():
        new_row = row.copy()
        new_row["indexes"] = [old_to_new[p] for p in row["indexes"]]
        new_row["number_indexes"] = len(new_row["indexes"])
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    return new_df, new_spec_idxs


def get_murcko_scaffold(smi):
    """Compute Murcko scaffold SMILES. Returns '' on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def scaffold_split(df_train, seed=42, val_fraction=0.1):
    """
    Split df_train into (train_indices, val_indices) by Murcko scaffold.

    Groups molecules by scaffold, shuffles scaffold groups with seed, assigns
    first 90% of scaffold groups to train and last 10% to val.

    Returns (train_mol_indices, val_mol_indices): lists of df_train row labels.
    """
    scaffolds = defaultdict(list)
    for idx, row in df_train.iterrows():
        scaffold = get_murcko_scaffold(row["canon_smiles"])
        scaffolds[scaffold].append(idx)

    scaffold_groups = list(scaffolds.items())

    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_groups)

    n_val = max(1, int(len(scaffold_groups) * val_fraction))
    n_train = len(scaffold_groups) - n_val

    train_groups = scaffold_groups[:n_train]
    val_groups = scaffold_groups[n_train:]

    train_mol_indices = [idx for _, idxs in train_groups for idx in idxs]
    val_mol_indices = [idx for _, idxs in val_groups for idx in idxs]

    return train_mol_indices, val_mol_indices


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Read MGF, filter to valid spectra, group by fold ─────────────────
    print("Reading MGF (filtering to SIMBA-valid spectra)...")
    fold_spectra = defaultdict(list)  # fold -> [(mgf_idx, spectrum)]
    n_total = 0
    n_valid = 0
    for mgf_idx, spec in enumerate(matchms.importing.load_from_mgf(MGF_PATH)):
        n_total += 1
        fold = spec.get("fold")
        if not fold:
            continue
        if not is_valid_for_simba(spec):
            continue
        n_valid += 1
        fold_spectra[fold].append((mgf_idx, spec))

    print(
        f"  Total spectra: {n_total}, valid: {n_valid}, filtered: {n_total - n_valid}"
    )
    for fold, items in fold_spectra.items():
        print(f"  {fold}: {len(items)} valid spectra")

    # ── 2. Read HDF5 ─────────────────────────────────────────────────────────
    print("Reading MCES HDF5...")
    with h5py.File(HDF5_PATH, "r") as f:
        smiles_order_raw = list(f["mces_smiles_order"][:])
        mces_flat = f["mces"][:]

    smiles_order = [s.decode() if isinstance(s, bytes) else s for s in smiles_order_raw]
    n = len(smiles_order)
    assert len(mces_flat) == n * (n - 1) // 2, "HDF5 size mismatch"
    print(f"  {n} unique SMILES, {len(mces_flat)} pairs in upper triangle")

    canon_to_hdf5_idx = {canonicalize(s): i for i, s in enumerate(smiles_order)}

    # ── 3. Build df_smiles and spectrum_indexes for MSG train, val, and test ──
    # "val" is kept as val_official (second validation set alongside scaffold val).
    dfs_orig = {}
    spec_idxs_orig = {}
    hdf5_idxs_orig = {}

    for fold in ("train", "val", "test"):
        items = fold_spectra.get(fold, [])
        if not items:
            print(f"  WARNING: fold '{fold}' is empty!")
            continue

        smiles_groups = defaultdict(list)
        smiles_meta = {}
        for local_pos, (_mgf_idx, spec) in enumerate(items):
            smi = canonicalize(spec.get("smiles") or "")
            smiles_groups[smi].append(local_pos)
            if smi not in smiles_meta:
                smiles_meta[smi] = {
                    "mz": spec.get("precursor_mz"),
                    "inchi": spec.get("inchikey"),
                }

        rows = []
        hdf5_idxs = []
        for smi, idxs in smiles_groups.items():
            meta = smiles_meta[smi]
            rows.append(
                {
                    "canon_smiles": smi,
                    "indexes": idxs,
                    "number_indexes": len(idxs),
                    "mz": meta["mz"],
                    "charge": None,
                    "library": "MassSpecGym",
                    "inchi": meta["inchi"],
                    "bms": None,
                    "superclass": None,
                    "classe": None,
                    "subclass": None,
                }
            )
            hdf5_idxs.append(canon_to_hdf5_idx.get(smi, -1))

        df = pd.DataFrame(rows)
        dfs_orig[fold] = df
        spec_idxs_orig[fold] = [mgf_idx for mgf_idx, _ in items]
        hdf5_idxs_orig[fold] = np.array(hdf5_idxs)

        missing = (hdf5_idxs_orig[fold] == -1).sum()
        print(f"  {fold}: {len(df)} unique SMILES, {missing} not found in HDF5")

    # ── 4. Scaffold split of MSG train into new_train / new_val ─────────────
    print("Computing Murcko scaffold split (seed=42, val_fraction=0.1)...")
    df_train_orig = dfs_orig["train"]
    train_mol_indices, val_mol_indices = scaffold_split(
        df_train_orig, seed=42, val_fraction=0.1
    )
    print(
        f"  {len(train_mol_indices)} molecules -> new_train, "
        f"{len(val_mol_indices)} molecules -> new_val"
    )

    # Subset rows; row labels are 0-based integers so .loc and numpy indexing agree
    df_new_train_raw = df_train_orig.loc[train_mol_indices]
    df_new_val_raw = df_train_orig.loc[val_mol_indices]

    # Rebuild local spectrum indexes for each new split
    orig_train_spec_idxs = spec_idxs_orig["train"]
    df_new_train, spec_idxs_new_train = build_split_spectra(
        df_new_train_raw, orig_train_spec_idxs
    )
    df_new_val, spec_idxs_new_val = build_split_spectra(
        df_new_val_raw, orig_train_spec_idxs
    )

    # HDF5 indices for new splits (preserve row order from scaffold_split)
    hdf5_new_train = hdf5_idxs_orig["train"][train_mol_indices]
    hdf5_new_val = hdf5_idxs_orig["train"][val_mol_indices]

    print(
        f"  new_train: {len(df_new_train)} molecules, "
        f"{(hdf5_new_train == -1).sum()} not in HDF5"
    )
    print(
        f"  new_val:   {len(df_new_val)} molecules, "
        f"{(hdf5_new_val == -1).sum()} not in HDF5"
    )

    # ── 5. Build pair npy files (intra-split only) ───────────────────────────
    splits = {
        "train": (df_new_train, spec_idxs_new_train, hdf5_new_train),
        "val": (df_new_val, spec_idxs_new_val, hdf5_new_val),
        "val_official": (
            dfs_orig.get("val", pd.DataFrame()),
            spec_idxs_orig.get("val", []),
            hdf5_idxs_orig.get("val", np.array([])),
        ),
        "test": (
            dfs_orig.get("test", pd.DataFrame()),
            spec_idxs_orig.get("test", []),
            hdf5_idxs_orig.get("test", np.array([])),
        ),
    }

    pair_counts = {}
    mces_lt10_counts = {}

    for fold, (_df_fold, _spec_fold, hdf5_fold) in splits.items():
        if len(hdf5_fold) == 0:
            print(f"  WARNING: {fold} has no molecules, skipping pair build.")
            pair_counts[fold] = 0
            mces_lt10_counts[fold] = 0
            continue

        valid_mask = hdf5_fold != -1
        local_indices = np.where(valid_mask)[0]
        global_indices_valid = hdf5_fold[valid_mask]

        print(
            f"Building pairs for {fold} "
            f"({len(local_indices)} molecules with HDF5 match)..."
        )
        pairs = build_pairs(local_indices, global_indices_valid, mces_flat, n)
        pair_counts[fold] = len(pairs)
        mces_lt10_counts[fold] = int((pairs[:, 3] < 10).sum()) if len(pairs) > 0 else 0
        print(f"  -> {len(pairs)} pairs with MCES <= {MCES_THRESHOLD}")

        out_path = OUT_DIR / f"ed_mces_indexes_tani_incremental_{fold}_node0_chunk0.npy"
        np.save(out_path, pairs)

    # ── 6. Save mapping.pkl ──────────────────────────────────────────────────
    mapping = {
        "df_smiles_train": df_new_train,
        "df_smiles_val": df_new_val,
        "df_smiles_val_official": dfs_orig.get("val", pd.DataFrame()),
        "df_smiles_test": dfs_orig.get("test", pd.DataFrame()),
        "spectrum_indexes_train": spec_idxs_new_train,
        "spectrum_indexes_val": spec_idxs_new_val,
        "spectrum_indexes_val_official": spec_idxs_orig.get("val", []),
        "spectrum_indexes_test": spec_idxs_orig.get("test", []),
        "mgf_path": MGF_PATH,
        "format_version": "lightweight",
    }
    with open(OUT_DIR / "mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print(f"\nDone. Output: {OUT_DIR}")
    print(
        f"{'Split':<14} {'Molecules':>10} {'Spectra':>10} {'Pairs':>10} {'MCES<10':>10}"
    )
    print("-" * 58)
    for fold, (df_fold, spec_fold, _) in splits.items():
        print(
            f"{fold:<14} {len(df_fold):>10} {len(spec_fold):>10} "
            f"{pair_counts[fold]:>10} {mces_lt10_counts[fold]:>10}"
        )


if __name__ == "__main__":
    main()
