"""
Prepare SIMBA preprocessing files from official MassSpecGym splits.

Reads fold=train/val/test from MassSpecGym.mgf, pairs molecules using the
precomputed MCES from all_smiles_mces.hdf5, and writes mapping.pkl + pair npy
files in the same format as simba preprocess output.

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
from tqdm import tqdm


MGF_PATH = "/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf"
HDF5_PATH = "/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5"
OUT_DIR = Path("/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_official")
MCES_THRESHOLD = 20.0

MIN_N_PEAKS = 6  # mirrors cfg.data.preprocessing.min_n_peaks default
PROTONIZED_ADDUCTS = {"M+", "[M+H]+", "M+H"}


def is_valid_for_simba(spec: matchms.Spectrum) -> bool:
    """Replicate SIMBA's is_valid_spectrum_janssen filter (use_only_protonized_adducts=True).

    Mirrors: LoadData.is_valid_spectrum_janssen + NaN check in loaders.py.
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
        print(f"  {fold}: {len(items)} spectra")

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

    # ── 3. Build df_smiles and spectrum_indexes per fold ────────────────────
    # df_smiles["indexes"] = local positions in spectrum_indexes (NOT global MGF indices)
    # spectrum_indexes     = global MGF indices (for loading spectra at training time)
    dfs, spec_idxs = {}, {}
    smiles_to_hdf5 = {}  # per-fold: local_mol_idx -> hdf5_idx

    for fold, items in fold_spectra.items():
        # Group by canonical SMILES; local_pos = position in items / spectrum_indexes
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
        dfs[fold] = df
        spec_idxs[fold] = [mgf_idx for mgf_idx, _ in items]
        smiles_to_hdf5[fold] = np.array(hdf5_idxs)

        missing = (smiles_to_hdf5[fold] == -1).sum()
        print(f"  {fold}: {len(df)} unique SMILES, {missing} not found in HDF5")

    # ── 4. Build pair npy files ──────────────────────────────────────────────
    for fold in ("train", "val", "test"):
        if fold not in dfs:
            continue
        global_indices = smiles_to_hdf5[fold]
        valid_mask = global_indices != -1
        local_indices = np.where(valid_mask)[0]
        global_indices_valid = global_indices[valid_mask]

        print(
            f"Building pairs for {fold} ({len(local_indices)} molecules with HDF5 match)..."
        )
        pairs = build_pairs(local_indices, global_indices_valid, mces_flat, n)
        print(f"  -> {len(pairs)} pairs with MCES <= {MCES_THRESHOLD}")

        out_path = OUT_DIR / f"ed_mces_indexes_tani_incremental_{fold}_node0_chunk0.npy"
        np.save(out_path, pairs)

    # ── 5. Save mapping.pkl ──────────────────────────────────────────────────
    mapping = {
        "df_smiles_train": dfs.get("train", pd.DataFrame()),
        "df_smiles_val": dfs.get("val", pd.DataFrame()),
        "df_smiles_test": dfs.get("test", pd.DataFrame()),
        "spectrum_indexes_train": spec_idxs.get("train", []),
        "spectrum_indexes_val": spec_idxs.get("val", []),
        "spectrum_indexes_test": spec_idxs.get("test", []),
        "mgf_path": MGF_PATH,
        "format_version": "lightweight",
    }
    with open(OUT_DIR / "mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)

    print(f"\nDone. Output: {OUT_DIR}")
    for fold in ("train", "val", "test"):
        if fold in dfs:
            print(
                f"  {fold}: {len(dfs[fold])} molecules, {len(spec_idxs[fold])} spectra"
            )


if __name__ == "__main__":
    main()
