"""
Prepare SIMBA preprocessing using max(lb_matrix, HDF5 MCES) as the distance source,
with train/val/test taken directly from Gaetan's split TSV instead of the official
MassSpecGym fold labels.

For each pair:
  - lb_matrix: tight lower bound for MCES >= 10, underestimate for < 10
  - HDF5 all_smiles_mces: exact for MCES < 10, weak lower bound for >= 10
  - Taking max gives exact values for similar pairs and tighter bounds for dissimilar pairs.

Splits: Gaetan's TSV (two columns, `id` and `fold`) assigns every spectrum directly
to train/val/test, keyed by the MGF's IDENTIFIER field — no scaffold re-split.
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


MGF_PATH = "/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf"
SPLIT_TSV = "/sofia/projects/2026_053/spectrawl_project/spectrawl/spectrawl/splits/split_massspecgym.tsv"
LB_MATRIX = "/sofia/projects/2026_053/simba_project/data/massspecgym/lb_matrix.npy"
LB_SMILES = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/lb_matrix.smiles.txt"
)
HDF5_PATH = "/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/all_smiles_mces.hdf5"
OUT_DIR = Path(
    "/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5"
)

MCES_CAP = 40.0

MIN_N_PEAKS = 6
PROTONIZED_ADDUCTS = {"M+", "[M+H]+", "M+H"}


# ── helpers ───────────────────────────────────────────────────────────────────


def is_valid_for_simba(spec):
    mz, intensities = spec.peaks.mz, spec.peaks.intensities
    if mz is None or intensities is None or len(mz) < MIN_N_PEAKS:
        return False
    if not np.all(intensities > 0):
        return False
    ionmode = spec.get("ionmode")
    if ionmode and ionmode.lower() != "positive":
        return False
    adduct = spec.get("adduct")
    if adduct and adduct not in PROTONIZED_ADDUCTS:
        return False
    smiles = spec.get("smiles")
    if smiles is not None and smiles == "N/A":
        return False
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


def load_split_assignments(tsv_path):
    """Read Gaetan's two-column (id, fold) TSV into an {id: fold} dict."""
    assignments = {}
    with open(tsv_path) as fh:
        next(fh)  # header
        for line in fh:
            id_, fold = line.rstrip("\n").split("\t")
            assignments[id_] = fold
    return assignments


def build_pairs(local_mol_indices, lb_indices, lb, hdf5_indices, hdf5_mces, hdf5_n):
    """
    Return (N,4) float64 [local_i, local_j, ed=0, mces] for ALL pairs.
    mces = max(lb_matrix value, hdf5 value) clipped to MCES_CAP.
    If a molecule is missing from either source (index == -1), falls back to the other.
    """
    rows = []
    m = len(local_mol_indices)
    for a in tqdm(range(m), desc="  pairing", leave=False):
        li = local_mol_indices[a]
        lb_i = lb_indices[a]
        hdf5_i = hdf5_indices[a]
        bs = np.arange(a + 1, m)
        lb_js = lb_indices[bs]
        hdf5_js = hdf5_indices[bs]

        # ── lb_matrix values ──────────────────────────────────────────
        hi = np.maximum(lb_i, lb_js)
        lo = np.minimum(lb_i, lb_js)
        fidx = hi * (hi - 1) // 2 + lo
        order = np.argsort(fidx)
        lb_vals = lb[fidx[order]][np.argsort(order)].astype(np.float64)

        # ── HDF5 values (both molecules must be in HDF5) ─────────────────────
        # HDF5MCESCache condensed-index convention (i < j): n*i - i*(i+1)//2 + j-i-1.
        # NOT the same layout as lb_matrix above — verified against known exact
        # MCES values; using the lb_matrix formula here silently returns wrong
        # distances (confirmed empirically, off by a lot, not just a rounding
        # difference).
        hdf5_mask = (hdf5_i >= 0) & (hdf5_js >= 0)
        hdf5_vals = np.zeros(len(bs), dtype=np.float64)
        if hdf5_mask.any() and hdf5_i >= 0:
            lo2 = np.minimum(hdf5_i, hdf5_js[hdf5_mask])
            hi2 = np.maximum(hdf5_i, hdf5_js[hdf5_mask])
            fidx2 = hdf5_n * lo2 - lo2 * (lo2 + 1) // 2 + hi2 - lo2 - 1
            hdf5_vals[hdf5_mask] = hdf5_mces[fidx2]

        # ── max of both sources ──────────────────────────────────────────────
        # Where HDF5 is available use max; otherwise use lb only
        vals = np.where(hdf5_mask, np.maximum(lb_vals, hdf5_vals), lb_vals)
        vals = np.clip(vals, 0.0, MCES_CAP)

        ljs = local_mol_indices[bs]
        rows.append(
            np.column_stack(
                [
                    np.full(len(ljs), li, dtype=np.float64),
                    ljs.astype(np.float64),
                    np.zeros(len(ljs), dtype=np.float64),
                    vals.astype(np.float64),
                ]
            )
        )

    return np.concatenate(rows) if rows else np.empty((0, 4), dtype=np.float64)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read Gaetan's split assignments
    print("Reading split TSV...")
    assignments = load_split_assignments(SPLIT_TSV)
    print(f"  {len(assignments)} spectra in split TSV")

    # 2. Read MGF, assigning folds from the split TSV instead of the MGF's own FOLD field
    print("Reading MGF...")
    fold_spectra = defaultdict(list)
    n_total = n_valid = n_missing_split = 0
    for mgf_idx, spec in tqdm(
        enumerate(matchms.importing.load_from_mgf(MGF_PATH)),
        desc="  MGF spectra",
        unit="spec",
    ):
        n_total += 1
        fold = assignments.get(spec.get("identifier"))
        if not fold:
            n_missing_split += 1
            continue
        if not is_valid_for_simba(spec):
            continue
        n_valid += 1
        fold_spectra[fold].append((mgf_idx, spec))

    print(f"  {n_total} total, {n_valid} valid, {n_missing_split} not in split TSV")
    for f, items in fold_spectra.items():
        print(f"  {f}: {len(items)} spectra")

    # 3. Load lb_matrix SMILES index
    print("Loading lb_matrix SMILES index...")
    lb_smiles_to_idx = {}
    with open(LB_SMILES) as fh:
        for i, line in tqdm(enumerate(fh), desc="  lb SMILES", unit="mol"):
            lb_smiles_to_idx[line.strip()] = i
    print(f"  {len(lb_smiles_to_idx)} molecules in lb_matrix")

    # 4. Load lb_matrix as mmap
    print("Opening lb_matrix.npy (mmap)...")
    lb = np.load(LB_MATRIX, mmap_mode="r")
    print(f"  shape={lb.shape}, dtype={lb.dtype}")

    # 5. Load HDF5 SMILES index and mces array into RAM (~2.4 GB)
    print("Loading HDF5 mces into RAM...")
    with h5py.File(HDF5_PATH, "r") as hf:
        hdf5_smiles_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in hf["mces_smiles_order"][:]
        ]
        hdf5_smiles_to_idx = {s: i for i, s in enumerate(hdf5_smiles_list)}
        hdf5_mces = hf["mces"][:].astype(np.float64)
    print(
        f"  {len(hdf5_smiles_to_idx)} molecules in HDF5, mces array shape={hdf5_mces.shape}"
    )

    # 6. Build df_smiles + spectrum_indexes per fold
    dfs, spec_idxs, lb_idxs_map, hdf5_idxs_map = {}, {}, {}, {}

    for fold, items in fold_spectra.items():
        smiles_groups = defaultdict(list)
        smiles_meta = {}
        for local_pos, (_mgf_idx, spec) in tqdm(
            enumerate(items), desc=f"  grouping {fold}", total=len(items), leave=False
        ):
            smi = canonicalize(spec.get("smiles") or "")
            smiles_groups[smi].append(local_pos)
            if smi not in smiles_meta:
                smiles_meta[smi] = {
                    "mz": spec.get("precursor_mz"),
                    "inchi": spec.get("inchikey"),
                }

        rows, lb_idxs, hdf5_idxs = [], [], []
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
            lb_idxs.append(lb_smiles_to_idx.get(smi, -1))
            hdf5_idxs.append(hdf5_smiles_to_idx.get(smi, -1))

        df = pd.DataFrame(rows)
        dfs[fold] = df
        spec_idxs[fold] = [mgf_idx for mgf_idx, _ in items]
        lb_idxs_map[fold] = np.array(lb_idxs)
        hdf5_idxs_map[fold] = np.array(hdf5_idxs)

        lb_missing = (lb_idxs_map[fold] == -1).sum()
        hdf5_missing = (hdf5_idxs_map[fold] == -1).sum()
        print(
            f"  {fold}: {len(df)} unique mols, lb_missing={lb_missing}, hdf5_missing={hdf5_missing}"
        )

    splits = {
        "train": (
            dfs["train"],
            spec_idxs["train"],
            lb_idxs_map["train"],
            hdf5_idxs_map["train"],
        ),
        "val": (
            dfs["val"],
            spec_idxs["val"],
            lb_idxs_map["val"],
            hdf5_idxs_map["val"],
        ),
        "test": (
            dfs.get("test", pd.DataFrame()),
            spec_idxs.get("test", []),
            lb_idxs_map.get("test", np.array([])),
            hdf5_idxs_map.get("test", np.array([])),
        ),
    }

    # 7. Build pair npy files
    for fold, (df_fold, _si, lb_fold_idxs, hdf5_fold_idxs) in splits.items():
        if df_fold.empty:
            print(f"  {fold}: empty, skipping")
            continue

        valid_mask = lb_fold_idxs != -1
        local_indices = np.where(valid_mask)[0]
        global_lb_idxs = lb_fold_idxs[valid_mask]
        global_hdf5_idxs = hdf5_fold_idxs[valid_mask]

        print(
            f"\nBuilding pairs for {fold} ({len(local_indices)}/{len(df_fold)} matched in lb_matrix)..."
        )
        pairs = build_pairs(
            local_indices,
            global_lb_idxs,
            lb,
            global_hdf5_idxs,
            hdf5_mces,
            len(hdf5_smiles_to_idx),
        )
        print(f"  -> {len(pairs):,} pairs (MCES clipped at {MCES_CAP})")

        out = OUT_DIR / f"ed_mces_indexes_tani_incremental_{fold}_node0_chunk0.npy"
        np.save(out, pairs)

    # 8. Save mapping.pkl
    mapping = {
        "df_smiles_train": splits["train"][0],
        "df_smiles_val": splits["val"][0],
        "df_smiles_test": splits["test"][0],
        "spectrum_indexes_train": splits["train"][1],
        "spectrum_indexes_val": splits["val"][1],
        "spectrum_indexes_test": splits["test"][1],
        "mgf_path": MGF_PATH,
        "format_version": "lightweight",
    }
    with open(OUT_DIR / "mapping.pkl", "wb") as fh:
        pickle.dump(mapping, fh)

    print(f"\nDone. Output: {OUT_DIR}")
    for fold, (df_fold, si, _, _) in splits.items():
        if not df_fold.empty:
            print(f"  {fold}: {len(df_fold)} molecules, {len(si)} spectra")


if __name__ == "__main__":
    main()
