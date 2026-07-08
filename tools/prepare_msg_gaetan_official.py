"""
Prepare SIMBA preprocessing files using Gaetan's lb_matrix as the distance source.

Splits:
  - Official MSG val  → df_smiles_val_official
  - Official MSG test → df_smiles_test
  - Official MSG train, 90% scaffold → df_smiles_train
  - Official MSG train, 10% scaffold → df_smiles_val  (our internal val)

Distance source: /mnt/data2/gdewaele/lb_matrix.npy
  Lower-triangle condensed format: flat[i,j] = i*(i-1)//2 + j  (i > j)
  240 637 molecules covering MSG + SpectraVerse + Enveda.
  File is 108 GB — accessed via mmap, never fully loaded.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import matchms
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


MGF_PATH = "/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf"
LB_MATRIX = "/mnt/data2/gdewaele/lb_matrix.npy"
LB_SMILES = "/mnt/data2/gdewaele/lb_matrix.smiles.txt"
OUT_DIR = Path("/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_gaetan_official")

MCES_CAP = 40.0
VAL_FRAC = 0.10
SEED = 42

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


def lb_flat(i, j):
    """Lower-triangle condensed index (i > j required; swaps if needed)."""
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j


def build_pairs(local_mol_indices, lb_indices, lb):
    """
    Return (N,4) float64 array [local_i, local_j, ed=0, mces] for ALL pairs,
    with lb values clipped to MCES_CAP (mirrors reference clip-40 behaviour).

    lb_indices: lb_matrix indices for each local molecule (aligned with local_mol_indices).
    lb: mmap array for lb_matrix.npy.
    Reads in sorted flat-index order per row for better mmap locality.
    """
    rows = []
    m = len(local_mol_indices)
    for a in tqdm(range(m), desc="  pairing", leave=False):
        li = local_mol_indices[a]
        lb_i = lb_indices[a]
        bs = np.arange(a + 1, m)
        lb_js = lb_indices[bs]

        # flat indices for this row, ensure i > j in lower-triangle convention
        hi = np.maximum(lb_i, lb_js)
        lo = np.minimum(lb_i, lb_js)
        fidx = hi * (hi - 1) // 2 + lo

        # sorted read for mmap locality
        order = np.argsort(fidx)
        vals = np.clip(lb[fidx[order]][np.argsort(order)], 0.0, MCES_CAP)

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

    # 1. Read MGF, filter valid spectra, group by official fold
    print("Reading MGF...")
    fold_spectra = defaultdict(list)
    n_total = n_valid = 0
    for mgf_idx, spec in tqdm(
        enumerate(matchms.importing.load_from_mgf(MGF_PATH)),
        desc="  MGF spectra",
        unit="spec",
    ):
        n_total += 1
        fold = spec.get("fold")
        if not fold:
            continue
        if not is_valid_for_simba(spec):
            continue
        n_valid += 1
        fold_spectra[fold].append((mgf_idx, spec))

    print(f"  {n_total} total, {n_valid} valid")
    for f, items in fold_spectra.items():
        print(f"  {f}: {len(items)} spectra")

    # 2. Load Gaetan lb_matrix SMILES index
    print("Loading lb_matrix SMILES index...")
    lb_smiles_to_idx = {}
    with open(LB_SMILES) as fh:
        for i, line in tqdm(enumerate(fh), desc="  lb SMILES", unit="mol"):
            lb_smiles_to_idx[line.strip()] = i
    print(f"  {len(lb_smiles_to_idx)} molecules in lb_matrix")

    # 3. Load lb_matrix as mmap
    print("Opening lb_matrix.npy (mmap)...")
    lb = np.load(LB_MATRIX, mmap_mode="r")
    print(f"  shape={lb.shape}, dtype={lb.dtype}")

    # 4. Build df_smiles + spectrum_indexes for each official fold,
    #    then carve 10% val from train.
    dfs, spec_idxs, lb_idxs_map = {}, {}, {}

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

        rows, lb_idxs = [], []
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

        df = pd.DataFrame(rows)
        dfs[fold] = df
        spec_idxs[fold] = [mgf_idx for mgf_idx, _ in items]
        lb_idxs_map[fold] = np.array(lb_idxs)

        missing = (lb_idxs_map[fold] == -1).sum()
        print(f"  {fold}: {len(df)} unique molecules, {missing} not in lb_matrix")

    # 5. Split official train → 90% train / 10% val by Murcko scaffold
    # Molecules sharing a scaffold are always on the same side (no leakage).
    print(
        f"\nSplitting official train by Murcko scaffold (VAL_FRAC={VAL_FRAC}, seed={SEED})..."
    )

    def murcko(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "__no_scaffold__"
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            return "__no_scaffold__"

    train_smiles = dfs["train"]["canon_smiles"].tolist()

    scaffold_groups = defaultdict(list)
    for mol_idx, smi in tqdm(
        enumerate(train_smiles), desc="  scaffolds", total=len(train_smiles)
    ):
        scaffold_groups[murcko(smi)].append(mol_idx)

    groups = list(scaffold_groups.items())
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)

    # Mirror prepare_msg_scaffold_splits.py exactly:
    # 10% of scaffold groups → val (last slice after shuffle)
    n_val_groups = max(1, int(len(groups) * VAL_FRAC))
    n_train_groups = len(groups) - n_val_groups
    train_groups = groups[:n_train_groups]
    val_groups = groups[n_train_groups:]

    train_mol_idx = np.sort([idx for _, idxs in train_groups for idx in idxs])
    val_mol_idx = np.sort([idx for _, idxs in val_groups for idx in idxs])
    n_val_pct = 100 * len(val_mol_idx) / len(train_smiles)
    print(
        f"  {len(groups)} scaffold groups → {n_train_groups} train / {n_val_groups} val groups"
    )
    print(
        f"  train: {len(train_mol_idx)} molecules, val: {len(val_mol_idx)} ({n_val_pct:.1f}%)"
    )

    # Remap spectrum_indexes: items are indexed by local_pos in fold_spectra["train"]
    # df.indexes stores local_pos values; spectrum_indexes["train"][local_pos] = mgf_idx.
    # After splitting, we need to remap local_pos to new indices in each sub-split.

    def subset_fold(df_full, spec_full, lb_full, mol_subset_idx):
        """Extract a subset of molecules from a fold, remapping local indexes."""
        df_sub = df_full.iloc[mol_subset_idx].reset_index(drop=True)
        # spec_full is the flat list of mgf_idxs for the fold; df.indexes contains
        # positions into that list.  Remap to new contiguous positions.
        new_spec_idxs = []
        new_indexes = []
        for _, row in df_sub.iterrows():
            start = len(new_spec_idxs)
            old_positions = row["indexes"]  # positions in spec_full
            mgf_idxs = [spec_full[p] for p in old_positions]
            new_spec_idxs.extend(mgf_idxs)
            new_indexes.append(list(range(start, start + len(mgf_idxs))))
        df_sub = df_sub.copy()
        df_sub["indexes"] = new_indexes
        df_sub["number_indexes"] = df_sub["indexes"].map(len)
        return df_sub, new_spec_idxs, lb_full[mol_subset_idx]

    df_train, si_train, lb_train = subset_fold(
        dfs["train"], spec_idxs["train"], lb_idxs_map["train"], train_mol_idx
    )
    df_val, si_val, lb_val = subset_fold(
        dfs["train"], spec_idxs["train"], lb_idxs_map["train"], val_mol_idx
    )

    splits = {
        "train": (df_train, si_train, lb_train),
        "val": (df_val, si_val, lb_val),
        "val_official": (dfs["val"], spec_idxs["val"], lb_idxs_map["val"]),
        "test": (
            dfs.get("test", pd.DataFrame()),
            spec_idxs.get("test", []),
            lb_idxs_map.get("test", np.array([])),
        ),
    }

    # 6. Build pair npy files
    for fold, (df_fold, _si, lb_fold_idxs) in splits.items():
        if df_fold.empty:
            print(f"  {fold}: empty, skipping")
            continue

        valid_mask = lb_fold_idxs != -1
        local_indices = np.where(valid_mask)[0]
        global_lb_idxs = lb_fold_idxs[valid_mask]

        print(
            f"\nBuilding pairs for {fold} ({len(local_indices)}/{len(df_fold)} molecules matched in lb_matrix)..."
        )
        pairs = build_pairs(local_indices, global_lb_idxs, lb)
        print(f"  -> {len(pairs):,} pairs (MCES clipped at {MCES_CAP})")

        out = OUT_DIR / f"ed_mces_indexes_tani_incremental_{fold}_node0_chunk0.npy"
        np.save(out, pairs)

    # 7. Save mapping.pkl
    mapping = {
        "df_smiles_train": splits["train"][0],
        "df_smiles_val": splits["val"][0],
        "df_smiles_val_official": splits["val_official"][0],
        "df_smiles_test": splits["test"][0],
        "spectrum_indexes_train": splits["train"][1],
        "spectrum_indexes_val": splits["val"][1],
        "spectrum_indexes_val_official": splits["val_official"][1],
        "spectrum_indexes_test": splits["test"][1],
        "mgf_path": MGF_PATH,
        "format_version": "lightweight",
    }
    with open(OUT_DIR / "mapping.pkl", "wb") as fh:
        pickle.dump(mapping, fh)

    print(f"\nDone. Output: {OUT_DIR}")
    for fold, (df_fold, si, _) in splits.items():
        if not df_fold.empty:
            print(f"  {fold}: {len(df_fold)} molecules, {len(si)} spectra")


if __name__ == "__main__":
    main()
