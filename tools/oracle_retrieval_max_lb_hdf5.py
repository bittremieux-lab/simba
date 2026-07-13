"""
Oracle retrieval: upper bound on the SIMBA NN-transfer approach.

For each test molecule, finds the nearest training-set molecule by
max(Gaetan lb_matrix, MSG HDF5) MCES — identical to the distance source
used during training — then transfers that molecule's Morgan FP and ranks
retrieval candidates by Tanimoto, exactly as simba_retrieval.py does.
"""

import json
import pickle
from pathlib import Path

import h5py
import matchms
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm.auto import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────

MAPPING_PKL = Path(
    "/mnt/data/nkubrakov/massspecgym/preprocessing_msg_max_lb_hdf5/mapping.pkl"
)
LB_MATRIX = Path("/mnt/data2/gdewaele/lb_matrix.npy")
LB_SMILES = Path("/mnt/data2/gdewaele/lb_matrix.smiles.txt")
HDF5_PATH = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5")
MGF_PATH = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf")
CANDIDATES = Path(
    "/mnt/data2/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json"
)
OUTPUT_TSV = Path("/home/nkubrakov/simba/results/oracle_retrieval_max_lb_hdf5.tsv")

MCES_CAP = 40.0
KS = (1, 5, 20)


# ── Helpers ───────────────────────────────────────────────────────────────────


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_scores(query_fp: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    inter = (query_fp & cand_fps).sum(axis=1)
    union = (query_fp | cand_fps).sum(axis=1)
    return np.where(union > 0, inter / union, 0.0)


def condensed_idx(i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray:
    """Lower-triangle index: entry (hi, lo) with hi > lo sits at hi*(hi-1)//2 + lo."""
    hi = np.maximum(i, j).astype(np.int64)
    lo = np.minimum(i, j).astype(np.int64)
    return hi * (hi - 1) // 2 + lo


def load_spectra(mgf_path: Path, fold: str):
    """Same minimal filter as simba_retrieval.py: needs SMILES and >=1 peak."""
    smiles_list, spectra = [], []
    for spec in tqdm(
        matchms.importing.load_from_mgf(str(mgf_path)),
        desc=f"Loading MGF ({fold})",
        unit="spec",
    ):
        if spec.get("fold") != fold:
            continue
        smi = spec.get("smiles")
        if not smi:
            continue
        if spec.peaks is None or len(spec.peaks.mz) < 1:
            continue
        smiles_list.append(smi)
        spectra.append(spec)
    return smiles_list, spectra


# ── Oracle NN ─────────────────────────────────────────────────────────────────


def oracle_nn_for_test_mol(
    test_lb_idx: int,
    test_hdf5_idx: int,
    train_lb_idxs: np.ndarray,
    train_hdf5_idxs: np.ndarray,
    lb: np.ndarray,
    hdf5_mces: np.ndarray,
) -> tuple[int, float]:
    """
    Return (train_idx, min_mces) for the oracle nearest train molecule.
    Uses max(lb_matrix, hdf5) exactly as the training data preprocessing.
    Returns (-1, nan) if neither source can provide a value for the test mol.
    """
    if test_lb_idx < 0 and test_hdf5_idx < 0:
        return -1, float("nan")

    n_train = len(train_lb_idxs)
    # Start from MCES_CAP — will be overwritten where sources are available.
    vals = np.full(n_train, MCES_CAP, dtype=np.float64)

    # Gaetan lb_matrix values
    if test_lb_idx >= 0:
        lb_valid = train_lb_idxs >= 0
        if lb_valid.any():
            fidx = condensed_idx(np.int64(test_lb_idx), train_lb_idxs[lb_valid])
            vals[lb_valid] = lb[fidx].astype(np.float64)

    # HDF5 values — take max with whatever lb gave us
    if test_hdf5_idx >= 0:
        hdf5_valid = train_hdf5_idxs >= 0
        if hdf5_valid.any():
            fidx2 = condensed_idx(np.int64(test_hdf5_idx), train_hdf5_idxs[hdf5_valid])
            vals[hdf5_valid] = np.maximum(vals[hdf5_valid], hdf5_mces[fidx2])

    vals = np.clip(vals, 0.0, MCES_CAP)
    nn_idx = int(np.argmin(vals))
    return nn_idx, float(vals[nn_idx])


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # 1. Train molecules from preprocessing (same pool the model was trained on)
    print("Loading mapping.pkl ...")
    with open(MAPPING_PKL, "rb") as f:
        mapping = pickle.load(f)
    train_canonical = [
        canonicalize(s) for s in mapping["df_smiles_train"]["canon_smiles"]
    ]
    print(f"  train molecules: {len(train_canonical)}")

    # 2. lb_matrix SMILES index
    print("Loading lb_matrix SMILES index ...")
    lb_smiles_to_idx: dict[str, int] = {}
    with open(LB_SMILES) as fh:
        for i, line in enumerate(tqdm(fh, desc="  lb SMILES", unit="mol")):
            lb_smiles_to_idx[line.strip()] = i
    print(f"  {len(lb_smiles_to_idx):,} molecules in lb_matrix")

    # 3. lb_matrix mmap (never fully loaded into RAM)
    print("Opening lb_matrix.npy (mmap) ...")
    lb = np.load(LB_MATRIX, mmap_mode="r")
    print(f"  shape={lb.shape}, dtype={lb.dtype}")

    # 4. HDF5 mces into RAM (~2.4 GB, same as preprocessing does)
    print("Loading HDF5 mces into RAM ...")
    with h5py.File(HDF5_PATH, "r") as hf:
        hdf5_smiles_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in hf["mces_smiles_order"][:]
        ]
        hdf5_smiles_to_idx = {s: i for i, s in enumerate(hdf5_smiles_list)}
        hdf5_mces = hf["mces"][:].astype(np.float64)
    print(f"  {len(hdf5_smiles_to_idx):,} molecules, mces shape={hdf5_mces.shape}")

    # 5. Map train molecules to source indices
    print("Mapping train SMILES to source indices ...")
    train_lb_idxs = np.array(
        [lb_smiles_to_idx.get(s, -1) for s in train_canonical], dtype=np.int64
    )
    train_hdf5_idxs = np.array(
        [hdf5_smiles_to_idx.get(s, -1) for s in train_canonical], dtype=np.int64
    )
    print(
        f"  lb_missing={int((train_lb_idxs < 0).sum())}, "
        f"hdf5_missing={int((train_hdf5_idxs < 0).sum())}"
    )

    # 6. Load test spectra (same minimal filter as simba_retrieval.py)
    print("\nLoading test spectra ...")
    test_smiles_raw, _ = load_spectra(MGF_PATH, "test")
    test_canonical = [canonicalize(s) for s in test_smiles_raw]
    print(f"  {len(test_canonical)} test spectra")

    # 7. Map unique test molecules to source indices
    unique_test = list(dict.fromkeys(test_canonical))
    test_lb_map = {s: lb_smiles_to_idx.get(s, -1) for s in unique_test}
    test_hdf5_map = {s: hdf5_smiles_to_idx.get(s, -1) for s in unique_test}
    print(
        f"  {len(unique_test)} unique test mols, "
        f"lb_missing={sum(1 for v in test_lb_map.values() if v < 0)}, "
        f"hdf5_missing={sum(1 for v in test_hdf5_map.values() if v < 0)}"
    )

    # 8. Oracle NN for each unique test molecule
    print("\nComputing oracle NN (max(lb_matrix, HDF5) MCES) ...")
    oracle_train_smi: dict[str, str | None] = {}
    oracle_mces_val: dict[str, float] = {}
    for smi in tqdm(unique_test, desc="  Oracle NN", unit="mol"):
        nn_idx, min_mces = oracle_nn_for_test_mol(
            test_lb_map[smi],
            test_hdf5_map[smi],
            train_lb_idxs,
            train_hdf5_idxs,
            lb,
            hdf5_mces,
        )
        oracle_train_smi[smi] = train_canonical[nn_idx] if nn_idx >= 0 else None
        oracle_mces_val[smi] = min_mces

    n_found = sum(1 for v in oracle_train_smi.values() if v is not None)
    found_mces = [v for v in oracle_mces_val.values() if not np.isnan(v)]
    print(f"  oracle NN found for {n_found}/{len(unique_test)} test mols")
    if found_mces:
        arr = np.array(found_mces)
        print(
            f"  oracle MCES to nearest train: mean={arr.mean():.1f}, "
            f"median={np.median(arr):.1f}, min={arr.min():.1f}, "
            f"p10={np.percentile(arr, 10):.1f}, p25={np.percentile(arr, 25):.1f}"
        )

    # 9. Load candidates and build FP lookup
    print("\nLoading candidates ...")
    with open(CANDIDATES) as fh:
        candidate_json = json.load(fh)
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}
    cand_lists = [cand_json_canon.get(s, []) for s in test_canonical]
    n_no_cands = sum(1 for c in cand_lists if not c)
    print(f"  {n_no_cands}/{len(test_canonical)} spectra with no candidates")

    print("Computing Morgan fingerprints ...")
    all_fp_smiles: set[str] = set()
    for smi in unique_test:
        nn_smi = oracle_train_smi.get(smi)
        if nn_smi:
            all_fp_smiles.add(nn_smi)
    for cands in cand_lists:
        all_fp_smiles.update(cands)
    fp_lookup: dict[str, np.ndarray] = {
        s: morgan_fp(s) for s in tqdm(all_fp_smiles, desc="  Morgan FPs")
    }

    # 10. Score
    print("\nScoring ...")
    hits = dict.fromkeys(KS, 0)
    n = 0
    n_skip_no_oracle = 0
    n_skip_no_cands = 0

    for q_smi, cands in zip(test_canonical, cand_lists):
        if not cands:
            n_skip_no_cands += 1
            continue
        nn_smi = oracle_train_smi.get(q_smi)
        if nn_smi is None or nn_smi not in fp_lookup:
            n_skip_no_oracle += 1
            continue

        q_fp = fp_lookup[nn_smi]
        cand_fp_mat = np.stack([fp_lookup.get(c, morgan_fp(c)) for c in cands])
        scores = tanimoto_scores(q_fp, cand_fp_mat)
        ranked = [cands[i] for i in np.argsort(-scores)]

        for k in KS:
            if any(canonicalize(c) == q_smi for c in ranked[:k]):
                hits[k] += 1
        n += 1

    print("\n=== Oracle Retrieval (max(Gaetan lb, HDF5) MCES) ===")
    print(
        f"n={n}  skipped: {n_skip_no_cands} no-candidates, {n_skip_no_oracle} no-oracle-NN"
    )
    for k in KS:
        r = hits[k] / n if n > 0 else 0.0
        print(f"  hit@{k:<2} = {r:.4f}  ({hits[k]}/{n},  {r * 100:.2f}%)")

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, "w") as f:
        f.write("n\thit@1\thit@5\thit@20\n")
        f.write(f"{n}\t{hits[1] / n:.6f}\t{hits[5] / n:.6f}\t{hits[20] / n:.6f}\n")
    print(f"\nSaved to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
