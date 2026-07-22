"""
Oracle retrieval using spectral cosine similarity as the oracle criterion.

For each test spectrum, finds the training spectrum with the highest
peak-based cosine similarity (1 Da binned, sqrt intensities, L2-normalized),
then transfers that molecule's Morgan FP and ranks retrieval candidates by Tanimoto.

Compare with:
  oracle_retrieval_max_lb_hdf5.py  — GT-MCES structural oracle (14.79%/30.92%/53.29%)
  simba_retrieval.py               — SIMBA embedding cosine  (4.26%/11.17%/21.46%)
"""

import json
from pathlib import Path

import matchms
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm.auto import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────

MGF_PATH = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf")
CANDIDATES = Path(
    "/mnt/data2/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json"
)
OUTPUT_TSV = Path(
    "/home/nkubrakov/simba/results/oracle_retrieval_spectral_cosine_bs2048_v2_step44k.tsv"
)

KS = (1, 5, 20)
BIN_SIZE = 1.0  # Da per bin
N_BINS = 2000  # 0–2000 Da


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


def spectrum_to_vector(spec) -> np.ndarray | None:
    """
    Bin spectrum peaks into a fixed-length vector and L2-normalize.
    Uses sqrt intensities (standard spectral cosine preprocessing).
    Returns None if spectrum has no usable peaks.
    """
    mzs = spec.peaks.mz
    ints = spec.peaks.intensities
    if mzs is None or len(mzs) == 0:
        return None
    ints = np.sqrt(ints.astype(np.float32))
    v = np.zeros(N_BINS, dtype=np.float32)
    idx = np.floor(mzs / BIN_SIZE).astype(int)
    mask = (idx >= 0) & (idx < N_BINS)
    np.add.at(v, idx[mask], ints[mask])
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    v /= norm
    return v


def load_fold(mgf_path: Path, fold: str):
    """Load SMILES + binned vectors for a given fold."""
    smiles_list, vectors = [], []
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
        v = spectrum_to_vector(spec)
        if v is None:
            continue
        smiles_list.append(smi)
        vectors.append(v)
    return smiles_list, np.stack(vectors, axis=0)  # (N, N_BINS)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # 1. Load train + test spectra, bin them
    train_smiles_raw, train_mat = load_fold(MGF_PATH, "train")
    test_smiles_raw, test_mat = load_fold(MGF_PATH, "test")

    train_can = [canonicalize(s) for s in train_smiles_raw]
    test_can = [canonicalize(s) for s in test_smiles_raw]
    print(f"  train: {len(train_can)} spectra,  test: {len(test_can)} spectra")
    print(f"  matrix shapes: train={train_mat.shape}, test={test_mat.shape}")

    # 2. Cosine sim via batch matrix multiply (no full matrix stored)
    print("Computing spectral cosine sim (batched) ...")
    BATCH = 200  # test spectra per batch
    nn_indices = np.empty(len(test_can), dtype=np.int64)
    nn_sims = np.empty(len(test_can), dtype=np.float32)

    for start in tqdm(range(0, len(test_can), BATCH), desc="  batches"):
        end = min(start + BATCH, len(test_can))
        sim_block = test_mat[start:end] @ train_mat.T  # (batch, n_train)
        nn_indices[start:end] = sim_block.argmax(axis=1)
        nn_sims[start:end] = sim_block.max(axis=1)

    print(
        f"  spectral cosine: mean={nn_sims.mean():.4f}  median={np.median(nn_sims):.4f}"
    )

    oracle_train_smi = [train_can[i] for i in nn_indices]

    # 3. Load candidates
    print("\nLoading candidates ...")
    with open(CANDIDATES) as fh:
        candidate_json = json.load(fh)
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}
    cand_lists = [cand_json_canon.get(s, []) for s in test_can]
    n_no_cands = sum(1 for c in cand_lists if not c)
    print(f"  {n_no_cands}/{len(test_can)} spectra with no candidates")

    # 4. Pre-compute Morgan FPs
    print("Computing Morgan fingerprints ...")
    all_smi: set[str] = set(oracle_train_smi)
    for cands in cand_lists:
        all_smi.update(cands)
    fp_lookup: dict[str, np.ndarray] = {
        s: morgan_fp(s) for s in tqdm(all_smi, desc="  Morgan FPs")
    }

    # 5. Score
    print("\nScoring ...")
    hits = dict.fromkeys(KS, 0)
    n, n_skip = 0, 0
    for q_smi, nn_smi, cands in zip(test_can, oracle_train_smi, cand_lists):
        if not cands:
            n_skip += 1
            continue
        q_fp = fp_lookup.get(nn_smi)
        if q_fp is None:
            n_skip += 1
            continue
        cand_fp_mat = np.stack([fp_lookup.get(c, morgan_fp(c)) for c in cands])
        scores = tanimoto_scores(q_fp, cand_fp_mat)
        ranked = [cands[i] for i in np.argsort(-scores)]
        for k in KS:
            if any(canonicalize(c) == q_smi for c in ranked[:k]):
                hits[k] += 1
        n += 1

    print("\n=== Oracle Retrieval (spectral cosine) ===")
    print(f"n={n}  skipped: {n_skip}")
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
