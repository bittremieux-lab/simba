"""
SIMBA retrieval baseline: SIMBA-embedding NN transfer + Tanimoto candidate ranking.

For each test spectrum:
  1. Find nearest train spectrum by cosine similarity of SIMBA embeddings.
  2. Transfer that train molecule's Morgan fingerprint as the predicted FP.
  3. Rank leaderboard candidates by Tanimoto similarity to the transferred FP.

Usage:
    uv run python tools/simba_retrieval.py \\
        --checkpoint /mnt/data/nkubrakov/experiments_3_dataset/training/msg_max_lb_hdf5_mces40/checkpoint-epoch=06-step=70000.ckpt \\
        --mgf /mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf \\
        --candidates /mnt/data2/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json \\
        --split test
"""

import argparse
import copy
import json
from pathlib import Path

import matchms
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm.auto import tqdm

from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS
from simba.core.data.encoding import encode_adduct_mass
from simba.core.data.preprocessor import Preprocessor
from simba.core.data.spectrum import SpectrumExt
from simba.core.models.similarity_models import SimilarityModelMultitask


MAX_PEAKS = 100  # must match model.transformer.context_length
N_ADDUCTS = len(ADDUCT_TO_MASS)


# ── Model ─────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, device: torch.device) -> SimilarityModelMultitask:
    model = SimilarityModelMultitask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        d_model=256,
        n_layers=5,
        n_classes=11,
        use_gumbel=False,
        lr=1e-4,
        use_cosine_distance=True,
        use_edit_distance=False,
        strict=False,
    )
    model.eval()
    return model.to(device)


# ── Spectrum loading ───────────────────────────────────────────────────────────


def _to_spectrum_ext(spec: matchms.Spectrum, fold: str) -> SpectrumExt:
    """Wrap a matchms Spectrum as a SpectrumExt so it can go through Preprocessor."""
    charge = spec.get("charge")
    charge = int(charge) if charge else 1

    ce_val = spec.get("ce") or spec.get("collision_energy")
    ce = None
    if ce_val is not None:
        try:
            ce = int(float(ce_val))
        except (ValueError, TypeError):
            ce = None

    ionmode = spec.get("ionmode")
    ionmode = ionmode.lower() if ionmode else "none"

    return SpectrumExt(
        identifier=str(spec.get("scans") or ""),
        precursor_mz=float(spec.get("precursor_mz") or 0.0),
        precursor_charge=charge,
        mz=np.asarray(spec.peaks.mz, dtype=np.float64),
        intensity=np.asarray(spec.peaks.intensities, dtype=np.float64),
        retention_time=np.nan,
        params={},
        library=None,
        inchi=None,
        smiles=spec.get("smiles"),
        ionmode=ionmode,
        adduct=spec.get("adduct"),
        ce=ce,
        ion_activation=None,
        ionization_method=None,
        bms=None,
        superclass=None,
        classe=None,
        subclass=None,
        fold=fold,
    )


def load_spectra(mgf_path: str, fold: str):
    """Return (smiles, spectra) for the given fold from the MGF file.

    Peaks are still raw at this point — preprocessing happens in
    spectra_to_tensors below.
    """
    smiles_list, spectra = [], []
    for spec in tqdm(
        matchms.importing.load_from_mgf(mgf_path), desc=f"Loading {fold}", unit="spec"
    ):
        if spec.get("fold") != fold:
            continue
        smi = spec.get("smiles")
        if not smi:
            continue
        if spec.peaks is None or len(spec.peaks.mz) < 1:
            continue
        smiles_list.append(smi)
        spectra.append(_to_spectrum_ext(spec, fold))
    return smiles_list, spectra


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


# ── Embedding ─────────────────────────────────────────────────────────────────


def _encode_ionmode(ionmode_val: str | None) -> float:
    # MassSpecGym has no ionmode field, so this is always the literal string
    # "none" rather than None — which doesn't match "positive" either, so it
    # ends up -1.0 whenever ion mode isn't explicitly positive.
    if ionmode_val is None or ionmode_val == "None":
        return 0.0
    return 1.0 if ionmode_val == "positive" else -1.0


def spectra_to_tensors(specs: list[SpectrumExt], device: torch.device):
    """Pad a list of SpectrumExt to (B, MAX_PEAKS) tensors plus metadata."""
    pp = Preprocessor()
    B = len(specs)
    mz_t = torch.zeros(B, MAX_PEAKS, dtype=torch.float32)
    int_t = torch.zeros(B, MAX_PEAKS, dtype=torch.float32)
    prec_t = torch.zeros(B, dtype=torch.float32)
    charge_t = torch.ones(B, dtype=torch.float32)  # default +1
    ionmode_t = torch.zeros(B, dtype=torch.float32)
    adduct_t = torch.zeros(B, N_ADDUCTS, dtype=torch.float32)
    ce_t = torch.zeros(B, dtype=torch.float32)

    for i, spec in enumerate(specs):
        processed = pp.preprocess_spectrum(
            copy.copy(spec),
            fragment_tol_mass=10,
            fragment_tol_mode="ppm",
            min_intensity=0.01,
            max_num_peaks=MAX_PEAKS,
            scale_intensity=None,
        )
        mz = np.asarray(processed.mz, dtype=np.float32)
        intensity = np.asarray(processed.intensity, dtype=np.float32)
        n = len(mz)
        mz_t[i, :n] = torch.from_numpy(mz)
        int_t[i, :n] = torch.from_numpy(intensity)
        prec_t[i] = float(spec.precursor_mz or 0.0)

        if spec.precursor_charge:
            charge_t[i] = float(spec.precursor_charge)

        ionmode_t[i] = _encode_ionmode(spec.ionmode)

        if spec.adduct:
            adduct_t[i] = torch.tensor(
                encode_adduct_mass(spec.adduct), dtype=torch.float32
            )

        ce_t[i] = 0.0 if spec.ce is None else float(spec.ce)

    # sqrt-compress then L2-normalize each spectrum's intensities.
    int_t = torch.sqrt(int_t.clamp(min=0.0))
    norms = int_t.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    int_t = int_t / norms

    meta = {
        "ionmode": ionmode_t.to(device),
        "adduct": adduct_t.to(device),
        "ce": ce_t.to(device),
    }
    return (
        mz_t.to(device),
        int_t.to(device),
        prec_t.to(device),
        charge_t.to(device),
        meta,
    )


@torch.no_grad()
def embed_spectra(
    model: SimilarityModelMultitask,
    spectra: list,
    batch_size: int,
    device: torch.device,
    force_ce_zero: bool = False,
) -> torch.Tensor:
    """
    Encode spectra through SIMBA encoder + MCES projection head.
    Passes metadata (ionmode/adduct/ce) when the model was trained with them.
    Returns L2-normalized (N, d_model) embeddings — normalized so that
    nearest_neighbor_transfer's all-pairs matmul (`chunk @ train_embs.T`)
    gives cosine similarity directly.

    force_ce_zero: zero out CE for all spectra before passing to the encoder.
    Useful for diagnosing CE-induced embedding collapse: with CE conditioning,
    the model clusters spectra by CE energy rather than molecular structure,
    hurting retrieval. Setting CE=0 everywhere gives CE-agnostic embeddings.
    """
    enc = model.spectrum_encoder
    use_ion_mode = getattr(enc, "use_ion_mode", False)
    use_adduct = getattr(enc, "use_adduct", False)
    use_ce = getattr(enc, "use_ce", False)

    all_embs = []
    for start in tqdm(
        range(0, len(spectra), batch_size), desc="Embedding", unit="batch"
    ):
        batch = spectra[start : start + batch_size]
        mz, intensity, prec, charge, meta = spectra_to_tensors(batch, device)

        if force_ce_zero:
            meta["ce"] = torch.zeros_like(meta["ce"])

        kwargs = {"precursor_mass": prec, "precursor_charge": charge}
        if use_ion_mode:
            kwargs["ionmode"] = meta["ionmode"]
        if use_adduct:
            kwargs["adduct"] = meta["adduct"]
        if use_ce:
            kwargs["ce"] = meta["ce"]

        emb, _ = model.spectrum_encoder(
            mz_array=mz, intensity_array=intensity, **kwargs
        )
        emb = emb[:, 0, :]  # CLS token → (B, d_model)
        emb = model.relu(emb)

        # MCES similarity projection head (same path as compute_from_embeddings)
        emb = model.linear2(emb)
        emb = model.dropout(emb)  # no-op in eval mode
        emb = model.relu(emb)
        emb = model.relu(model.linear2_cossim(emb))

        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs)  # (N, d_model)


# ── Nearest-neighbor transfer ─────────────────────────────────────────────────


def nearest_neighbor_transfer(
    test_embs: torch.Tensor,
    train_embs: torch.Tensor,
    train_fps: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each test embedding find nearest train embedding by cosine similarity.
    Returns (transferred_fps, nn_indices): both (N_test,) / (N_test, nbits) tensors.
    """
    transferred = []
    all_indices = []
    for start in tqdm(
        range(0, len(test_embs), chunk_size), desc="NN transfer", unit="chunk"
    ):
        chunk = test_embs[start : start + chunk_size]  # (C, d)
        sims = chunk @ train_embs.T  # (C, N_train)
        nn_idx = sims.argmax(dim=1)
        transferred.append(train_fps[nn_idx])
        all_indices.append(nn_idx)
    return torch.cat(transferred), torch.cat(all_indices)  # (N_test, nbits), (N_test,)


# ── Morgan fingerprints ───────────────────────────────────────────────────────


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_fp_lookup(smiles_iter, nbits: int = 2048) -> dict:
    return {s: morgan_fp(s, nbits=nbits) for s in tqdm(smiles_iter, desc="Morgan FPs")}


# ── Tanimoto scoring + hit rates ──────────────────────────────────────────────


def tanimoto_scores(query_fp: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    inter = (query_fp & cand_fps).sum(axis=1)
    union = (query_fp | cand_fps).sum(axis=1)
    return np.where(union > 0, inter / union, 0.0)


def compute_hit_rates(
    test_smiles: list,
    cand_lists: list,
    transferred_fps: torch.Tensor,
    fp_lookup: dict,
    ks: tuple = (1, 5, 20),
) -> dict:
    hits = dict.fromkeys(ks, 0)
    n = 0

    for q_smi, cands, q_fp_t in zip(test_smiles, cand_lists, transferred_fps):
        if not cands:
            continue
        q_fp = q_fp_t.numpy().astype(np.uint8)
        cand_fp_mat = np.stack([fp_lookup[c] for c in cands])
        scores = tanimoto_scores(q_fp, cand_fp_mat)
        ranked_cands = [cands[i] for i in np.argsort(-scores)]

        q_canon = canonicalize(q_smi)
        for k in ks:
            if any(canonicalize(c) == q_canon for c in ranked_cands[:k]):
                hits[k] += 1
        n += 1

    return {f"hit@{k}": hits[k] / n if n > 0 else 0.0 for k in ks}


# ── Main ──────────────────────────────────────────────────────────────────────


def save_intermediates(
    out_dir: Path,
    split: str,
    train_smiles,
    test_smiles,
    train_embs,
    test_embs,
    nn_indices,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_embs, out_dir / "train_embeddings.pt")
    torch.save(test_embs, out_dir / f"{split}_embeddings.pt")
    torch.save(nn_indices, out_dir / f"{split}_nn_indices.pt")
    (out_dir / "train_smiles.json").write_text(json.dumps(train_smiles))
    (out_dir / f"{split}_smiles.json").write_text(json.dumps(test_smiles))
    print(f"Intermediates saved to {out_dir}/")


def run(
    checkpoint: str,
    mgf: str,
    candidates: str,
    split: str = "test",
    batch_size: int = 256,
    output_tsv: str | None = None,
    intermediates_dir: str | None = None,
    force_ce_zero: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading candidate JSON from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)

    print(f"\nLoading spectra (split={split}, ref=train) ...")
    train_smiles, train_spectra = load_spectra(mgf, "train")
    test_smiles, test_spectra = load_spectra(mgf, split)
    print(f"  train: {len(train_smiles)}  {split}: {len(test_smiles)}")

    # Deduplicate training: keep first spectrum per unique canonical SMILES.
    # This ensures each training molecule is represented exactly once,
    # so argmax cosine-sim maps unambiguously to a unique molecule.
    print("Deduplicating training spectra (first per unique molecule) ...")
    _seen: set[str] = set()
    _dedup_smi, _dedup_spec = [], []
    for _smi, _spec in zip(train_smiles, train_spectra):
        _c = canonicalize(_smi)
        if _c not in _seen:
            _seen.add(_c)
            _dedup_smi.append(_smi)
            _dedup_spec.append(_spec)
    train_smiles, train_spectra = _dedup_smi, _dedup_spec
    print(f"  After dedup: {len(train_smiles)} unique training molecules")

    # Build candidate lists aligned to test spectra, canonicalize keys
    print("\nBuilding candidate lists ...")
    cand_json_canon = {canonicalize(k): v for k, v in candidate_json.items()}
    cand_lists = [cand_json_canon.get(canonicalize(s), []) for s in test_smiles]
    n_missing = sum(1 for c in cand_lists if not c)
    if n_missing:
        print(f"  Warning: {n_missing}/{len(test_smiles)} queries have no candidates")

    # Fingerprints for all unique SMILES
    print("\nComputing Morgan fingerprints ...")
    all_smiles = set(train_smiles) | {c for lst in cand_lists for c in lst}
    fp_lookup = build_fp_lookup(all_smiles)
    train_fps = torch.from_numpy(
        np.stack([fp_lookup[s] for s in train_smiles])
    )  # (N_train, 2048)

    # Load model and embed
    print(f"\nLoading SIMBA checkpoint: {checkpoint}")
    model = load_model(checkpoint, device)

    if force_ce_zero:
        print("  [CE=0 mode] CE zeroed out for all spectra — CE-agnostic embeddings.")

    print("\nEmbedding train spectra ...")
    train_embs = embed_spectra(
        model, train_spectra, batch_size, device, force_ce_zero=force_ce_zero
    )
    print("\nEmbedding test spectra ...")
    test_embs = embed_spectra(
        model, test_spectra, batch_size, device, force_ce_zero=force_ce_zero
    )

    # NN transfer
    print("\nNearest-neighbor transfer ...")
    transferred_fps, nn_indices = nearest_neighbor_transfer(
        test_embs, train_embs, train_fps
    )

    # Save intermediates before scoring so they're available even if scoring is re-run
    if intermediates_dir:
        save_intermediates(
            Path(intermediates_dir),
            split,
            train_smiles,
            test_smiles,
            train_embs,
            test_embs,
            nn_indices,
        )

    # Evaluate
    print("\nScoring candidates ...")
    results = compute_hit_rates(test_smiles, cand_lists, transferred_fps, fp_lookup)

    print(f"\n=== SIMBA retrieval ({split}, n={len(test_smiles)}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if output_tsv:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "split": split,
                    "model": Path(checkpoint).parent.name,
                    "n": len(test_smiles),
                    **results,
                }
            ]
        ).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results


def main():
    p = argparse.ArgumentParser(
        description="SIMBA retrieval: embedding NN transfer + Tanimoto ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="SIMBA .ckpt file")
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidates", required=True, help="Candidate JSON {smiles: [cands]}"
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory to save embeddings and NN indices",
    )
    p.add_argument(
        "--force_ce_zero",
        action="store_true",
        help="Zero out CE for all spectra — CE-agnostic embedding diagnostic",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
