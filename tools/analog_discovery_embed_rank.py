"""014_2 analog discovery (see NOTES_014_2_ANALOG_DISCOVERY.md), stage 2:
embed CASMI queries + one reference library with 014_2 and score every
query x library molecule pair under three schemes:

  - simba_raw:    raw MCES regression, (1 - cosine(raw_emb0, raw_emb1)) * mces_max_value
  - simba_corn:   CORN-corrected ranking score (corrected*1000 + raw) --
                  substitutes for the paper's "MCES + edit-distance tiebreak"
                  since 014_2 has no edit-distance head (see NOTES doc)
  - cosine:       plain (non-modified) binned-peak spectral cosine, no SIMBA
                  involved -- reported as a distance (1 - cosine) so all
                  three scores share an ascending-is-closer convention

Produces one (n_query, n_library) score matrix per scheme plus the ordered
canonical-SMILES lists for both axes -- the common ranking input for the
boxplot/ROC/ranking-performance panels. Exact GT MCES for the same
n_query x n_library molecule-pair space is computed separately, on CPU
(tools/analog_discovery_exact_mces.py), and combined at analysis time.

GPU. Reuses the validated 014_2-aware loader from tools/simba_retrieval.py
(load_model/embed_spectra/load_spectra) and the plain-cosine building block
from tools/cosine_baseline_iceberg.py (bin_spectra) -- NOT
simba/analog_discovery/*, whose checkpoint loader silently drops 014_2's
CORN bucket head (see NOTES_014_2_ANALOG_DISCOVERY.md, "Existing codebase").

Run once per search (search_A_nist_msg.mgf, search_B_gnps.mgf).

Usage:
    uv run python tools/analog_discovery_embed_rank.py \\
        --checkpoint experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt \\
        --mgf data/analog_discovery/search_A_nist_msg.mgf \\
        --output_dir data/analog_discovery/search_A_scores
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from cosine_baseline_iceberg import bin_spectra
from simba_retrieval import canonicalize, embed_spectra, load_model, load_spectra
from simba_retrieval_iceberg import _corn_corrected_ranking_score, _corn_decode_bucket
from tqdm.auto import tqdm


def score_simba(
    model,
    q_embs_raw: torch.Tensor,
    lib_embs_raw: torch.Tensor,
    device: torch.device,
    pair_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score_raw, score_corn), each (n_query, n_library), both
    ascending-is-closer. Loops one query at a time against library chunks
    through model.compute_from_embeddings -- the bucket head is a function of
    the pair (abs(emb0-emb1) on raw, magnitude-sensitive embeddings), so it
    can't be reduced to a single cached matmul the way cosine similarity can
    (same reasoning as rank_candidates_corn_corrected in simba_retrieval_iceberg.py).
    """
    n_q, n_lib = q_embs_raw.shape[0], lib_embs_raw.shape[0]
    score_raw = np.empty((n_q, n_lib), dtype=np.float32)
    score_corn = np.empty((n_q, n_lib), dtype=np.float32)

    for i in tqdm(range(n_q), desc="SIMBA pairwise scoring", unit="query"):
        emb0_full = q_embs_raw[i : i + 1]
        for start in range(0, n_lib, pair_chunk_size):
            end = min(start + pair_chunk_size, n_lib)
            emb0 = emb0_full.expand(end - start, -1).to(device)
            emb1 = lib_embs_raw[start:end].to(device)
            with torch.no_grad():
                _, emb_sim_2, emb_sim_3 = model.compute_from_embeddings(emb0, emb1)
            pred_mces = ((1.0 - emb_sim_2) * model.mces_max_value).cpu().numpy()
            bucket_pred = _corn_decode_bucket(emb_sim_3).cpu().numpy()
            score_raw[i, start:end] = pred_mces
            score_corn[i, start:end] = _corn_corrected_ranking_score(
                pred_mces, bucket_pred
            )

    return score_raw, score_corn


def score_cosine(
    q_spectra: list, lib_spectra: list, bin_width: float, max_mz: float
) -> np.ndarray:
    """Plain binned-peak spectral cosine, as a (n_query, n_library) distance
    matrix (1 - cosine), ascending-is-closer to match the other two scores."""
    q_mat = bin_spectra(q_spectra, bin_width, max_mz)
    lib_mat = bin_spectra(lib_spectra, bin_width, max_mz)
    sims = np.asarray((q_mat @ lib_mat.T).todense())
    return 1.0 - sims


def run(
    checkpoint: str,
    mgf: str,
    output_dir: str,
    batch_size: int = 256,
    pair_chunk_size: int = 20000,
    precursor_mass_mode: str = "theoretical",
    head_mode: str = "cosine_no_head",
    mces_bucket_use_mlp: bool = False,
    mces_bucket_use_product: bool = False,
    bin_width: float = 0.01,
    max_mz: float = 1100.0,
    d_model: int = 256,
    n_layers: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint} (device={device}) ...")
    model = load_model(
        checkpoint,
        device,
        head_mode=head_mode,
        use_mces_bucket_head=True,
        mces_bucket_use_mlp=mces_bucket_use_mlp,
        mces_bucket_use_product=mces_bucket_use_product,
        d_model=d_model,
        n_layers=n_layers,
    )

    print(
        f"Loading query/library spectra from {mgf} (precursor_mass_mode={precursor_mass_mode}) ..."
    )
    q_smiles, q_spectra = load_spectra(mgf, "query", precursor_mass_mode)
    lib_smiles, lib_spectra = load_spectra(mgf, "library", precursor_mass_mode)
    q_smiles = [canonicalize(s) for s in q_smiles]
    lib_smiles = [canonicalize(s) for s in lib_smiles]
    print(f"  {len(q_smiles)} query molecules, {len(lib_smiles)} library molecules")

    print("Embedding queries ...")
    _, q_embs_raw = embed_spectra(model, q_spectra, batch_size, device, return_raw=True)
    print("Embedding library ...")
    _, lib_embs_raw = embed_spectra(
        model, lib_spectra, batch_size, device, return_raw=True
    )

    print("Scoring SIMBA (raw regression + CORN-corrected) ...")
    score_raw, score_corn = score_simba(
        model, q_embs_raw, lib_embs_raw, device, pair_chunk_size
    )

    print(f"Scoring plain cosine (bin_width={bin_width}, max_mz={max_mz}) ...")
    score_cos = score_cosine(q_spectra, lib_spectra, bin_width, max_mz)

    print(f"Saving to {out_dir} ...")
    out_dir.joinpath("smiles_query.txt").write_text("\n".join(q_smiles) + "\n")
    out_dir.joinpath("smiles_library.txt").write_text("\n".join(lib_smiles) + "\n")
    np.save(out_dir / "score_simba_raw.npy", score_raw)
    np.save(out_dir / "score_simba_corn.npy", score_corn)
    np.save(out_dir / "score_cosine.npy", score_cos.astype(np.float32))

    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--mgf", required=True, help="Combined query+library MGF with FOLD= field"
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--batch_size", type=int, default=256, help="SIMBA embedding batch size"
    )
    p.add_argument(
        "--pair_chunk_size",
        type=int,
        default=20000,
        help="Library chunk size per query for the pairwise bucket-head forward pass",
    )
    p.add_argument(
        "--precursor_mass_mode",
        default="theoretical",
        choices=["measured", "theoretical"],
    )
    p.add_argument("--head_mode", default="cosine_no_head")
    p.add_argument(
        "--mces_bucket_use_mlp",
        action="store_true",
        help="Must match the checkpoint's model.tasks.mces_bucket.use_mlp (true for 014_2)",
    )
    p.add_argument(
        "--mces_bucket_use_product",
        action="store_true",
        help="Must match the checkpoint's model.tasks.mces_bucket.use_product",
    )
    p.add_argument(
        "--bin_width", type=float, default=0.01, help="Plain-cosine m/z bin width (Da)"
    )
    p.add_argument(
        "--max_mz", type=float, default=1100.0, help="Plain-cosine max m/z (Da)"
    )
    p.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Must match the checkpoint's model.transformer.d_model -- the checkpoint "
        "itself doesn't record this, so a mismatch silently drops weights (strict=False) "
        "instead of erroring",
    )
    p.add_argument(
        "--n_layers",
        type=int,
        default=5,
        help="Must match the checkpoint's model.transformer.n_layers (see --d_model)",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
