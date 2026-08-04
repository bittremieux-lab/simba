"""
SIMBA + ICEBERG retrieval: score real test spectra directly against
ICEBERG-predicted candidate spectra with SIMBA's own similarity head — no
train spectra involved.

For each test spectrum:
  1. Look up its candidate pool (formula-matched) and the ICEBERG-predicted
     spectrum for each candidate (see tools/../ICEBERG/build_candidate_tsv.py
     and the predict_smis.py run that produced them).
  2. Embed the real test spectrum and every candidate's predicted spectrum
     with SIMBA (same encoder + head as embed_spectra in simba_retrieval.py).
  3. Rank candidates by cosine similarity of the (L2-normalized) embeddings —
     equivalent to the model's own trained similarity function for every
     head_mode, since exp(-dist) and cosine similarity are both monotonic in
     the same angle between L2-normalized vectors.

Usage:
    uv run python tools/simba_retrieval_iceberg.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --head_mode cosine_no_head \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --iceberg_preds /path/to/preds.hdf5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from simba_retrieval import canonicalize, embed_spectra, load_model, load_spectra
from tqdm.auto import tqdm

from simba.core.data.spectrum import SpectrumExt


def load_candidate_index(candidate_tsv: str) -> pd.DataFrame:
    """Load the (smiles, adduct) -> cand_id / precursor mapping built for ICEBERG."""
    df = pd.read_csv(candidate_tsv, sep="\t")
    df = df.set_index(["smiles", "ionization"])
    return df


def load_iceberg_spectra(iceberg_preds: str, cand_ids: list[str]) -> dict:
    """Load (masses, intensities) for each cand_id from the ICEBERG predictions HDF5."""
    result = {}
    with h5py.File(iceberg_preds, "r") as f:
        manifest = f["__predspec_manifest__"]
        name_to_leaf = {}
        for name, leaf in zip(manifest["name"][:], manifest["leaf_path"][:]):
            # manifest names are "pred_<spec>"; TSV/cand_ids use "<spec>" without the prefix
            name_to_leaf[name.decode().removeprefix("pred_")] = leaf.decode()

        wanted = set(cand_ids)
        for name, leaf in tqdm(name_to_leaf.items(), desc="Loading ICEBERG spectra"):
            if name not in wanted:
                continue
            arr = f[leaf]["f"][:]
            mask = arr[:, 0] > 0
            result[name] = (
                arr[mask, 0].astype(np.float64),
                arr[mask, 1].astype(np.float64),
            )
    return result


def build_candidate_spectra(
    cand_index: pd.DataFrame, iceberg_specs: dict
) -> tuple[list[str], list[SpectrumExt]]:
    """Build one SpectrumExt per (smiles, adduct) row that has an ICEBERG prediction."""
    smiles_out, specs_out = [], []
    for (smi, adduct), row in cand_index.iterrows():
        cand_id = row["spec"]
        if cand_id not in iceberg_specs:
            continue
        mz, intensity = iceberg_specs[cand_id]
        specs_out.append(
            SpectrumExt(
                identifier=cand_id,
                precursor_mz=float(row["precursor"]),
                precursor_charge=1,
                mz=mz,
                intensity=intensity,
                retention_time=np.nan,
                params={},
                library=None,
                inchi=None,
                smiles=smi,
                ionmode="positive",
                adduct=adduct,
                ce=35,
                ion_activation=None,
                ionization_method=None,
                bms=None,
                superclass=None,
                classe=None,
                subclass=None,
                fold="candidate_pred",
            )
        )
        smiles_out.append(smi)
    return smiles_out, specs_out


def rank_candidates(
    test_smiles: list,
    test_adducts: list,
    query_candidates: dict,
    cand_smi_to_row: dict,
    test_embs: torch.Tensor,
    cand_embs: torch.Tensor,
    top_k: int = 20,
) -> list:
    """For each test query, rank its candidates by SIMBA similarity and keep the top_k.

    Returns a list (same order/length as test_smiles) of either a ranked
    candidate-SMILES list (length <= top_k) or None if the query had no
    usable candidates.
    """
    per_query = []
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if not row_idxs:
            per_query.append(None)
            continue

        sims = (test_embs[i : i + 1] @ cand_embs[row_idxs].T).squeeze(0)
        order = torch.argsort(sims, descending=True)
        ranked = [cand_smis[j] for j in order.tolist()]
        per_query.append(ranked[:top_k])
    return per_query


def compute_hit_rates_from_ranking(
    test_smiles: list, per_query_ranked: list, ks: tuple = (1, 5, 20)
) -> tuple[dict, int]:
    hits = dict.fromkeys(ks, 0)
    n = 0
    n_no_candidates = 0

    for q_smi, ranked in zip(test_smiles, per_query_ranked):
        if ranked is None:
            n_no_candidates += 1
            continue
        q_canon = canonicalize(q_smi)
        for k in ks:
            if any(canonicalize(c) == q_canon for c in ranked[:k]):
                hits[k] += 1
        n += 1

    return {f"hit@{k}": hits[k] / n if n > 0 else 0.0 for k in ks}, n_no_candidates


def compute_mces_batch(
    smiles_0: list, smiles_1: list, threshold_mces: int, num_jobs: int, work_dir: str
) -> np.ndarray:
    """Batch-compute myopic MCES for parallel (smiles_0[i], smiles_1[i]) pairs.

    Uses a dedicated work_dir (not the shared "temp/" the MCES.compute_mces_list_smiles
    convenience wrapper hardcodes) so this can't collide with a concurrent run, and
    exposes num_jobs since that wrapper hardcodes --num_jobs 1.
    """
    import subprocess

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    input_csv = work / "mces_input.csv"
    output_csv = work / "mces_output.csv"

    pd.DataFrame({"smiles_0": smiles_0, "smiles_1": smiles_1}).to_csv(
        input_csv, header=False
    )

    command = [
        "myopic_mces",
        str(input_csv),
        str(output_csv),
        "--num_jobs",
        str(num_jobs),
        "--threshold",
        str(int(threshold_mces)),
        "--solver_onethreaded",
        "--solver_no_msg",
        "--choose_bound_dynamically",
        "--catch_computation_errors",
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

    # output columns: index, mces distance, computation time (s), computation mode.
    # Don't assume row order matches input order — parallel workers (--num_jobs)
    # can return out of sequence; realign explicitly via the index column.
    results = pd.read_csv(output_csv, header=None)
    results = results.sort_values(0)
    return results[1].to_numpy()


def compute_mces_stats(
    test_smiles: list,
    per_query_ranked: list,
    threshold_mces: int = 40,
    num_jobs: int = -1,
    work_dir: str = "temp_mces",
    ks: tuple = (1, 5, 20),
) -> dict:
    """GT MCES between top-1 / min-over-top-k ranked candidates and the true molecule."""
    pair_query_smi, pair_cand_smi, pair_query_idx = [], [], []
    for qi, (q_smi, ranked) in enumerate(zip(test_smiles, per_query_ranked)):
        if not ranked:
            continue
        for c in ranked:
            pair_query_smi.append(q_smi)
            pair_cand_smi.append(c)
            pair_query_idx.append(qi)

    print(
        f"Computing myopic MCES for {len(pair_query_smi)} (query, candidate) pairs ..."
    )
    mces_vals = compute_mces_batch(
        pair_query_smi, pair_cand_smi, threshold_mces, num_jobs, work_dir
    )

    per_query_mces = {}
    for qi, m in zip(pair_query_idx, mces_vals):
        per_query_mces.setdefault(qi, []).append(m)

    # k=1 is just the top-1 candidate's own MCES (min of a single-element list);
    # named "mces_top1" rather than "mces_min_top1" since "min" is misleading there.
    results = {}
    for k in ks:
        vals = [min(v[:k]) for v in per_query_mces.values()]
        arr = np.array(vals, dtype=float)
        name = "mces_top1" if k == 1 else f"mces_min_top{k}"
        results[f"{name}_mean"] = float(arr.mean())
        results[f"{name}_median"] = float(np.median(arr))
    return results


def run(
    checkpoint: str,
    head_mode: str,
    mgf: str,
    candidates: str,
    candidate_tsv: str,
    iceberg_preds: str,
    split: str = "test",
    batch_size: int = 512,
    output_tsv: str | None = None,
    intermediates_dir: str | None = None,
    threshold_mces: int = 40,
    num_mces_jobs: int = -1,
    skip_mces: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {split}-fold real spectra from {mgf} ...")
    test_smiles, test_spectra = load_spectra(mgf, split)
    test_adducts = [s.adduct for s in test_spectra]
    print(f"  {len(test_smiles)} real test spectra")

    print(f"\nLoading candidate pools from {candidates} ...")
    with open(candidates) as fh:
        candidate_json = json.load(fh)
    query_candidates = {canonicalize(k): v for k, v in candidate_json.items()}

    print(f"\nLoading candidate index from {candidate_tsv} ...")
    cand_index = load_candidate_index(candidate_tsv)
    cand_smi_to_row = {}  # (smiles, adduct) -> row position, filled in after embedding

    print(f"\nLoading ICEBERG-predicted spectra from {iceberg_preds} ...")
    all_cand_ids = cand_index["spec"].tolist()
    iceberg_specs = load_iceberg_spectra(iceberg_preds, all_cand_ids)
    print(
        f"  {len(iceberg_specs)} / {len(all_cand_ids)} candidates have a predicted spectrum"
    )

    print("\nBuilding candidate SpectrumExt objects ...")
    cand_smiles, cand_spectra = build_candidate_spectra(cand_index, iceberg_specs)
    for row_idx, (smi, spec) in enumerate(zip(cand_smiles, cand_spectra)):
        cand_smi_to_row[(smi, spec.adduct)] = row_idx
    print(f"  {len(cand_spectra)} candidate spectra ready")

    print(f"\nLoading SIMBA checkpoint: {checkpoint} (head_mode={head_mode})")
    model = load_model(checkpoint, device, head_mode=head_mode)

    print("\nEmbedding real test spectra ...")
    test_embs = embed_spectra(model, test_spectra, batch_size, device)
    print("\nEmbedding ICEBERG-predicted candidate spectra ...")
    cand_embs = embed_spectra(model, cand_spectra, batch_size, device)

    if intermediates_dir:
        out_dir = Path(intermediates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(test_embs, out_dir / "test_embeddings.pt")
        torch.save(cand_embs, out_dir / "candidate_embeddings.pt")
        (out_dir / "test_smiles.json").write_text(json.dumps(test_smiles))
        (out_dir / "candidate_smiles.json").write_text(json.dumps(cand_smiles))
        print(f"Intermediates saved to {out_dir}/")

    print("\nRanking candidates ...")
    per_query_ranked = rank_candidates(
        test_smiles,
        test_adducts,
        query_candidates,
        cand_smi_to_row,
        test_embs,
        cand_embs,
        top_k=20,
    )
    results, n_no_candidates = compute_hit_rates_from_ranking(
        test_smiles, per_query_ranked
    )
    if n_no_candidates:
        print(
            f"  Warning: {n_no_candidates}/{len(test_smiles)} queries had no usable candidates"
        )

    n_scored = len(test_smiles) - n_no_candidates
    print(f"\n=== SIMBA+ICEBERG retrieval ({split}, n={n_scored}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if not skip_mces:
        print("\nComputing GT MCES between ranked candidates and the true molecule ...")
        mces_work_dir = (
            str(Path(intermediates_dir) / "mces_work")
            if intermediates_dir
            else "temp_mces"
        )
        mces_results = compute_mces_stats(
            test_smiles,
            per_query_ranked,
            threshold_mces=threshold_mces,
            num_jobs=num_mces_jobs,
            work_dir=mces_work_dir,
        )
        results.update(mces_results)
        print("\n=== GT MCES to true molecule (myopic, capped at threshold) ===")
        for k, v in mces_results.items():
            print(f"  {k}: {v:.3f}")

    if output_tsv:
        pd.DataFrame(
            [
                {
                    "split": split,
                    "model": Path(checkpoint).parent.name,
                    "head_mode": head_mode,
                    "n": n_scored,
                    **results,
                }
            ]
        ).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results


def main():
    p = argparse.ArgumentParser(
        description="SIMBA+ICEBERG retrieval: score real spectra against ICEBERG-predicted candidate spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="SIMBA .ckpt file")
    p.add_argument("--head_mode", default="cosine_relu")
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidates", required=True, help="Candidate JSON {smiles: [cands]}"
    )
    p.add_argument(
        "--candidate_tsv",
        required=True,
        help="ICEBERG candidate TSV (smiles/ionization/precursor)",
    )
    p.add_argument("--iceberg_preds", required=True, help="ICEBERG predictions HDF5")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory to save embeddings and SMILES lists",
    )
    p.add_argument(
        "--threshold_mces",
        type=int,
        default=40,
        help="Cap for GT MCES computation (myopic_mces --threshold)",
    )
    p.add_argument(
        "--num_mces_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for MCES computation (-1 = all logical CPUs)",
    )
    p.add_argument(
        "--skip_mces",
        action="store_true",
        help="Skip GT MCES computation, report only hit@k",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
