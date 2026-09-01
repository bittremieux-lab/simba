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
  4. (3c) Extend hit@1/5/20 with the MIN GT MCES to the true molecule among
     the top-1/5/20 ranked candidates — the closest (best) wrong guess, not
     the farthest — via a lookup built by tools/prepare_gt_mces_retrieval.py
     + asimov2's exact-MCES computation (see that script's docstring).

--candidate_tsv / --iceberg_preds each accept one path or several (matched
1:1 by position) -- for scoring a query set whose candidates are split
across the original candidates_test_official.tsv/preds.hdf5 plus a delta
file of only the newly-generated pairs (ICEBERG/build_candidate_tsv_delta.py
+ a fresh predict_smis.py run), without needing to physically merge the two
TSVs/HDF5s first. Delta files use disjoint cand_id numbering by
construction, so results merge unambiguously.

Usage:
    uv run python tools/simba_retrieval_iceberg.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --head_mode cosine_no_head \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --candidate_tsv /path/to/candidates_test_official.tsv [/path/to/delta.tsv ...] \\
        --iceberg_preds /path/to/preds.hdf5 [/path/to/delta_preds.hdf5 ...] \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates
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


# Must match callbacks.py's _MCES_BUCKET_EDGES / dashboard_app.py's
# _CORN_BUCKET_EDGES -- the merged 6-class scheme every mces_bucket_head
# checkpoint (013, 014_1-4) was trained with.
CORN_BUCKET_EDGES = np.array([2.0, 4.0, 6.0, 8.0])


def load_candidate_index(candidate_tsv: str | list[str]) -> pd.DataFrame:
    """Load the (smiles, adduct) -> cand_id / precursor mapping built for
    ICEBERG. Accepts one path or a list (e.g. the original
    candidates_test_official.tsv plus a delta file of new pairs for a
    different split's query set, built with disjoint cand_id numbering so
    they never collide) -- concatenated, then indexed as usual."""
    paths = [candidate_tsv] if isinstance(candidate_tsv, str) else list(candidate_tsv)
    dfs = [pd.read_csv(p, sep="\t") for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = df.set_index(["smiles", "ionization"])
    return df


def load_iceberg_spectra(iceberg_preds: str | list[str], cand_ids: list[str]) -> dict:
    """Load (masses, intensities) for each cand_id from the ICEBERG
    predictions HDF5. Accepts one path or a list (paired with
    load_candidate_index's multi-path support -- e.g. the original
    preds.hdf5 plus a delta run's own preds.hdf5) -- results merged into one
    dict, looked up the same way regardless of which file a cand_id came
    from."""
    paths = [iceberg_preds] if isinstance(iceberg_preds, str) else list(iceberg_preds)
    result: dict = {}
    wanted = set(cand_ids)
    for path in paths:
        with h5py.File(path, "r") as f:
            manifest = f["__predspec_manifest__"]
            name_to_leaf = {}
            for name, leaf in zip(manifest["name"][:], manifest["leaf_path"][:]):
                # manifest names are "pred_<spec>"; TSV/cand_ids use "<spec>" without the prefix
                name_to_leaf[name.decode().removeprefix("pred_")] = leaf.decode()

            for name, leaf in tqdm(
                name_to_leaf.items(),
                desc=f"Loading ICEBERG spectra ({Path(path).parent.name})",
            ):
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


def _corn_decode_bucket(logits: torch.Tensor) -> torch.Tensor:
    """Chain-rule decode (cumulative product of conditional probabilities) to a
    predicted ordinal bucket index -- mirrors
    SimilarityModelMultitask._corn_decode_bin_generic exactly (duplicated
    here rather than imported since it's a one-line staticmethod and pulling
    in the Lightning module just for this would be more coupling than it's
    worth)."""
    probas = torch.sigmoid(logits)
    cumprod = torch.cumprod(probas, dim=-1)
    return (cumprod > 0.5).sum(dim=-1)


def _corn_corrected_mces(pred_mces: np.ndarray, bucket_pred: np.ndarray) -> np.ndarray:
    """Clip the primary head's raw MCES prediction into its predicted CORN
    bucket's [left, right] range. Mirrors tools/dashboard_app.py's
    _corn_corrected_mces: boundaries = [0, 0, *CORN_BUCKET_EDGES, inf] so
    bucket 0 clips to exactly [0, 0] (self/singleton), bucket i>=1 clips into
    (boundaries[i], boundaries[i+1]], last bucket is open-ended."""
    boundaries = np.concatenate([[0.0, 0.0], CORN_BUCKET_EDGES, [np.inf]])
    left = boundaries[bucket_pred]
    right = boundaries[bucket_pred + 1]
    return np.clip(pred_mces, left, right)


def _corn_corrected_ranking_score(
    pred_mces: np.ndarray, bucket_pred: np.ndarray
) -> np.ndarray:
    """corrected*1000 + raw breaks exact ties within a bucket (most severely
    bucket 0, where every self-bucket candidate corrects to exactly 0) for
    Hit@k, while leaving the corrected value itself (used for MAE/overlap
    elsewhere) untouched -- see tools/dashboard_app.py's identical formula
    and BASELINE_AND_DASHBOARD.md section 11 for why this is needed."""
    corrected = _corn_corrected_mces(pred_mces, bucket_pred)
    return corrected * 1000.0 + pred_mces


def rank_candidates_corn_corrected(
    test_smiles: list,
    test_adducts: list,
    query_candidates: dict,
    cand_smi_to_row: dict,
    test_embs_raw: torch.Tensor,
    cand_embs_raw: torch.Tensor,
    model,
    device: torch.device,
    top_k: int = 20,
) -> list:
    """Same contract as rank_candidates, but ranking by CORN-corrected MCES
    (lower = closer = better) computed from a genuine pairwise bucket-head
    forward pass instead of a cached-embedding dot product.

    The primary head's cosine-similarity ranking IS decomposable into
    independent per-spectrum embeddings (nn.CosineSimilarity re-normalizes
    internally, so normalized or raw embeddings give the same result) -- but
    the bucket head takes abs(emb0 - emb1) on raw, magnitude-sensitive
    embeddings, which is a function of the *pair*, not of two independent
    embeddings. That's why this needs model.compute_from_embeddings run
    directly on each query's (small, formula-matched) candidate batch,
    rather than a cheap matmul -- only the bucket head's own small MLP runs
    pairwise; the expensive transformer encoding is still done once per
    spectrum, exactly as before.
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

        emb0 = test_embs_raw[i : i + 1].expand(len(row_idxs), -1).to(device)
        emb1 = cand_embs_raw[row_idxs].to(device)
        with torch.no_grad():
            _, emb_sim_2, emb_sim_3 = model.compute_from_embeddings(emb0, emb1)
        pred_mces = ((1.0 - emb_sim_2) * model.mces_max_value).cpu().numpy()
        bucket_pred = _corn_decode_bucket(emb_sim_3).cpu().numpy()
        score = _corn_corrected_ranking_score(pred_mces, bucket_pred)

        order = np.argsort(score)  # ascending: lowest (closest) MCES first
        ranked = [cand_smis[j] for j in order]
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


def load_gt_mces_lookup(gt_mces_dir: str) -> dict[tuple[str, str], float]:
    """Load the exact-MCES lookup built by tools/prepare_gt_mces_retrieval.py +
    asimov2's compute_block/combine (see that script's docstring).

    Reads smiles.txt (mol_idx -> canonical SMILES) and the combined
    mces_exact.npy ((N, 3) = [mol_idx_a, mol_idx_b, mces]) written by that
    script's `combine` subcommand, and returns a symmetric
    {(canon_smi_a, canon_smi_b): mces} dict.

    -1 (solver failed) and NaN (block never computed) entries are dropped —
    "no reliable GT value", not "GT value is -1/NaN".
    """
    gt_dir = Path(gt_mces_dir)
    smiles = gt_dir.joinpath("smiles.txt").read_text().splitlines()
    arr = np.load(gt_dir / "mces_exact.npy")

    valid = ~np.isnan(arr[:, 2]) & (arr[:, 2] != -1.0)
    n_dropped = len(arr) - int(valid.sum())
    if n_dropped:
        print(
            f"  {n_dropped}/{len(arr)} GT MCES pairs unresolved (failed/missing), dropped"
        )

    lookup: dict[tuple[str, str], float] = {}
    for a, b, m in arr[valid]:
        smi_a, smi_b = smiles[int(a)], smiles[int(b)]
        lookup[(smi_a, smi_b)] = float(m)
        lookup[(smi_b, smi_a)] = float(m)
    return lookup


def compute_mces_stats(
    test_smiles: list,
    per_query_ranked: list,
    gt_lookup: dict[tuple[str, str], float],
    ks: tuple = (1, 5, 20),
) -> dict:
    """GT MCES between top-1 / MIN-over-top-k ranked candidates and the true
    molecule — the closest (best) retrieved candidate, not the farthest, so
    this reads as "how close did we get" even when the exact hit was missed.

    A ranked candidate that IS the true molecule (a hit) is credited as
    MCES=0 directly — that pair is deliberately absent from gt_lookup (see
    prepare_gt_mces_retrieval.py: trivial true-match pairs aren't computed),
    so it must not be treated as "missing data" and dropped, or every hit
    would be silently excluded from its own min-over-top-k.
    """
    per_query_mces: dict[int, list[float | None]] = {}
    n_pairs_missing = 0
    for qi, (q_smi, ranked) in enumerate(zip(test_smiles, per_query_ranked)):
        if not ranked:
            continue
        q_canon = canonicalize(q_smi)
        vals: list[float | None] = []
        for c in ranked:
            c_canon = canonicalize(c)
            if c_canon == q_canon:
                vals.append(0.0)  # true match: GT MCES = 0 by definition
                continue
            m = gt_lookup.get((q_canon, c_canon))
            if m is None:
                n_pairs_missing += 1
            vals.append(m)  # None preserves rank position for the v[:k] windows below
        if any(v is not None for v in vals):
            per_query_mces[qi] = vals

    if n_pairs_missing:
        print(
            f"  {n_pairs_missing} ranked (query, candidate) pairs had no GT MCES in "
            "the lookup (excluded from the min, not treated as 0)"
        )

    # k=1 is just the top-1 candidate's own MCES (min of a single-element list);
    # named "mces_top1" rather than "mces_min_top1" since "min" is misleading there.
    results = {}
    for k in ks:
        mins = []
        for v in per_query_mces.values():
            window = [x for x in v[:k] if x is not None]
            if window:
                mins.append(min(window))
        arr = np.array(mins, dtype=float)
        name = "mces_top1" if k == 1 else f"mces_min_top{k}"
        results[f"{name}_mean"] = float(arr.mean()) if len(arr) else float("nan")
        results[f"{name}_median"] = float(np.median(arr)) if len(arr) else float("nan")
        results[f"{name}_n"] = int(len(arr))
    return results


def run(
    checkpoint: str,
    head_mode: str,
    mgf: str,
    candidates: str,
    candidate_tsv: str | list[str],
    iceberg_preds: str | list[str],
    split: str = "test",
    batch_size: int = 512,
    output_tsv: str | None = None,
    intermediates_dir: str | None = None,
    gt_mces_dir: str | None = None,
    skip_mces: bool = False,
    precursor_mass_mode: str = "measured",
    corn_corrected: bool = False,
    mces_bucket_use_mlp: bool = False,
    mces_bucket_use_product: bool = False,
    min_peaks: int | None = None,
    d_model: int = 256,
    n_layers: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(
        f"\nLoading {split}-fold real spectra from {mgf} "
        f"(precursor_mass_mode={precursor_mass_mode}) ..."
    )
    test_smiles, test_spectra = load_spectra(mgf, split, precursor_mass_mode)
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

    print(
        f"\nLoading SIMBA checkpoint: {checkpoint} (head_mode={head_mode}, "
        f"corn_corrected={corn_corrected})"
    )
    model = load_model(
        checkpoint,
        device,
        head_mode=head_mode,
        use_mces_bucket_head=corn_corrected,
        mces_bucket_use_mlp=mces_bucket_use_mlp,
        mces_bucket_use_product=mces_bucket_use_product,
        d_model=d_model,
        n_layers=n_layers,
    )

    print("\nEmbedding real test spectra ...")
    test_embs, test_embs_raw = embed_spectra(
        model, test_spectra, batch_size, device, return_raw=True
    )
    print("\nEmbedding ICEBERG-predicted candidate spectra ...")
    cand_embs, cand_embs_raw = embed_spectra(
        model, cand_spectra, batch_size, device, return_raw=True
    )

    if intermediates_dir:
        out_dir = Path(intermediates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(test_embs, out_dir / "test_embeddings.pt")
        torch.save(cand_embs, out_dir / "candidate_embeddings.pt")
        (out_dir / "test_smiles.json").write_text(json.dumps(test_smiles))
        (out_dir / "test_adducts.json").write_text(json.dumps(test_adducts))
        (out_dir / "candidate_smiles.json").write_text(json.dumps(cand_smiles))
        (out_dir / "candidate_adducts.json").write_text(
            json.dumps([s.adduct for s in cand_spectra])
        )
        print(f"Intermediates saved to {out_dir}/")

    print(
        f"\nRanking candidates ({'CORN-corrected' if corn_corrected else 'cosine'}) ..."
    )
    if corn_corrected:
        per_query_ranked = rank_candidates_corn_corrected(
            test_smiles,
            test_adducts,
            query_candidates,
            cand_smi_to_row,
            test_embs_raw,
            cand_embs_raw,
            model,
            device,
            top_k=20,
        )
    else:
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

    subset_results = None
    if min_peaks is not None:
        keep_idx = [i for i, s in enumerate(test_spectra) if len(s.mz) >= min_peaks]
        sub_smiles = [test_smiles[i] for i in keep_idx]
        sub_ranked = [per_query_ranked[i] for i in keep_idx]
        subset_results, n_no_cand_sub = compute_hit_rates_from_ranking(
            sub_smiles, sub_ranked
        )
        n_scored_sub = len(sub_smiles) - n_no_cand_sub
        print(
            f"\n=== SIMBA+ICEBERG retrieval ({split}, peaks>={min_peaks}, "
            f"n={n_scored_sub}/{len(sub_smiles)} queries) ==="
        )
        for k, v in subset_results.items():
            print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    if not skip_mces:
        print(f"\nLoading GT MCES lookup from {gt_mces_dir} ...")
        gt_lookup = load_gt_mces_lookup(gt_mces_dir)
        print(f"  {len(gt_lookup) // 2} unique (test, candidate) pairs with a GT value")

        print("\nComputing GT MCES between ranked candidates and the true molecule ...")
        mces_results = compute_mces_stats(test_smiles, per_query_ranked, gt_lookup)
        results.update(mces_results)
        print("\n=== GT MCES to true molecule (exact, threshold=20) ===")
        for k, v in mces_results.items():
            print(f"  {k}: {v}")

    if output_tsv:
        rows = [
            {
                "split": split,
                "model": Path(checkpoint).parent.name,
                "head_mode": head_mode,
                "corn_corrected": corn_corrected,
                "precursor_mass_mode": precursor_mass_mode,
                "peak_filter": "none",
                "n": n_scored,
                **results,
            }
        ]
        if subset_results is not None:
            rows.append(
                {
                    "split": split,
                    "model": Path(checkpoint).parent.name,
                    "head_mode": head_mode,
                    "corn_corrected": corn_corrected,
                    "precursor_mass_mode": precursor_mass_mode,
                    "peak_filter": f"peaks>={min_peaks}",
                    "n": n_scored_sub,
                    **subset_results,
                }
            )
        pd.DataFrame(rows).to_csv(output_tsv, sep="\t", index=False)
        print(f"\nSaved to {output_tsv}")

    return results, subset_results


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
        nargs="+",
        help="ICEBERG candidate TSV(s) (smiles/ionization/precursor) -- "
        "one path, or several (e.g. the original plus a delta file) "
        "to be concatenated, matched 1:1 by position with --iceberg_preds",
    )
    p.add_argument(
        "--iceberg_preds",
        required=True,
        nargs="+",
        help="ICEBERG predictions HDF5(s) -- one path, or several to merge",
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--output_tsv", default=None)
    p.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory to save embeddings and SMILES lists",
    )
    p.add_argument(
        "--gt_mces_dir",
        default=None,
        help=(
            "Dir with smiles.txt + combined mces_exact.npy from "
            "tools/prepare_gt_mces_retrieval.py (run its combine step first). "
            "Required unless --skip_mces."
        ),
    )
    p.add_argument(
        "--skip_mces",
        action="store_true",
        help="Skip GT MCES computation, report only hit@k",
    )
    p.add_argument(
        "--precursor_mass_mode",
        default="measured",
        choices=["measured", "theoretical"],
        help=(
            "'measured' reads the MGF's own PRECURSOR_MZ (historical default); "
            "'theoretical' recomputes it from the true SMILES+adduct instead, "
            "matching what every checkpoint from experiment 010 onward was "
            "trained with. ICEBERG candidate spectra are always theoretical "
            "already (hardcoded in build_candidate_tsv*.py), unaffected by "
            "this flag -- it only changes the real query spectra."
        ),
    )
    p.add_argument(
        "--corn_corrected",
        action="store_true",
        help=(
            "Rank by CORN-corrected MCES (pairwise bucket-head forward pass) "
            "instead of plain embedding cosine similarity. Requires a "
            "checkpoint trained with model.tasks.mces_bucket.enabled=true "
            "(013, 014_1-4) -- pass --mces_bucket_use_mlp/--mces_bucket_use_product "
            "matching that checkpoint's own training config."
        ),
    )
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
        "--min_peaks",
        type=int,
        default=None,
        help=(
            "Additionally report hit@k restricted to test spectra with >= "
            "this many peaks (SIMBA's own canonical filter, min_n_peaks=6 in "
            "simba/configs/data/default.yaml and all data-prep scripts) "
            "alongside the unrestricted full-test-set numbers."
        ),
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
