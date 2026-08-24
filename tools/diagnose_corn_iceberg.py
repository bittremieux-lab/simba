"""
One-off diagnostic for the new CORN-corrected ICEBERG+SIMBA retrieval path
(tools/simba_retrieval_iceberg.py's rank_candidates_corn_corrected).

Answers two questions raised after the first real run came back with a much
bigger gap to ICEBERG+Cosine than 014_2's own held-out validation loss would
suggest:

1. Is the plain embedding-cosine signal (head_mode=cosine_no_head, no bucket
   correction at all -- the exact same code path used since checkpoint 005)
   ALSO this weak on ICEBERG-predicted candidates, or does the bucket-head
   correction specifically make things worse? Computed from the SAME
   embeddings in one pass, so it's a direct apples-to-apples comparison, not
   a second noisy run.
2. Concrete per-pair arithmetic for a handful of real queries: cosine
   similarity -> raw pred_mces -> bucket_pred -> corrected -> final score,
   printed for manual verification against the formulas in
   rank_candidates_corn_corrected / _corn_corrected_mces.

Usage:
    uv run python tools/diagnose_corn_iceberg.py \\
        --checkpoint /path/to/014_2/checkpoint.ckpt \\
        --mgf /path/to/gaetan_test.mgf \\
        --candidates /path/to/MassSpecGym_retrieval_candidates_formula.json \\
        --candidate_tsv /path/to/a.tsv /path/to/b.tsv \\
        --iceberg_preds /path/to/a.hdf5 /path/to/b.hdf5 \\
        --n_diagnose 5
"""

import argparse
import json

import numpy as np
import torch
from simba_retrieval import canonicalize, embed_spectra, load_model, load_spectra
from simba_retrieval_iceberg import (
    CORN_BUCKET_EDGES,
    _corn_corrected_mces,
    _corn_corrected_ranking_score,
    _corn_decode_bucket,
    build_candidate_spectra,
    compute_hit_rates_from_ranking,
    load_candidate_index,
    load_iceberg_spectra,
)


def rank_plain_cosine(
    test_smiles, test_adducts, query_candidates, cand_smi_to_row, test_embs, cand_embs
):
    """Exact copy of rank_candidates' logic (top_k=20), inlined here so this
    script has no dependency on anything except the read-only loading utils
    -- avoids any risk of accidentally exercising the new corn-corrected
    code path when the point is to isolate it."""
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
        per_query.append([cand_smis[j] for j in order.tolist()][:20])
    return per_query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--candidate_tsv", required=True, nargs="+")
    p.add_argument("--iceberg_preds", required=True, nargs="+")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_diagnose", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_smiles, test_spectra = load_spectra(args.mgf, args.split, "theoretical")
    test_adducts = [s.adduct for s in test_spectra]
    print(f"{len(test_smiles)} real test spectra")

    with open(args.candidates) as fh:
        candidate_json = json.load(fh)
    query_candidates = {canonicalize(k): v for k, v in candidate_json.items()}

    cand_index = load_candidate_index(args.candidate_tsv)
    all_cand_ids = cand_index["spec"].tolist()
    iceberg_specs = load_iceberg_spectra(args.iceberg_preds, all_cand_ids)
    cand_smiles, cand_spectra = build_candidate_spectra(cand_index, iceberg_specs)
    cand_smi_to_row = {
        (smi, spec.adduct): row_idx
        for row_idx, (smi, spec) in enumerate(zip(cand_smiles, cand_spectra))
    }
    print(f"{len(cand_spectra)} candidate spectra ready")

    model = load_model(
        args.checkpoint,
        device,
        head_mode="cosine_no_head",
        use_mces_bucket_head=True,
        mces_bucket_use_mlp=True,
    )

    print("Embedding real test spectra ...")
    test_embs, test_embs_raw = embed_spectra(
        model, test_spectra, args.batch_size, device, return_raw=True
    )
    print("Embedding ICEBERG-predicted candidate spectra ...")
    cand_embs, cand_embs_raw = embed_spectra(
        model, cand_spectra, args.batch_size, device, return_raw=True
    )

    print("\n=== Ranking by PLAIN cosine similarity (no bucket correction) ===")
    plain_ranked = rank_plain_cosine(
        test_smiles,
        test_adducts,
        query_candidates,
        cand_smi_to_row,
        test_embs,
        cand_embs,
    )
    plain_results, n_no_cand = compute_hit_rates_from_ranking(test_smiles, plain_ranked)
    print(f"n_no_candidates={n_no_cand}")
    for k, v in plain_results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")

    # Sanity check: are cosine similarities even discriminative, or collapsed
    # to a narrow band (which would explain weak ranking regardless of the
    # bucket head)?
    sample_sims = []
    for i in range(min(200, len(test_smiles))):
        q_smi, q_adduct = test_smiles[i], test_adducts[i]
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs = [
            cand_smi_to_row[(c, q_adduct)]
            for c in cand_list
            if (c, q_adduct) in cand_smi_to_row
        ]
        if not row_idxs:
            continue
        sims = (test_embs[i : i + 1] @ cand_embs[row_idxs].T).squeeze(0)
        sample_sims.append(sims)
    all_sims = torch.cat(sample_sims)
    print(
        f"\nCosine-sim distribution over 200 queries' candidate pools: "
        f"min={all_sims.min():.4f} max={all_sims.max():.4f} "
        f"mean={all_sims.mean():.4f} std={all_sims.std():.4f}"
    )

    print(f"\n=== Per-pair diagnostics for {args.n_diagnose} example queries ===")
    print(f"CORN_BUCKET_EDGES = {CORN_BUCKET_EDGES.tolist()}")
    shown = 0
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        if shown >= args.n_diagnose:
            break
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if len(row_idxs) < 3:
            continue
        shown += 1

        emb0 = test_embs_raw[i : i + 1].expand(len(row_idxs), -1).to(device)
        emb1 = cand_embs_raw[row_idxs].to(device)
        with torch.no_grad():
            _, emb_sim_2, emb_sim_3 = model.compute_from_embeddings(emb0, emb1)
        pred_mces = ((1.0 - emb_sim_2) * model.mces_max_value).cpu().numpy()
        bucket_pred = _corn_decode_bucket(emb_sim_3).cpu().numpy()
        corrected = _corn_corrected_mces(pred_mces, bucket_pred)
        score = _corn_corrected_ranking_score(pred_mces, bucket_pred)

        q_canon = canonicalize(q_smi)
        order = np.argsort(score)
        true_rank_corn = next(
            (r for r, j in enumerate(order) if canonicalize(cand_smis[j]) == q_canon),
            None,
        )
        plain_sims = (test_embs[i : i + 1] @ cand_embs[row_idxs].T).squeeze(0)
        plain_order = torch.argsort(plain_sims, descending=True).tolist()
        true_rank_plain = next(
            (
                r
                for r, j in enumerate(plain_order)
                if canonicalize(cand_smis[j]) == q_canon
            ),
            None,
        )

        print(
            f"\n--- Query {i} ({len(row_idxs)} candidates), true molecule rank: "
            f"plain-cosine={true_rank_plain}, corn-corrected={true_rank_corn} ---"
        )
        print(
            f"{'cand#':>6} {'is_true':>8} {'cos_sim':>8} {'pred_mces':>10} "
            f"{'bucket':>7} {'corrected':>10} {'score':>12}"
        )
        top_n = order[:8]
        for j in top_n:
            is_true = canonicalize(cand_smis[j]) == q_canon
            print(
                f"{j:>6} {str(is_true):>8} {emb_sim_2[j].item():>8.4f} "
                f"{pred_mces[j]:>10.3f} {bucket_pred[j]:>7d} "
                f"{corrected[j]:>10.3f} {score[j]:>12.3f}"
            )
        if true_rank_corn is not None and true_rank_corn >= 8:
            j = order[true_rank_corn]
            print(
                f"  [true molecule, rank {true_rank_corn}] cand#{j} "
                f"cos_sim={emb_sim_2[j].item():.4f} pred_mces={pred_mces[j]:.3f} "
                f"bucket={bucket_pred[j]} corrected={corrected[j]:.3f} "
                f"score={score[j]:.3f}"
            )


if __name__ == "__main__":
    main()
