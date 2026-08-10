"""3e: top-1 diagnostics for SIMBA+ICEBERG retrieval, three subplots in one figure.

(a) SIMBA-predicted MCES for the TRUE candidate pair — test spectrum vs the
    SAME molecule's ICEBERG-predicted spectrum. GT MCES is 0 by definition
    for every one of these (never computed/stored, see
    prepare_gt_mces_retrieval.py); this shows what SIMBA itself predicts for
    that pair regardless of whether it ends up ranked #1. If SIMBA
    recognized the true match well, this should cluster near 0.

(b) For the queries where SIMBA's top-1 pick is wrong: the distribution of
    (GT MCES - SIMBA-predicted MCES) for that specific wrongly-picked
    candidate. Since SIMBA ranked it #1, it predicted a low MCES/high
    similarity for it — if GT is actually much higher (as it should be for a
    wrong pick), this skews positive: SIMBA is systematically
    underestimating distance for its own wrong top choice, not just noisily
    off in both directions.

(c) 2D heatmap (count-colored, log scale) of (a)'s value against (b)'s value,
    paired per miss case — is a badly-predicted true candidate (large panel
    (a) value) associated with a badly-miscalibrated wrong pick (large panel
    (b) value) for the SAME query, or are the two failure modes independent?

The query is each individual test SPECTRUM (not one averaged embedding per
test molecule) — see ood_generalization_check.py's module docstring for why
that matters (averaging first inflates hit@1 by denoising the query side in
a way the real evaluation never benefits from). Each candidate is likewise
matched to the query's own adduct specifically, never averaged across the
candidate's other adducts (cand_smi_adduct_to_emb is keyed by (smiles,
adduct); a candidate missing an embedding for the query's adduct is dropped,
same as ood_generalization_check.py's expand_and_score_ragged). Each query's
candidate pool + GT values come directly from mces_exact.npy's own
(query_idx, cand_idx, gt) rows grouped by the underlying molecule (shared
across all of that molecule's spectra) — no need to reload the raw (400MB)
candidates JSON. Ranking is over {true candidate (added back in, GT=0) + all
pool candidates}, matching what real retrieval evaluation ranks over.

Usage:
    uv run python tools/mces_top1_diagnostics.py \\
        --intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --output_dir /path/to/output \\
        --mces_max_value 40
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from ood_generalization_check import build_candidate_embeddings_by_smi_adduct
from simba_retrieval import canonicalize


def compute_top1_diagnostics(
    test_smiles_raw: list[str],
    test_adducts_raw: list[str],
    test_embeddings: torch.Tensor,
    cand_smi_adduct_to_emb: dict[tuple[str, str], np.ndarray],
    gt_pairs: np.ndarray,
    idx_to_smiles: list[str],
    mces_max_value: float,
) -> dict:
    query_pool: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in gt_pairs:
        a, b, gt = int(row[0]), int(row[1]), float(row[-1])
        query_pool[a].append((b, gt))

    smi_to_molidx = {smi: i for i, smi in enumerate(idx_to_smiles)}
    test_emb_norm = torch.nn.functional.normalize(test_embeddings, p=2, dim=-1).numpy()

    self_preds = []
    wrong_top1_diffs = []
    miss_self_preds = []  # paired 1:1 with wrong_top1_diffs, for the (c) heatmap
    n_hit = 0
    n_miss = 0
    n_no_self = 0
    n_no_pool = 0

    for spec_i, (smi, q_adduct) in enumerate(zip(test_smiles_raw, test_adducts_raw)):
        q_canon = canonicalize(smi)
        q_idx = smi_to_molidx.get(q_canon)
        if q_idx is None:
            continue
        pool = query_pool.get(q_idx)
        if not pool:
            n_no_pool += 1
            continue

        emb_self = cand_smi_adduct_to_emb.get((q_canon, q_adduct))
        if emb_self is None:
            n_no_self += 1
            continue

        emb_q = test_emb_norm[spec_i]
        sim_self = float(np.dot(emb_q, emb_self))
        pred_self = max(0.0, mces_max_value * (1.0 - sim_self))
        self_preds.append(pred_self)

        cand_embs = [emb_self]
        cand_gts = [0.0]
        cand_is_self = [True]
        for cand_idx, gt in pool:
            emb_c = cand_smi_adduct_to_emb.get((idx_to_smiles[cand_idx], q_adduct))
            if emb_c is None:
                continue
            cand_embs.append(emb_c)
            cand_gts.append(gt)
            cand_is_self.append(False)

        mat = np.stack(cand_embs)
        sims = mat @ emb_q
        best = int(np.argmax(sims))
        if cand_is_self[best]:
            n_hit += 1
        else:
            n_miss += 1
            pred_best = max(0.0, mces_max_value * (1.0 - sims[best]))
            wrong_top1_diffs.append(cand_gts[best] - pred_best)
            miss_self_preds.append(pred_self)

    return {
        "self_preds": np.array(self_preds),
        "wrong_top1_diffs": np.array(wrong_top1_diffs),
        "miss_self_preds": np.array(miss_self_preds),
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_no_self": n_no_self,
        "n_no_pool": n_no_pool,
    }


def plot_top1_diagnostics(result: dict, title_prefix: str, out_path: Path) -> None:
    self_preds = result["self_preds"]
    diffs = result["wrong_top1_diffs"]
    miss_self_preds = result["miss_self_preds"]
    n_hit, n_miss = result["n_hit"], result["n_miss"]
    hit_rate = n_hit / (n_hit + n_miss) * 100 if (n_hit + n_miss) else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_a, ax_b = axes[0]
    ax_c_log, ax_c_lin = axes[1]

    ax_a.hist(self_preds, bins=60, color="tab:blue", alpha=0.8)
    ax_a.axvline(
        self_preds.mean(),
        color="black",
        linestyle="--",
        label=f"mean={self_preds.mean():.2f}",
    )
    ax_a.set_xlabel("SIMBA-predicted MCES for the TRUE candidate (GT=0)")
    ax_a.set_ylabel("Count")
    ax_a.set_title(f"(a) True-candidate prediction (n={len(self_preds):,})")
    ax_a.legend()

    ax_b.hist(diffs, bins=60, color="tab:orange", alpha=0.8)
    ax_b.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax_b.axvline(
        diffs.mean(), color="black", linestyle="--", label=f"mean={diffs.mean():.2f}"
    )
    ax_b.set_xlabel("GT MCES - SIMBA-predicted MCES (wrong top-1 pick)")
    ax_b.set_title(f"(b) Wrong top-1 error (n={len(diffs):,}, hit@1={hit_rate:.1f}%)")
    ax_b.legend()

    h_log = ax_c_log.hist2d(
        miss_self_preds, diffs, bins=40, cmap="viridis", norm=mcolors.LogNorm()
    )
    cb_log = fig.colorbar(h_log[3], ax=ax_c_log)
    cb_log.set_label("count (log scale)")
    ax_c_log.set_xlabel("(a) SIMBA-predicted MCES, true candidate")
    ax_c_log.set_ylabel("(b) GT - SIMBA-predicted, wrong top-1")
    ax_c_log.set_title(f"(c) Joint distribution, log color (n={len(diffs):,})")

    h_lin = ax_c_lin.hist2d(miss_self_preds, diffs, bins=40, cmap="viridis")
    cb_lin = fig.colorbar(h_lin[3], ax=ax_c_lin)
    cb_lin.set_label("count (linear scale)")
    ax_c_lin.set_xlabel("(a) SIMBA-predicted MCES, true candidate")
    ax_c_lin.set_ylabel("(b) GT - SIMBA-predicted, wrong top-1")
    ax_c_lin.set_title(f"(d) Joint distribution, linear color (n={len(diffs):,})")

    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved {out_path}  hit@1={hit_rate:.1f}% (n_hit={n_hit}, n_miss={n_miss}, "
        f"n_no_self={result['n_no_self']}, n_no_pool={result['n_no_pool']}); "
        f"self_pred mean={self_preds.mean():.2f}; wrong_top1_diff mean={diffs.mean():.2f}"
    )


def run(
    intermediates_dir: str,
    gt_mces_dir: str,
    output_dir: str,
    mces_max_value: float = 40.0,
) -> None:
    inter = Path(intermediates_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved embeddings + SMILES + adducts ...")
    test_embeddings = torch.load(inter / "test_embeddings.pt", map_location="cpu")
    test_smiles = json.loads((inter / "test_smiles.json").read_text())
    test_adducts = json.loads((inter / "test_adducts.json").read_text())
    candidate_embeddings = torch.load(
        inter / "candidate_embeddings.pt", map_location="cpu"
    )
    candidate_smiles = json.loads((inter / "candidate_smiles.json").read_text())
    candidate_adducts = json.loads((inter / "candidate_adducts.json").read_text())

    cand_smi_adduct_to_emb = build_candidate_embeddings_by_smi_adduct(
        candidate_smiles, candidate_adducts, candidate_embeddings
    )

    gt_dir = Path(gt_mces_dir)
    idx_to_smiles = gt_dir.joinpath("smiles.txt").read_text().splitlines()
    gt_pairs = np.load(gt_dir / "mces_exact.npy")
    valid = (gt_pairs[:, 2] >= 0) & ~np.isnan(gt_pairs[:, 2])
    gt_pairs = gt_pairs[valid]
    print(
        f"  {len(gt_pairs):,} GT pairs, {len(test_smiles)} test spectra, "
        f"{len(cand_smi_adduct_to_emb)} (candidate molecule, adduct) embeddings"
    )

    print(
        "Computing top-1 diagnostics (per test spectrum, own-adduct candidate match) ..."
    )
    result = compute_top1_diagnostics(
        test_smiles,
        test_adducts,
        test_embeddings,
        cand_smi_adduct_to_emb,
        gt_pairs,
        idx_to_smiles,
        mces_max_value,
    )
    plot_top1_diagnostics(
        result,
        "Test-to-candidate: top-1 diagnostics",
        out_dir / "test_to_candidate_top1_diagnostics.png",
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--intermediates_dir", required=True)
    p.add_argument("--gt_mces_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mces_max_value", type=float, default=40.0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
