"""014_2 analog discovery (see NOTES_014_2_ANALOG_DISCOVERY.md), stage 4:
combine stage 2's per-method score matrices with stage 3's exhaustive exact
GT-MCES matrix into the paper's three Figure-2-style panels, for ONE search
(A or B):

  (a) boxplot of best TRUE MCES among each query's top-10 inferred candidates
      (the paper's "retrieved analog" procedure, following Huber et al.)
  (b) ROC curve discriminating true analogs from unrelated molecules, pooled
      over every (query, library) pair with a resolved GT MCES -- NOT just
      top-10, which is the whole reason stage 3 computes the GT matrix
      exhaustively rather than only for each method's own top-10 (three
      different methods have three different top-10s; an exhaustive common
      GT source is what makes the three curves comparable and lets every
      pair vote, not just the ones a given method already ranked high)
  (c) ranking-performance: cumulative fraction of queries whose single
      TRUE best analog (lowest GT MCES in the whole library) is retrieved
      within the model's own top-K, for K = 1..20

Two paper simplifications carried over from NOTES_014_2_ANALOG_DISCOVERY.md
(agreed with the user before building this pipeline): SIMBA vs plain cosine
only (no modified-cosine/Spec2Vec/MS2DeepScore/DreaMS), and RAW MCES
throughout rather than the paper's molecular-size-normalized MCES -- so the
paper's normalized-MCES<0.3 "true analog" cutoff doesn't directly transfer.
--roc_threshold (default 4.0) is a raw-MCES stand-in, chosen to match this
project's own already-trained CORN bucket boundary (bucket edges [2,4,6,8],
see 014_2's training config) rather than an arbitrary pick -- pass a
different value to see the ROC curve at another cutoff; it does not require
recomputing anything upstream since the underlying GT matrix is continuous.

GT MCES values above the exact-MCES computation's threshold=20 are
LOWER BOUNDS, not exact distances (myopic_mces's own documented early-
rejection behavior -- see analog_discovery_exact_mces.py's docstring). This
only matters for very close cutoffs; a lower bound > roc_threshold still
conclusively means "not a true analog" at any threshold <= 20, which covers
every roc_threshold this script would sensibly be run with.

Usage:
    uv run python tools/analog_discovery_analyze.py \\
        --scores_dir data/analog_discovery/search_A_scores \\
        --exact_mces_dir data/analog_discovery/search_A_exact_mces \\
        --output_dir data/analog_discovery/search_A_results \\
        --search_label "Search A (NIST20 + MassSpecGym)"
"""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


METHODS = ["simba_raw", "simba_corn", "cosine"]
METHOD_LABELS = {
    "simba_raw": "SIMBA (raw MCES)",
    "simba_corn": "SIMBA (CORN-corrected)",
    "cosine": "Cosine (plain)",
}
METHOD_COLORS = {
    "simba_raw": "tab:blue",
    "simba_corn": "tab:orange",
    "cosine": "tab:green",
}


def load_scores(scores_dir: Path) -> tuple[list, list, dict]:
    q_smiles = (scores_dir / "smiles_query.txt").read_text().splitlines()
    lib_smiles = (scores_dir / "smiles_library.txt").read_text().splitlines()
    scores = {m: np.load(scores_dir / f"score_{m}.npy") for m in METHODS}
    for m, arr in scores.items():
        assert arr.shape == (len(q_smiles), len(lib_smiles)), (
            f"score_{m}.npy shape {arr.shape} != ({len(q_smiles)}, {len(lib_smiles)})"
        )
    return q_smiles, lib_smiles, scores


def build_gt_matrix(
    q_smiles: list, lib_smiles: list, exact_mces_dir: Path
) -> np.ndarray:
    """(n_query, n_library) GT-MCES matrix, NaN where unresolved (solver
    failure or, in principle, a pair analog_discovery_exact_mces.py's
    `prepare` never generated -- shouldn't happen here since it builds the
    full query x library cross product by construction)."""
    exact_smiles = (exact_mces_dir / "smiles.txt").read_text().splitlines()
    arr = np.load(exact_mces_dir / "mces_exact.npy")

    q_pos = {s: i for i, s in enumerate(q_smiles)}
    lib_pos = {s: i for i, s in enumerate(lib_smiles)}
    to_q_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    to_lib_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    for i, s in enumerate(exact_smiles):
        if s in q_pos:
            to_q_idx[i] = q_pos[s]
        if s in lib_pos:
            to_lib_idx[i] = lib_pos[s]

    idx_a = arr[:, 0].astype(np.int64)
    idx_b = arr[:, 1].astype(np.int64)
    mces = arr[:, 2]

    # By analog_discovery_exact_mces.py's own `prepare` construction, column
    # 0 is always the query-side molecule and column 1 the library-side --
    # not swapped, so no need to also try the reverse mapping.
    row = to_q_idx[idx_a]
    col = to_lib_idx[idx_b]

    resolved = (mces != -1.0) & ~np.isnan(mces)
    mapped = (row >= 0) & (col >= 0)
    n_unmapped = int((resolved & ~mapped).sum())
    if n_unmapped:
        print(
            f"  WARNING: {n_unmapped} resolved GT pairs could not be mapped to the "
            "score matrix's query/library SMILES -- canonical-SMILES mismatch "
            "between analog_discovery_exact_mces.py and analog_discovery_embed_rank.py "
            "runs? Excluded from analysis."
        )

    keep = resolved & mapped
    gt = np.full((len(q_smiles), len(lib_smiles)), np.nan, dtype=np.float32)
    gt[row[keep], col[keep]] = mces[keep]

    n_total = gt.size
    n_resolved = int(keep.sum())
    print(
        f"  GT matrix: {n_resolved:,} / {n_total:,} query-library pairs resolved ({100 * n_resolved / n_total:.1f}%)"
    )
    return gt


def panel_boxplot(scores: dict, gt: np.ndarray, top_k: int, output_dir: Path) -> dict:
    n_q = gt.shape[0]
    per_method = {}
    summary = {}
    for m in METHODS:
        best_mces = []
        n_no_gt = 0
        for i in range(n_q):
            top = np.argsort(scores[m][i])[:top_k]
            vals = gt[i, top]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                n_no_gt += 1
                continue
            best_mces.append(float(vals.min()))
        per_method[m] = best_mces
        summary[m] = {
            "n": len(best_mces),
            "n_excluded_no_gt": n_no_gt,
            "mean": float(np.mean(best_mces)) if best_mces else float("nan"),
            "median": float(np.median(best_mces)) if best_mces else float("nan"),
        }
        print(
            f"  [{m}] best-in-top{top_k} MCES: n={len(best_mces)} "
            f"mean={summary[m]['mean']:.2f} median={summary[m]['median']:.2f} "
            f"({n_no_gt} queries excluded, no resolved GT in top{top_k})"
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [per_method[m] for m in METHODS]
    ax.boxplot(data, tick_labels=[METHOD_LABELS[m] for m in METHODS], showmeans=True)
    ax.set_ylabel(f"Best true MCES among top-{top_k} candidates")
    ax.set_title("Analog discovery: retrieved-analog quality")
    fig.tight_layout()
    fig.savefig(output_dir / "analog_discovery_boxplot.png", dpi=150)
    plt.close(fig)
    return summary


def panel_roc(
    scores: dict, gt: np.ndarray, roc_threshold: float, output_dir: Path
) -> dict:
    valid = ~np.isnan(gt)
    labels = (gt[valid] < roc_threshold).astype(np.int32)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    print(
        f"  ROC pool: {valid.sum():,} pairs, {n_pos:,} true analogs (<{roc_threshold}), {n_neg:,} unrelated"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    summary = {}
    for m in METHODS:
        pred = -scores[m][
            valid
        ]  # higher = closer/more-likely-analog, matching roc_curve's convention
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        summary[m] = {"auc": float(roc_auc)}
        ax.plot(
            fpr,
            tpr,
            label=f"{METHOD_LABELS[m]} (AUC={roc_auc:.3f})",
            color=METHOD_COLORS[m],
        )
        print(f"  [{m}] AUC={roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Analog discovery ROC (true analog: MCES < {roc_threshold})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "analog_discovery_roc.png", dpi=150)
    plt.close(fig)
    return summary


def panel_ranking_performance(
    scores: dict, gt: np.ndarray, max_k: int, output_dir: Path
) -> dict:
    n_q, n_lib = gt.shape
    ks = np.arange(1, max_k + 1)

    ranks_by_method = {m: [] for m in METHODS}
    n_no_best_analog = 0
    for i in range(n_q):
        row = gt[i]
        if np.all(np.isnan(row)):
            n_no_best_analog += 1
            continue
        best_j = int(np.nanargmin(row))
        for m in METHODS:
            order = np.argsort(scores[m][i])
            rank = int(np.nonzero(order == best_j)[0][0]) + 1  # 1-indexed
            ranks_by_method[m].append(rank)

    print(
        f"  {n_no_best_analog} / {n_q} queries excluded (no resolved GT anywhere in library)"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    summary = {}
    for m in METHODS:
        ranks = np.asarray(ranks_by_method[m])
        cum_frac = [(ranks <= k).mean() for k in ks]
        summary[m] = {
            "cum_frac_at_1": float(cum_frac[0]),
            "cum_frac_at_10": float(cum_frac[min(9, len(cum_frac) - 1)]),
        }
        ax.plot(ks, cum_frac, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        print(
            f"  [{m}] hit@1={cum_frac[0]:.3f}  hit@10={summary[m]['cum_frac_at_10']:.3f}"
        )

    ax.set_xlabel("K")
    ax.set_ylabel("Fraction of queries with true best analog in top-K")
    ax.set_title("Analog discovery ranking performance")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "analog_discovery_ranking_performance.png", dpi=150)
    plt.close(fig)
    return summary


def run(
    scores_dir: str,
    exact_mces_dir: str,
    output_dir: str,
    search_label: str = "",
    top_k: int = 10,
    roc_threshold: float = 4.0,
    max_rank_k: int = 20,
):
    scores_dir, exact_mces_dir, output_dir = (
        Path(scores_dir),
        Path(exact_mces_dir),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {scores_dir} ...")
    q_smiles, lib_smiles, scores = load_scores(scores_dir)
    print(f"  {len(q_smiles)} queries, {len(lib_smiles)} library molecules")

    print(f"Loading exact GT MCES from {exact_mces_dir} ...")
    gt = build_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)

    print(f"\n(a) Boxplot (top-{top_k}) ...")
    boxplot_summary = panel_boxplot(scores, gt, top_k, output_dir)

    print(f"\n(b) ROC (threshold={roc_threshold}) ...")
    roc_summary = panel_roc(scores, gt, roc_threshold, output_dir)

    print(f"\n(c) Ranking performance (K=1..{max_rank_k}) ...")
    ranking_summary = panel_ranking_performance(scores, gt, max_rank_k, output_dir)

    summary = {
        "search_label": search_label,
        "n_query": len(q_smiles),
        "n_library": len(lib_smiles),
        "top_k": top_k,
        "roc_threshold": roc_threshold,
        "boxplot": boxplot_summary,
        "roc": roc_summary,
        "ranking_performance": ranking_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote plots + summary.json to {output_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--scores_dir", required=True, help="Output of analog_discovery_embed_rank.py"
    )
    p.add_argument(
        "--exact_mces_dir",
        required=True,
        help="Output of analog_discovery_exact_mces.py (after combine)",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--search_label", default="")
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K for the boxplot panel (paper uses 10)",
    )
    p.add_argument(
        "--roc_threshold",
        type=float,
        default=4.0,
        help="Raw-MCES 'true analog' cutoff for the ROC panel",
    )
    p.add_argument(
        "--max_rank_k",
        type=int,
        default=20,
        help="Max K for the ranking-performance panel",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
