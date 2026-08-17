"""8b: three checks against build_retrieval_comparison_table.py's CSV, all
computed from the table alone (no re-scoring, no SLURM needed -- everything
here is a plain in-memory column computation).

1. Confusion matrix (SIMBA top-1 correct x cosine top-1 correct) + hit@k
   double-check against the already-committed retrieval_results.tsv values,
   and a check of how often cosine's top-1 "win" is really just an
   arbitrary tie among a candidate pool that's entirely flat at ~0
   similarity (no real discrimination at all).

   cosine "correct" here (and in every check below, since they all share
   the same cosine_hit1 flag) EXCLUDES these floor-tie wins -- confirmed
   directly that literally 100% of them have the true candidate listed
   FIRST in the candidate JSON, so the "win" is arbitrary list-order luck,
   not real discrimination; counted as a cosine ERROR instead. This moves
   cosine's table-derived hit@1 from the raw/committed 37.59% down to
   ~31.89% -- report_zero_cosine_hits still reports the RAW
   (uncorrected) numbers via cosine_hit1_raw, since its whole point is
   diagnosing this exact phenomenon.

2. Retrieval difficulty vs real-spectrum peak count (n_peaks_test). Only
   n_peaks_test is plotted -- n_peaks_candidate (the true candidate's raw
   ICEBERG peak count) is degenerate, always exactly 100 for every
   candidate, because ICEBERG was run with --sparse-k 100 --threshold 0.0,
   so it always emits exactly 100 nonzero-mass fragments regardless of
   molecule size. A 2D heatmap against it collapses to one row; this is
   reported (not plotted) as a sanity check instead.

3. Precursor-mass discrepancy (a test spectrum's own measured PRECURSOR_MZ
   vs its true candidate's formula-implied calculated m/z) across the 4
   confusion-matrix cells, as a boxplot -- full population per cell (all
   17,555 test spectra split 4 ways), not a subsample. This can't be why
   SIMBA picks the wrong candidate WITHIN a pool -- every candidate in a
   formula-matched pool shares the same calculated precursor (checked
   directly: the candidate SIMBA actually top-ranks always matches the true
   candidate's calculated precursor to within 0.01 ppm, in every case
   checked) -- but a query's own measured-vs-theoretical gap could still
   correlate with overall difficulty, which is what this checks. Reads
   test_precursor_mz/candidate_precursor_mz directly off the table (added by
   tools/add_precursor_columns.py) -- no MGF scan or candidate_tsv lookup
   here anymore, just a vectorized column computation.

Usage:
    uv run python tools/plot_retrieval_comparison_checks.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --simba_retrieval_results_tsv /path/to/008_2_.../retrieval_iceberg/retrieval_results.tsv \\
        --cosine_retrieval_results_tsv /path/to/cosine_baseline_iceberg/retrieval_results.tsv \\
        --output_dir /path/to/output
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KS = (1, 5, 20)

CELLS = [
    ("both_correct", "SIMBA correct\ncosine correct"),
    ("simba_only", "SIMBA correct\ncosine wrong"),
    ("cosine_only", "SIMBA wrong\ncosine correct"),
    ("neither", "SIMBA wrong\ncosine wrong"),
]


def load_true_candidate_rows(table_csv: str, extra_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        table_csv,
        usecols=[
            "test_spec_idx",
            "test_smiles",
            "test_adduct",
            "candidate_smiles",
            "simba_rank",
            "cosine_rank",
            "cosine_similarity",
        ]
        + [c for c in extra_cols if c != "cosine_similarity"],
    )
    correct = df[df["test_smiles"] == df["candidate_smiles"]].copy()
    correct["simba_hit1"] = correct["simba_rank"] == 1
    # cosine_hit1_raw: the literal rank==1 flag, used only for
    # report_zero_cosine_hits' own diagnostic of this exact phenomenon.
    # cosine_hit1 (used everywhere else -- confusion matrix, heatmaps,
    # n_peaks/precursor checks): a rank==1 "win" where the ENTIRE candidate
    # pool sits at exactly 0 similarity is arbitrary list-order luck, not
    # real discrimination (confirmed directly: literally 100% of these have
    # the true candidate listed first in the candidate JSON) -- counted as a
    # cosine ERROR here, not a hit.
    correct["cosine_hit1_raw"] = correct["cosine_rank"] == 1
    correct["cosine_hit1"] = correct["cosine_hit1_raw"] & (
        correct["cosine_similarity"].abs() > 1e-9
    )
    return correct


# --- 1. confusion matrix + hit@k double-check ------------------------------


def load_known_hit_rates(tsv_path: str) -> dict[int, float]:
    df = pd.read_csv(tsv_path, sep="\t")
    row = df.iloc[0]
    return {k: float(row[f"hit@{k}"]) * 100 for k in KS}


def run_confusion_and_hitk(
    correct: pd.DataFrame,
    simba_retrieval_results_tsv: str,
    cosine_retrieval_results_tsv: str,
    out_dir: Path,
) -> None:
    n = len(correct)
    both = int((correct["simba_hit1"] & correct["cosine_hit1"]).sum())
    simba_only = int((correct["simba_hit1"] & ~correct["cosine_hit1"]).sum())
    cosine_only = int((~correct["simba_hit1"] & correct["cosine_hit1"]).sum())
    neither = int((~correct["simba_hit1"] & ~correct["cosine_hit1"]).sum())
    assert both + simba_only + cosine_only + neither == n

    print("\n=== Confusion matrix: SIMBA top-1 correct x cosine top-1 correct ===")
    print(
        f"  n = {n:,} test spectra (with a resolvable true-candidate rank) -- "
        "cosine 'correct' EXCLUDES floor-tie wins (cosine_similarity==0 for the "
        "whole pool, i.e. arbitrary list-order luck, not real discrimination)"
    )
    print("                      cosine correct   cosine wrong")
    print(f"  SIMBA correct       {both:>10,}      {simba_only:>10,}")
    print(f"  SIMBA wrong         {cosine_only:>10,}      {neither:>10,}")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    mat = np.array([[both, simba_only], [cosine_only, neither]])
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cosine correct", "cosine wrong"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SIMBA correct", "SIMBA wrong"])
    for i in range(2):
        for j in range(2):
            pct = 100 * mat[i, j] / n
            ax.text(
                j,
                i,
                f"{mat[i, j]:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if mat[i, j] > mat.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, label="count")
    ax.set_title(
        f"Top-1 correctness: SIMBA vs cosine (n={n:,})\n"
        "cosine floor-ties (whole pool at sim=0) counted as errors, not hits"
    )
    fig.tight_layout()
    out_path = out_dir / "confusion_matrix_simba_vs_cosine_top1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    table_hits = {}
    for method, rank_col in [("simba", "simba_rank"), ("cosine", "cosine_rank")]:
        table_hits[method] = {k: 100 * (correct[rank_col] <= k).mean() for k in KS}
    # hit@1 specifically uses the floor-tie-corrected flags (cosine's in
    # particular -- simba_hit1 is identical to simba_rank==1, no correction
    # needed there). hit@5/20 are left as raw rank-based -- the floor-tie
    # correction is a hit@1/confusion-matrix concept here, not extended to
    # wider k.
    table_hits["simba"][1] = 100 * correct["simba_hit1"].mean()
    table_hits["cosine"][1] = 100 * correct["cosine_hit1"].mean()
    known_hits = {
        "simba": load_known_hit_rates(simba_retrieval_results_tsv),
        "cosine": load_known_hit_rates(cosine_retrieval_results_tsv),
    }

    print("\n=== hit@k: table-derived vs already-committed retrieval_results.tsv ===")
    print(
        "  (hit@1 intentionally diverges for cosine -- 'table' excludes floor-tie "
        "wins, 'known' is the raw committed number)"
    )
    for method in ("simba", "cosine"):
        print(f"  {method}:")
        for k in KS:
            print(
                f"    hit@{k}: table={table_hits[method][k]:.2f}%  "
                f"known={known_hits[method][k]:.2f}%"
            )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(KS))
    width = 0.2
    bars = [
        ("SIMBA (table)", [table_hits["simba"][k] for k in KS], "tab:blue", -1.5),
        ("SIMBA (known)", [known_hits["simba"][k] for k in KS], "tab:cyan", -0.5),
        ("cosine (table)", [table_hits["cosine"][k] for k in KS], "tab:orange", 0.5),
        ("cosine (known)", [known_hits["cosine"][k] for k in KS], "tab:red", 1.5),
    ]
    for label, vals, color, offset in bars:
        ax.bar(x + offset * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hit@{k}" for k in KS])
    ax.set_ylabel("%")
    ax.set_title("hit@k: comparison-table-derived vs already-committed numbers")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "hit_at_k_double_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 2. n_peaks_test dependency + cosine zero-tie check --------------------


def report_zero_cosine_hits(
    table_csv: str, correct: pd.DataFrame, eps: float = 1e-9
) -> None:
    # Uses cosine_hit1_raw (the literal rank==1 flag), not the floor-tie-
    # corrected cosine_hit1 -- this function's whole point is diagnosing the
    # floor-tie phenomenon itself, so it needs the uncorrected population,
    # not one that's already excluded it by construction.
    n_cosine_hit1 = int(correct["cosine_hit1_raw"].sum())
    zero_hits = correct[
        correct["cosine_hit1_raw"] & (correct["cosine_similarity"].abs() <= eps)
    ]
    print(
        f"\ncosine hit@1 total (raw, uncorrected): {n_cosine_hit1} / {len(correct)} test spectra"
    )
    print(
        f"cosine hit@1 with cosine_similarity ~ 0 (|sim| <= {eps}): {len(zero_hits)} "
        f"({100 * len(zero_hits) / n_cosine_hit1:.1f}% of cosine hit@1s)"
    )
    if len(zero_hits) == 0:
        return

    target_idx = set(zero_hits["test_spec_idx"].astype(int).tolist())
    print(
        f"Re-checking full candidate pools for these {len(target_idx)} test spectra ..."
    )
    thin = pd.read_csv(table_csv, usecols=["test_spec_idx", "cosine_similarity"])
    thin = thin[thin["test_spec_idx"].isin(target_idx)]
    n_all_tied = 0
    for _, grp in thin.groupby("test_spec_idx"):
        valid = grp["cosine_similarity"].dropna()
        if not valid.empty and (valid.abs() <= eps).all():
            n_all_tied += 1
    print(
        f"  {n_all_tied} of these have EVERY candidate in the pool at ~0 similarity "
        "-- a genuine floor-tie, the 'hit' is arbitrary sort order, not signal"
    )
    print(
        f"  {len(target_idx) - n_all_tied} have at least one candidate with nonzero cosine similarity"
    )


def run_npeaks_checks(table_csv: str, correct: pd.DataFrame, out_dir: Path) -> None:
    cand_peaks = correct["n_peaks_candidate"].dropna()
    print(
        f"\nn_peaks_candidate (true candidate's raw ICEBERG peak count): "
        f"min={cand_peaks.min():.0f} max={cand_peaks.max():.0f} median={cand_peaks.median():.0f}, "
        f"{cand_peaks.nunique()} unique value(s) -- "
        f"{'degenerate (ICEBERG always emits exactly 100 peaks), not plotted' if cand_peaks.nunique() <= 2 else 'has real variation'}"
    )

    bins = [0, 5, 10, 20, 30, 50, 100, 300]
    binned = correct.copy()
    binned["bin"] = pd.cut(binned["n_peaks_test"], bins, include_lowest=True)
    g = binned.groupby("bin", observed=True).agg(
        n=("simba_hit1", "size"),
        simba_rate=("simba_hit1", "mean"),
        cosine_rate=("cosine_hit1", "mean"),
    )
    print("\n=== hit@1 rate by n_peaks_test (real query spectrum) ===")
    print(g)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(g))
    width = 0.35
    ax.bar(x - width / 2, g["simba_rate"] * 100, width, label="SIMBA", color="tab:blue")
    ax.bar(
        x + width / 2, g["cosine_rate"] * 100, width, label="cosine", color="tab:orange"
    )
    for i, (_, row) in enumerate(g.iterrows()):
        ax.text(
            i,
            max(row["simba_rate"], row["cosine_rate"]) * 100 + 1.5,
            f"n={int(row['n']):,}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in g.index], rotation=30, ha="right")
    ax.set_xlabel("n_peaks_test (real query spectrum, raw peak count)")
    ax.set_ylabel("hit@1 rate (%)")
    ax.set_ylim(0, max(g["simba_rate"].max(), g["cosine_rate"].max()) * 100 + 8)
    ax.set_title(
        "Retrieval hit@1 rate by real-spectrum peak count\n(n_peaks_candidate omitted: degenerate, always 100 -- see module docstring)"
    )
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "npeaks_test_vs_hit1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    report_zero_cosine_hits(table_csv, correct)


# --- 3. precursor-mass discrepancy boxplot ---------------------------------


def split_by_cell(correct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full per-cell population, no sampling -- every test spectrum in that
    cell gets a precursor-discrepancy value, not just a subsample of it."""
    masks = {
        "both_correct": correct["simba_hit1"] & correct["cosine_hit1"],
        "simba_only": correct["simba_hit1"] & ~correct["cosine_hit1"],
        "cosine_only": ~correct["simba_hit1"] & correct["cosine_hit1"],
        "neither": ~correct["simba_hit1"] & ~correct["cosine_hit1"],
    }
    return {key: correct[mask] for key, mask in masks.items()}


def run_precursor_boxplot(correct: pd.DataFrame, out_dir: Path) -> None:
    """test_precursor_mz/candidate_precursor_mz now live directly on the
    table (added by tools/add_precursor_columns.py -- a one-time MGF scan +
    a cached canonical candidate_tsv lookup, see that script's module
    docstring), so this is a plain vectorized column computation -- no MGF
    scan, no per-row RDKit lookup, no multiprocessing needed here at all."""
    picked = split_by_cell(correct)
    print(
        "\n=== Precursor-mass discrepancy by confusion-matrix cell (full population) ==="
    )

    ppm_by_cell: dict[str, list[float]] = {}
    for key, _ in CELLS:
        sub = picked[key]
        meas = sub["test_precursor_mz"]
        calc = sub["candidate_precursor_mz"]
        valid = meas.notna() & calc.notna()
        n_missing = int((~valid).sum())
        ppms = (calc[valid] - meas[valid]).abs() / meas[valid] * 1e6
        ppm_by_cell[key] = ppms.tolist()
        arr = ppms.to_numpy()
        print(
            f"  {key}: {len(ppms)} resolved, {n_missing} missing -- "
            f"median={np.median(arr):.3f} ppm, p5={np.percentile(arr, 5):.3f}, "
            f"p95={np.percentile(arr, 95):.3f}, max={arr.max():.3f}"
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    data = [ppm_by_cell[key] for key, _ in CELLS]
    labels = [f"{label}\n(n={len(ppm_by_cell[key]):,})" for key, label in CELLS]
    ax.boxplot(data, tick_labels=labels, whis=(5, 95), showfliers=False)
    ax.set_ylabel("|measured - calculated| precursor m/z (ppm)")
    ax.set_title(
        "Precursor mass discrepancy by confusion-matrix cell "
        "(full population, whiskers=5th/95th pct)"
    )
    fig.tight_layout()
    out_path = out_dir / "precursor_ppm_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 4. SIMBA-MCES x cosine-similarity heatmap by confusion-matrix cell ----


def _plot_score_heatmap_grid(
    correct: pd.DataFrame,
    masks: dict,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    log_scale: bool,
    out_path: Path,
) -> None:
    """One cell per confusion-matrix quadrant, each laid out as a joint plot:
    the 2D SIMBA-MCES x cosine-similarity histogram in the middle, with its
    two marginal (per-axis) histograms alongside -- same x/y bins for the
    joint plot and its own marginals. Title reports n plus how many/what
    fraction of that cell's pairs have cosine similarity < 0.05 (the
    floor-tie region already flagged as an arbitrary-ranking artifact in
    run_npeaks_checks' zero-similarity check)."""
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(13, 11))
    outer = fig.add_gridspec(
        2, 2, wspace=0.4, hspace=0.55, left=0.06, right=0.96, top=0.90, bottom=0.06
    )

    for (key, label), outer_cell in zip(CELLS, outer):
        sub = correct[masks[key]]
        x = sub["simba_mces"].to_numpy()
        y = sub["cosine_similarity"].to_numpy()
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]

        n_low_cos = int((y < 0.05).sum())
        pct_low_cos = 100 * n_low_cos / len(y) if len(y) else 0.0

        # 2x3: main/top-hist in col 0, right-hist in col 1, a DEDICATED
        # colorbar axis in col 2 -- letting fig.colorbar steal space from
        # ax_right directly (the simpler approach) collided with ax_main's
        # x-axis label below it.
        inner = outer_cell.subgridspec(
            2,
            3,
            width_ratios=[4, 1, 0.2],
            height_ratios=[1, 4],
            wspace=0.1,
            hspace=0.05,
        )
        ax_main = fig.add_subplot(inner[1, 0])
        ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
        ax_cbar = fig.add_subplot(inner[1, 2])

        norm = LogNorm() if log_scale else None
        _, _, _, im = ax_main.hist2d(
            x, y, bins=[x_edges, y_edges], cmap="viridis", norm=norm
        )
        ax_main.set_xlabel("SIMBA-predicted MCES (true candidate)")
        ax_main.set_ylabel("cosine similarity (true candidate)")

        ax_top.hist(x, bins=x_edges, color="tab:gray")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("count")
        ax_top.set_title(
            f"{label.replace(chr(10), ' / ')} (n={len(x):,}, "
            f"cosine<0.05: {n_low_cos:,} [{pct_low_cos:.1f}%])",
            fontsize=9,
        )

        ax_right.hist(y, bins=y_edges, orientation="horizontal", color="tab:gray")
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_xlabel("count")

        fig.colorbar(im, cax=ax_cbar, label="count")

    scale_label = "log" if log_scale else "linear"
    fig.suptitle(
        "SIMBA-predicted MCES vs cosine similarity of the TRUE candidate, "
        f"by confusion-matrix cell ({scale_label} color scale)"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_score_heatmaps(correct: pd.DataFrame, out_dir: Path, n_bins: int = 25) -> None:
    """For each of the 4 confusion-matrix cells, a 2D histogram (+ per-axis
    marginal histograms) of the TRUE candidate's own SIMBA-predicted MCES
    (x) vs raw cosine similarity (y) across that cell's test spectra -- same
    population/cells run_confusion_and_hitk counts (one point per test
    spectrum, the true candidate's own row), visualized as a joint
    distribution instead of a single count. Same x/y bin edges across all 4
    subplots (0-40 MCES, 0-1 similarity) so panels are directly comparable.
    Saved twice, log- and linear-color-scale: log makes small cells (475 to
    10,481 points, >20x range) visible at all; linear shows true relative
    density within each cell without a log squashing it."""
    x_edges = np.linspace(0.0, 40.0, n_bins + 1)
    y_edges = np.linspace(0.0, 1.0, n_bins + 1)

    masks = {
        "both_correct": correct["simba_hit1"] & correct["cosine_hit1"],
        "simba_only": correct["simba_hit1"] & ~correct["cosine_hit1"],
        "cosine_only": ~correct["simba_hit1"] & correct["cosine_hit1"],
        "neither": ~correct["simba_hit1"] & ~correct["cosine_hit1"],
    }

    _plot_score_heatmap_grid(
        correct,
        masks,
        x_edges,
        y_edges,
        log_scale=True,
        out_path=out_dir / "simba_mces_vs_cosine_similarity_by_cell_log.png",
    )
    _plot_score_heatmap_grid(
        correct,
        masks,
        x_edges,
        y_edges,
        log_scale=False,
        out_path=out_dir / "simba_mces_vs_cosine_similarity_by_cell_linear.png",
    )


# --- entry point ------------------------------------------------------------


def run(
    table_csv: str,
    simba_retrieval_results_tsv: str,
    cosine_retrieval_results_tsv: str,
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading true-candidate rows from {table_csv} ...")
    correct = load_true_candidate_rows(
        table_csv,
        extra_cols=[
            "simba_mces",
            "n_peaks_test",
            "n_peaks_candidate",
            "test_precursor_mz",
            "candidate_precursor_mz",
        ],
    )
    print(f"  {len(correct):,} true-candidate rows (one per scored test spectrum)")

    run_confusion_and_hitk(
        correct, simba_retrieval_results_tsv, cosine_retrieval_results_tsv, out_dir
    )
    run_npeaks_checks(table_csv, correct, out_dir)
    run_precursor_boxplot(correct, out_dir)
    run_score_heatmaps(correct, out_dir)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--simba_retrieval_results_tsv", required=True)
    p.add_argument("--cosine_retrieval_results_tsv", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
