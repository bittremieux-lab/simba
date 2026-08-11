"""8b: three checks against build_retrieval_comparison_table.py's CSV, all
computed from the table alone (no re-scoring, no SLURM needed -- everything
here is a thin column read plus a handful of small candidate_tsv lookups).

1. Confusion matrix (SIMBA top-1 correct x cosine top-1 correct) + hit@k
   double-check against the already-committed retrieval_results.tsv values,
   and a check of how often cosine's top-1 "win" is really just an
   arbitrary tie among a candidate pool that's entirely flat at ~0
   similarity (no real discrimination at all).

2. Retrieval difficulty vs real-spectrum peak count (n_peaks_test). Only
   n_peaks_test is plotted -- n_peaks_candidate (the true candidate's raw
   ICEBERG peak count) is degenerate, always exactly 100 for every
   candidate, because ICEBERG was run with --sparse-k 100 --threshold 0.0,
   so it always emits exactly 100 nonzero-mass fragments regardless of
   molecule size. A 2D heatmap against it collapses to one row; this is
   reported (not plotted) as a sanity check instead.

3. Precursor-mass discrepancy (a test spectrum's own measured PRECURSOR_MZ
   vs its true candidate's formula-implied calculated m/z) across the 4
   confusion-matrix cells, as a boxplot. This can't be why SIMBA picks the
   wrong candidate WITHIN a pool -- every candidate in a formula-matched
   pool shares the same calculated precursor (checked directly: the
   candidate SIMBA actually top-ranks always matches the true candidate's
   calculated precursor to within 0.01 ppm, in every case checked) -- but a
   query's own measured-vs-theoretical gap could still correlate with
   overall difficulty, which is what this checks.

Usage:
    uv run python tools/plot_retrieval_comparison_checks.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --simba_retrieval_results_tsv /path/to/008_2_.../retrieval_iceberg/retrieval_results.tsv \\
        --cosine_retrieval_results_tsv /path/to/cosine_baseline_iceberg/retrieval_results.tsv \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --output_dir /path/to/output \\
        --n_per_cell 100 --seed 42
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simba_retrieval import canonicalize


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
        ]
        + extra_cols,
    )
    correct = df[df["test_smiles"] == df["candidate_smiles"]].copy()
    correct["simba_hit1"] = correct["simba_rank"] == 1
    correct["cosine_hit1"] = correct["cosine_rank"] == 1
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
    print(f"  n = {n:,} test spectra (with a resolvable true-candidate rank)")
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
    ax.set_title(f"Top-1 correctness: SIMBA vs cosine (n={n:,})")
    fig.tight_layout()
    out_path = out_dir / "confusion_matrix_simba_vs_cosine_top1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    table_hits = {}
    for method, rank_col in [("simba", "simba_rank"), ("cosine", "cosine_rank")]:
        table_hits[method] = {k: 100 * (correct[rank_col] <= k).mean() for k in KS}
    known_hits = {
        "simba": load_known_hit_rates(simba_retrieval_results_tsv),
        "cosine": load_known_hit_rates(cosine_retrieval_results_tsv),
    }

    print("\n=== hit@k: table-derived vs already-committed retrieval_results.tsv ===")
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
    n_cosine_hit1 = int(correct["cosine_hit1"].sum())
    zero_hits = correct[
        correct["cosine_hit1"] & (correct["cosine_similarity"].abs() <= eps)
    ]
    print(f"\ncosine hit@1 total: {n_cosine_hit1} / {len(correct)} test spectra")
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


def pick_cases_per_cell(
    correct: pd.DataFrame, n: int, seed: int
) -> dict[str, pd.DataFrame]:
    masks = {
        "both_correct": correct["simba_hit1"] & correct["cosine_hit1"],
        "simba_only": correct["simba_hit1"] & ~correct["cosine_hit1"],
        "cosine_only": ~correct["simba_hit1"] & correct["cosine_hit1"],
        "neither": ~correct["simba_hit1"] & ~correct["cosine_hit1"],
    }
    rng = np.random.default_rng(seed)
    return {
        key: correct[mask].sample(
            n=min(n, int(mask.sum())), random_state=rng.integers(0, 2**31 - 1)
        )
        for key, mask in masks.items()
    }


def get_test_precursor_by_index(
    mgf_path: str, target_indices: set[int]
) -> dict[int, float]:
    result = {}
    test_idx = -1
    current = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current = {}
                continue
            if line == "END IONS":
                if current.get("FOLD") == "test":
                    test_idx += 1
                    if test_idx in target_indices:
                        result[test_idx] = float(current["PRECURSOR_MZ"])
                        if len(result) == len(target_indices):
                            return result
                current = None
                continue
            if current is None:
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                current[k] = v
    return result


def lookup_true_candidate_precursor(
    cand_index: pd.DataFrame,
    smi_canon: str,
    adduct: str,
    measured_mz: float,
    mass_tol: float = 0.05,
) -> float | None:
    subset = cand_index[
        (cand_index["ionization"] == adduct)
        & (cand_index["precursor"].sub(measured_mz).abs() < mass_tol)
    ]
    if subset.empty:
        return None
    canon_smi = subset["smiles"].map(canonicalize)
    match = subset[canon_smi == smi_canon]
    return float(match.iloc[0]["precursor"]) if not match.empty else None


def run_precursor_boxplot(
    correct: pd.DataFrame,
    mgf: str,
    candidate_tsv: str,
    out_dir: Path,
    n_per_cell: int,
    seed: int,
) -> None:
    picked = pick_cases_per_cell(correct, n_per_cell, seed)
    print("\n=== Precursor-mass discrepancy by confusion-matrix cell ===")
    for key, _ in CELLS:
        print(f"  {key}: {len(picked[key])} cases")

    all_rows = pd.concat(picked.values())
    target_idx = set(all_rows["test_spec_idx"].astype(int).tolist())
    print(
        f"Extracting measured precursor m/z for {len(target_idx)} test spectra from {mgf} ..."
    )
    measured = get_test_precursor_by_index(mgf, target_idx)

    print(f"Loading {candidate_tsv} ...")
    cand_index = pd.read_csv(candidate_tsv, sep="\t")

    ppm_by_cell: dict[str, list[float]] = {}
    for key, _ in CELLS:
        ppms = []
        n_missing = 0
        for _, row in picked[key].iterrows():
            spec_idx = int(row["test_spec_idx"])
            if spec_idx not in measured:
                n_missing += 1
                continue
            meas = measured[spec_idx]
            calc = lookup_true_candidate_precursor(
                cand_index, row["test_smiles"], row["test_adduct"], meas
            )
            if calc is None:
                n_missing += 1
                continue
            ppms.append(abs(calc - meas) / meas * 1e6)
        ppm_by_cell[key] = ppms
        arr = np.array(ppms)
        print(
            f"  {key}: {len(ppms)} resolved, {n_missing} missing -- "
            f"median={np.median(arr):.3f} ppm, p5={np.percentile(arr, 5):.3f}, "
            f"p95={np.percentile(arr, 95):.3f}, max={arr.max():.3f}"
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [ppm_by_cell[key] for key, _ in CELLS]
    labels = [label for _, label in CELLS]
    ax.boxplot(data, tick_labels=labels, whis=(5, 95), showfliers=False)
    ax.set_ylabel("|measured - calculated| precursor m/z (ppm)")
    ax.set_title(
        f"Precursor mass discrepancy by confusion-matrix cell "
        f"(n={n_per_cell}/cell, whiskers=5th/95th pct)"
    )
    fig.tight_layout()
    out_path = out_dir / "precursor_ppm_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- entry point ------------------------------------------------------------


def run(
    table_csv: str,
    simba_retrieval_results_tsv: str,
    cosine_retrieval_results_tsv: str,
    mgf: str,
    candidate_tsv: str,
    output_dir: str,
    n_per_cell: int = 100,
    seed: int = 42,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading true-candidate rows from {table_csv} ...")
    correct = load_true_candidate_rows(
        table_csv, extra_cols=["cosine_similarity", "n_peaks_test", "n_peaks_candidate"]
    )
    print(f"  {len(correct):,} true-candidate rows (one per scored test spectrum)")

    run_confusion_and_hitk(
        correct, simba_retrieval_results_tsv, cosine_retrieval_results_tsv, out_dir
    )
    run_npeaks_checks(table_csv, correct, out_dir)
    run_precursor_boxplot(correct, mgf, candidate_tsv, out_dir, n_per_cell, seed)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--simba_retrieval_results_tsv", required=True)
    p.add_argument("--cosine_retrieval_results_tsv", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidate_tsv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_per_cell", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
