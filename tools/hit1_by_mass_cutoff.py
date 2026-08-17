"""Item 8c extension: hit@1 vs. mass cutoff, test-to-candidate only. Reuses
item 8b's retrieval_comparison_table.csv directly (simba_rank + is_correct
per test spectrum already computed there) -- no re-scoring, no re-embedding,
just mass filtering + aggregation.

Unlike mces_calibration_plots.py's mass-restricted comparison (item 8c,
which filters PAIRS by max(mass_query, mass_other) since MAE/Spearman there
is a pair-level metric), hit@1 is a per-test-spectrum metric, so filtering
is by the query molecule's own mass only -- candidates in a formula-matched
pool share essentially the same mass anyway (already confirmed directly in
item 8b, plot_retrieval_comparison_checks.py's precursor-mass check).

SIMBA only (not cosine): this is specifically probing whether SIMBA's known
mass-dependent calibration/ranking degradation (item 8d: per-cell Spearman
goes to ~0/negative for the largest-mass bins) shows up as an actual hit@1
drop, not just a Spearman drop.

Produces hit1_by_mass_cutoff.png: 1x2, left = SIMBA hit@1 rate vs cutoff,
right = fraction of test spectra retained at each cutoff (so a reader can
see how thin the tail gets before trusting a given point).

Usage:
    uv run python tools/hit1_by_mass_cutoff.py \\
        --comparison_csv /path/to/retrieval_comparison_table.csv \\
        --output_dir /path/to/output \\
        --mass_cutoffs 300,350,400,450,500,750,1000
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mces_calibration_plots import mass_from_smiles


def run(
    comparison_csv: str,
    output_dir: str,
    mass_cutoffs: str = "300,350,400,450,500,750,1000",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {comparison_csv} ...")
    df = pd.read_csv(
        comparison_csv,
        usecols=["test_spec_idx", "test_smiles", "simba_rank", "is_correct"],
    )
    top1 = df[df["simba_rank"] == 1].set_index("test_spec_idx", verify_integrity=True)
    n_test_total = len(top1)
    print(f"  {n_test_total:,} test spectra (exactly one simba_rank==1 row each)")

    print("Computing query mass (RDKit ExactMolWt, cached) ...")
    mass = top1["test_smiles"].map(mass_from_smiles)

    cutoffs = [float(x) for x in mass_cutoffs.split(",") if x.strip()]
    labels = [f"< {c:g} Da" for c in cutoffs] + ["no limit"]

    hit1_rates, coverage_fracs, ns = [], [], []
    for row, label in enumerate(labels):
        cutoff = cutoffs[row] if row < len(cutoffs) else None
        keep = (
            mass < cutoff if cutoff is not None else pd.Series(True, index=top1.index)
        )
        n = int(keep.sum())
        hit1 = float(top1.loc[keep, "is_correct"].mean()) if n else float("nan")
        hit1_rates.append(hit1)
        coverage_fracs.append(n / n_test_total)
        ns.append(n)
        print(
            f"  {label}: n={n:,} ({n / n_test_total:.1%} of test set), hit@1={hit1:.4f}"
        )

    x_pos = list(range(len(labels)))
    x_labels = [f"{c:g}" for c in cutoffs] + ["no limit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(x_pos, [h * 100 for h in hit1_rates], "o-", color="tab:green")
    for x, h, n in zip(x_pos, hit1_rates, ns):
        ax1.annotate(
            f"n={n:,}",
            (x, h * 100),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right")
    ax1.set_xlabel("mass cutoff (query molecule mass, Da)")
    ax1.set_ylabel("SIMBA hit@1 (%)")
    ax1.set_title("hit@1 vs mass cutoff (test-to-candidate)")

    ax2.plot(x_pos, [c * 100 for c in coverage_fracs], "o-", color="tab:gray")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right")
    ax2.set_xlabel("mass cutoff (query molecule mass, Da)")
    ax2.set_ylabel("% of test spectra retained")
    ax2.set_title("Coverage vs mass cutoff")

    fig.tight_layout()
    out_path = out_dir / "hit1_by_mass_cutoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--comparison_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mass_cutoffs", default="300,350,400,450,500,750,1000")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
