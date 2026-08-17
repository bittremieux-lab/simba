"""Tests the hypothesis: does SIMBA's predicted MCES for the TRUE candidate
grow with precursor-mass discrepancy (the same |measured - calculated| ppm
value plot_retrieval_comparison_checks.py's precursor_ppm_boxplot.png
already shows per confusion-matrix cell)? If SIMBA partly relies on
precursor agreement as a proxy for "is this really the same molecule," a
larger measured-vs-calculated gap might make it less confident (higher
predicted MCES) even for the actual true candidate.

All 4 confusion-matrix cells combined into one population here (not split),
using test_precursor_mz/candidate_precursor_mz directly off the table (see
tools/add_precursor_columns.py) -- no new lookups, just two existing
columns plotted against a third (simba_mces).

ppm spans ~8 orders of magnitude (min 8e-6, max ~2,480, heavily
right-skewed: median 0.075, p95 8.58) so the x-axis is log-scale. A binned
mean trend line (equal-count quantile bins in log-ppm space) is overlaid to
read the hypothesis directly off the plot rather than eyeballing a cloud of
17,555 points.

Usage:
    uv run python tools/plot_mces_vs_precursor_ppm.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --output_dir /path/to/output \\
        --n_bins 20
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(table_csv: str, output_dir: str, n_bins: int = 20) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {table_csv} ...")
    df = pd.read_csv(
        table_csv,
        usecols=[
            "test_smiles",
            "candidate_smiles",
            "simba_mces",
            "test_precursor_mz",
            "candidate_precursor_mz",
        ],
    )
    correct = df[df["test_smiles"] == df["candidate_smiles"]].copy()
    ppm = (
        (correct["candidate_precursor_mz"] - correct["test_precursor_mz"]).abs()
        / correct["test_precursor_mz"]
        * 1e6
    )
    mces = correct["simba_mces"]
    valid = ppm.notna() & mces.notna() & (ppm > 0)
    ppm, mces = ppm[valid].to_numpy(), mces[valid].to_numpy()
    print(f"  {len(ppm):,} true-candidate rows with both values")

    log_ppm = np.log10(ppm)
    edges = np.quantile(log_ppm, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(log_ppm, edges[1:-1])
    bin_centers, bin_means, bin_ns = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_centers.append(10 ** np.median(log_ppm[mask]))
        bin_means.append(mces[mask].mean())
        bin_ns.append(int(mask.sum()))
    print("Binned means (median ppm in bin -> mean SIMBA-predicted MCES):")
    for c, m, n in zip(bin_centers, bin_means, bin_ns):
        print(f"  ppm~{c:.4g} (n={n}): mean_mces={m:.2f}")

    rho = np.corrcoef(log_ppm, mces)[0, 1]
    print(f"\nPearson correlation (log10(ppm), simba_mces): {rho:.3f}")

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.05,
        hspace=0.05,
        left=0.09,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        ppm, mces, s=4, alpha=0.15, color="tab:blue", label="true-candidate rows"
    )
    ax_main.plot(bin_centers, bin_means, "o-", color="tab:red", label="binned mean")
    ax_main.set_xscale("log")
    ax_main.set_xlabel("|measured - calculated| precursor m/z (ppm)")
    ax_main.set_ylabel("SIMBA-predicted MCES (true candidate)")
    ax_main.legend(loc="upper left", fontsize=8)

    # ax_top shares ax_main's x-axis (ppm, log-scaled) via sharex -- bins
    # must be log-spaced in raw ppm units, not log10(ppm), or the bars would
    # land at the wrong positions against ax_main's actual ppm scale.
    log_spaced_bins = np.logspace(log_ppm.min(), log_ppm.max(), 60)
    ax_top.hist(ppm, bins=log_spaced_bins, color="tab:gray")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("count")

    mces_bins = np.linspace(0, max(mces.max(), 40), 41)
    ax_right.hist(mces, bins=mces_bins, orientation="horizontal", color="tab:gray")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("count")

    fig.suptitle(
        f"SIMBA-predicted MCES vs precursor-mass discrepancy\n"
        f"(n={len(ppm):,}, r(log ppm, mces)={rho:.3f})"
    )
    out_path = out_dir / "mces_vs_precursor_ppm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_bins", type=int, default=20)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
