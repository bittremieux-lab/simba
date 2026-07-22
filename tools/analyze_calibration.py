"""
Calibration error decomposition for SIMBA retrieval picks.

Reads the diagnostic CSV and decomposes simba_err (GT MCES − SIMBA pred MCES)
by pair-level molecular properties to understand why SIMBA is overconfident.

Usage:
    uv run python tools/analyze_calibration.py \
        --csv results/retrieval_diagnostics_bs2048_v2_step44k.csv \
        --output results/calibration_analysis_bs2048_v2_step44k.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/calibration_analysis.png")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CSV ...")
    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"  {len(rows):,} spectra")

    # ── Parse scalars ─────────────────────────────────────────────────────────
    simba_err = np.array([float(r["simba_err"]) for r in rows])
    simba_pred = np.array([float(r["simba_pred_mces"]) for r in rows])
    simba_gt = np.array([float(r["simba_gt_mces"]) for r in rows])
    oracle_pred = np.array([float(r["oracle_pred_mces"]) for r in rows])
    simba_tani = np.array([float(r["simba_tanimoto"]) for r in rows])
    covered = np.array([r["covered"] == "True" for r in rows])

    # Spectral cosine similarities (peak-based, from MGF) — use if available
    has_spectral = (
        "simba_spectral_cos" in rows[0] and rows[0]["simba_spectral_cos"] != ""
    )
    if has_spectral:
        cosine_simba = np.array([float(r["simba_spectral_cos"]) for r in rows])
        cosine_oracle = np.array([float(r["oracle_spectral_cos"]) for r in rows])
        cosine_label = "Spectral cosine similarity (peak-based)"
    else:
        # Fallback: embedding-derived cosine (less meaningful but available)
        cosine_simba = 1.0 - simba_pred / 40.0
        cosine_oracle = 1.0 - oracle_pred / 40.0
        cosine_label = "Cosine sim (embedding, 1−pred/40) ← re-run diagnose_retrieval to get spectral"

    om = covered  # covered mask

    # ── Print summary stats ───────────────────────────────────────────────────
    print(f"\n=== Summary (covered {om.sum():,} spectra) ===")
    print(
        f"  {'spectral' if has_spectral else 'embedding'} cosine SIMBA pick:  mean={cosine_simba[om].mean():.3f}  median={np.median(cosine_simba[om]):.3f}"
    )
    print(
        f"  {'spectral' if has_spectral else 'embedding'} cosine oracle pick: mean={cosine_oracle[om].mean():.3f}  median={np.median(cosine_oracle[om]):.3f}"
    )
    print(f"  Tanimoto SIMBA pair:    mean={simba_tani[om].mean():.3f}")

    # ── Figure 1: panels 1 + 6 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Cosine sim distribution: SIMBA pick vs oracle pick
    ax = axes[0]
    bins1 = np.linspace(0, 1, 51)
    ax.hist(
        cosine_oracle[om],
        bins=bins1,
        color="#4E9A7A",
        alpha=0.75,
        edgecolor="none",
        label=f"Oracle pick  μ={cosine_oracle[om].mean():.3f}",
    )
    ax.hist(
        cosine_simba[om],
        bins=bins1,
        color="#E07B54",
        alpha=0.75,
        edgecolor="none",
        label=f"SIMBA pick   μ={cosine_simba[om].mean():.3f}",
    )
    ax.set_xlabel(cosine_label)
    ax.set_ylabel("# test spectra")
    ax.set_title("1 · Spectral cosine: oracle vs SIMBA pick", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 6. GT MCES distribution of SIMBA picks with pred mean
    ax = axes[1]
    bins6 = np.arange(0, 42.5, 2.5)
    ax.hist(simba_gt[om], bins=bins6, color="#5B8DB8", edgecolor="none", alpha=0.85)
    ax.axvline(
        simba_gt[om].mean(),
        color="red",
        lw=1.5,
        ls="--",
        label=f"GT mean={simba_gt[om].mean():.1f}",
    )
    ax.axvline(
        simba_pred[om].mean(),
        color="green",
        lw=1.5,
        ls="--",
        label=f"pred mean={simba_pred[om].mean():.1f}",
    )
    ax.set_xlabel("GT MCES of SIMBA's picked training molecule")
    ax.set_ylabel("# test spectra")
    ax.set_title("6 · GT MCES of SIMBA picks", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"SIMBA Calibration Error Anatomy — {om.sum():,} covered test spectra\n"
        f"simba_err mean={simba_err[om].mean():.1f}  std={simba_err[om].std():.1f}  "
        f"(GT MCES − pred MCES, positive = overconfident)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
