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
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm.auto import tqdm


def _exact_mass(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.ExactMolWt(mol) if mol else float("nan")


def _heavy_atoms(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol else 0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/calibration_analysis.png")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig2_path = out.with_name(out.stem + "_pop.png")

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
    oracle_gt = np.array([float(r["oracle_gt_mces"]) for r in rows])
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

    # ── RDKit properties (cached) ─────────────────────────────────────────────
    print("Computing molecular properties ...")
    mass_cache: dict[str, float] = {}
    atom_cache: dict[str, int] = {}
    unique_smi = {r["test_smi"] for r in rows} | {r["simba_smi"] for r in rows}
    for s in tqdm(unique_smi, desc="RDKit"):
        mass_cache[s] = _exact_mass(s)
        atom_cache[s] = _heavy_atoms(s)

    mass_test = np.array([mass_cache[r["test_smi"]] for r in rows])
    mass_simba = np.array([mass_cache[r["simba_smi"]] for r in rows])
    atoms_test = np.array([atom_cache[r["test_smi"]] for r in rows])
    atoms_simba = np.array([atom_cache[r["simba_smi"]] for r in rows])
    mass_diff = np.abs(mass_test - mass_simba)
    atoms_diff = np.abs(atoms_test - atoms_simba).astype(float)

    om = covered  # covered mask

    # ── Print summary stats ───────────────────────────────────────────────────
    print(f"\n=== Summary (covered {om.sum():,} spectra) ===")
    print(
        f"  {'spectral' if has_spectral else 'embedding'} cosine SIMBA pick:  mean={cosine_simba[om].mean():.3f}  median={np.median(cosine_simba[om]):.3f}"
    )
    print(
        f"  {'spectral' if has_spectral else 'embedding'} cosine oracle pick: mean={cosine_oracle[om].mean():.3f}  median={np.median(cosine_oracle[om]):.3f}"
    )
    print(
        f"  mass_diff pair:         mean={mass_diff[om].mean():.1f} Da  median={np.median(mass_diff[om]):.1f} Da"
    )
    print(
        f"  atoms_diff pair:        mean={atoms_diff[om].mean():.1f}   median={np.median(atoms_diff[om]):.1f}"
    )
    print(f"  Tanimoto SIMBA pair:    mean={simba_tani[om].mean():.3f}")

    HIGH = om & (simba_err > 25)
    LOW = om & (simba_err < 5)
    print(
        f"\n  High error (>25): n={HIGH.sum():,}  mass_diff mean={mass_diff[HIGH].mean():.1f} Da  "
        f"atoms_diff mean={atoms_diff[HIGH].mean():.1f}  tani mean={simba_tani[HIGH].mean():.3f}"
    )
    print(
        f"  Low error  (<5):  n={LOW.sum():,}   mass_diff mean={mass_diff[LOW].mean():.1f} Da  "
        f"atoms_diff mean={atoms_diff[LOW].mean():.1f}  tani mean={simba_tani[LOW].mean():.3f}"
    )

    # ── Figure 1: Error anatomy (2×3) ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Cosine sim distribution: SIMBA pick vs oracle pick
    ax = axes[0, 0]
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

    # 2. simba_err vs mass_diff (hexbin)
    ax = axes[0, 1]
    hb = ax.hexbin(mass_diff[om], simba_err[om], gridsize=30, cmap="Reds", mincnt=1)
    plt.colorbar(hb, ax=ax, label="count")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("|mass_test − mass_SIMBA_pick|  (Da)")
    ax.set_ylabel("simba_err  (GT MCES − pred MCES)")
    ax.set_title("2 · Calibration error vs mass difference", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # 3. simba_err vs atoms_diff
    ax = axes[0, 2]
    hb = ax.hexbin(atoms_diff[om], simba_err[om], gridsize=30, cmap="Purples", mincnt=1)
    plt.colorbar(hb, ax=ax, label="count")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("|atoms_test − atoms_SIMBA_pick|")
    ax.set_ylabel("simba_err")
    ax.set_title("3 · Calibration error vs atom count diff", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # 4. simba_err vs atoms_test
    ax = axes[1, 0]
    hb = ax.hexbin(
        atoms_test[om].astype(float), simba_err[om], gridsize=30, cmap="Blues", mincnt=1
    )
    plt.colorbar(hb, ax=ax, label="count")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Heavy atom count of test molecule")
    ax.set_ylabel("simba_err")
    ax.set_title("4 · Calibration error vs test mol size", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # 5. simba_err vs Tanimoto(test, SIMBA pick)
    ax = axes[1, 1]
    hb = ax.hexbin(simba_tani[om], simba_err[om], gridsize=30, cmap="Oranges", mincnt=1)
    plt.colorbar(hb, ax=ax, label="count")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Tanimoto similarity (test, SIMBA pick)")
    ax.set_ylabel("simba_err")
    ax.set_title("5 · Calibration error vs structural similarity", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # 6. GT MCES distribution of SIMBA picks with pred mean
    ax = axes[1, 2]
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
    print(f"\nFigure 1 saved → {out}")

    # ── Figure 2: Population analysis (2×2) ──────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

    # A. mass_diff distribution: high vs low error
    ax = axes2[0, 0]
    bins_m = np.linspace(0, np.percentile(mass_diff[om], 99), 60)
    ax.hist(
        mass_diff[LOW],
        bins=bins_m,
        color="#4E9A7A",
        alpha=0.75,
        density=True,
        edgecolor="none",
        label=f"err<5   n={LOW.sum():,}  μ={mass_diff[LOW].mean():.1f} Da",
    )
    ax.hist(
        mass_diff[HIGH],
        bins=bins_m,
        color="#E07B54",
        alpha=0.75,
        density=True,
        edgecolor="none",
        label=f"err>25  n={HIGH.sum():,}  μ={mass_diff[HIGH].mean():.1f} Da",
    )
    ax.set_xlabel("|mass_test − mass_SIMBA_pick|  (Da)")
    ax.set_ylabel("density")
    ax.set_title("A · Mass diff: low-error vs high-error pairs", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # B. 2D hexbin (mass_test, mass_simba) colored by mean simba_err
    ax = axes2[0, 1]
    lim = np.percentile(np.concatenate([mass_test[om], mass_simba[om]]), 99)
    hb = ax.hexbin(
        mass_test[om],
        mass_simba[om],
        C=simba_err[om],
        gridsize=35,
        cmap="RdYlGn_r",
        reduce_C_function=np.mean,
        mincnt=3,
    )
    plt.colorbar(hb, ax=ax, label="mean simba_err")
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.7, label="test = SIMBA pick")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Exact mass — test molecule  (Da)")
    ax.set_ylabel("Exact mass — SIMBA pick  (Da)")
    ax.set_title("B · Mass pairs colored by mean calibration error", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # C. Tanimoto distribution: high vs low error
    ax = axes2[1, 0]
    bins_t = np.linspace(0, 1, 41)
    ax.hist(
        simba_tani[LOW],
        bins=bins_t,
        color="#4E9A7A",
        alpha=0.75,
        density=True,
        edgecolor="none",
        label=f"err<5   μ={simba_tani[LOW].mean():.3f}",
    )
    ax.hist(
        simba_tani[HIGH],
        bins=bins_t,
        color="#E07B54",
        alpha=0.75,
        density=True,
        edgecolor="none",
        label=f"err>25  μ={simba_tani[HIGH].mean():.3f}",
    )
    ax.set_xlabel("Tanimoto similarity (test, SIMBA pick)")
    ax.set_ylabel("density")
    ax.set_title("C · Tanimoto: low-error vs high-error pairs", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # D. SIMBA pick GT vs oracle GT colored by simba_err
    ax = axes2[1, 1]
    hb = ax.hexbin(
        oracle_gt[om],
        simba_gt[om],
        C=simba_err[om],
        gridsize=30,
        cmap="RdYlGn_r",
        reduce_C_function=np.mean,
        mincnt=3,
    )
    plt.colorbar(hb, ax=ax, label="mean simba_err")
    ax.plot([0, 40], [0, 40], "k--", lw=1.2, alpha=0.7, label="SIMBA = oracle")
    ax.set_xlabel("Oracle GT MCES")
    ax.set_ylabel("SIMBA pick GT MCES")
    ax.set_title("D · SIMBA GT vs oracle GT (colored by error)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig2.suptitle(
        "Population Analysis: High-error (err>25) vs Low-error (err<5) SIMBA Retrieval Pairs",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=140, bbox_inches="tight")
    print(f"Figure 2 saved → {fig2_path}")


if __name__ == "__main__":
    main()
