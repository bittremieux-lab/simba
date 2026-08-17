"""Wout's question: how can cosine be "correct" for ~1000 queries with ~0
cosine similarity? Most of that (1,001/6,599 = 15.2% of cosine's hit@1s) is
a genuine floor-tie: the ENTIRE candidate pool sits at cosine_similarity==0,
so the "win" is pure list-order luck, not signal (confirmed directly in
plot_retrieval_comparison_checks.py's report_zero_cosine_hits). Excluding
those, hit@1 corrects from 37.59% to 31.89%.

But 145 more cases (17.4% - 15.2% of cosine's hits) have LOW but genuinely
NONZERO cosine similarity (0 < sim < 0.05) -- these aren't floor-ties, so
"why did cosine still rank the true candidate first" is a real question.
This script plots 10 random examples from exactly that population: real test
spectrum (blue, up) vs ICEBERG-predicted true-candidate spectrum (red,
down), raw and after SIMBA's own preprocessing, reusing
plot_confusion_matrix_examples.py's extraction/preprocessing/plotting
functions unchanged (no new logic, just a narrower selection). Titles also
report each query's candidate pool size and its second-best cosine
similarity -- context for how much of a "win" the true candidate's rank-1
really is (a near-tied second place says a lot less than a clear margin).

Usage:
    uv run python tools/plot_low_cosine_hit_examples.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --iceberg_preds /path/to/preds.hdf5 \\
        --output_dir /path/to/output \\
        --n 10 --seed 42
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_confusion_matrix_examples import (
    extract_test_spectra_by_index,
    load_iceberg_peaks,
    plot_mirror,
    simba_preprocess,
)


def pick_low_cosine_examples(table_csv: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(
        table_csv,
        usecols=[
            "test_spec_idx",
            "test_smiles",
            "test_adduct",
            "candidate_smiles",
            "candidate_adduct",
            "cosine_rank",
            "cosine_similarity",
        ],
    )
    correct = df[df["test_smiles"] == df["candidate_smiles"]].copy()
    pool = correct[
        (correct["cosine_rank"] == 1)
        & (correct["cosine_similarity"] > 0)
        & (correct["cosine_similarity"] < 0.05)
    ]
    print(
        f"{len(pool)} cosine-correct queries with 0 < cosine_similarity < 0.05 "
        "(genuinely low, not a floor-tie)"
    )
    rng = np.random.default_rng(seed)
    picked = pool.sample(
        n=min(n, len(pool)), random_state=rng.integers(0, 2**31 - 1)
    ).copy()

    # Pool size + second-best cosine similarity per picked example -- both
    # need the FULL candidate pool for that test_spec_idx (not just the true
    # candidate's own row), so re-read just those groups from the table.
    target_idx = set(picked["test_spec_idx"].astype(int).tolist())
    full = df[df["test_spec_idx"].isin(target_idx)]
    pool_sizes, second_best = {}, {}
    for spec_idx, grp in full.groupby("test_spec_idx"):
        sims = grp["cosine_similarity"].dropna().sort_values(ascending=False)
        pool_sizes[spec_idx] = len(grp)
        second_best[spec_idx] = float(sims.iloc[1]) if len(sims) > 1 else float("nan")
    picked["pool_size"] = picked["test_spec_idx"].map(pool_sizes)
    picked["second_best_cosine"] = picked["test_spec_idx"].map(second_best)
    return picked


def run(
    table_csv: str,
    mgf: str,
    candidate_tsv: str,
    iceberg_preds: str,
    output_dir: str,
    n: int = 5,
    seed: int = 42,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = pick_low_cosine_examples(table_csv, n, seed)

    target_indices = set(picked["test_spec_idx"].astype(int).tolist())
    print(f"\nExtracting {len(target_indices)} real test spectra from {mgf} ...")
    real_spectra = extract_test_spectra_by_index(mgf, target_indices)

    wanted_triples = []
    for _, row in picked.iterrows():
        spec_idx = int(row["test_spec_idx"])
        if spec_idx not in real_spectra:
            continue
        _, _, prec_mz, _, _ = real_spectra[spec_idx]
        wanted_triples.append(
            (row["candidate_smiles"], row["candidate_adduct"], prec_mz)
        )
    print(f"\nLooking up {len(wanted_triples)} true-candidate ICEBERG spectra ...")
    iceberg_peaks = load_iceberg_peaks(candidate_tsv, iceberg_preds, wanted_triples)
    print(f"  {len(iceberg_peaks)} / {len(wanted_triples)} resolved")

    n_cols = len(picked)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6.5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_i, (_, row) in enumerate(picked.iterrows()):
        raw_ax, proc_ax = axes[0, col_i], axes[1, col_i]
        spec_idx = int(row["test_spec_idx"])
        cand_key = (row["candidate_smiles"], row["candidate_adduct"])

        if spec_idx not in real_spectra or cand_key not in iceberg_peaks:
            raw_ax.set_title("missing data", fontsize=8)
            raw_ax.axis("off")
            proc_ax.axis("off")
            continue

        _, _, real_prec_mz, real_mz, real_intensity = real_spectra[spec_idx]
        cand_prec_mz, iceberg_mz, iceberg_intensity = iceberg_peaks[cand_key]

        title = (
            f"idx={spec_idx} [raw], pool={int(row['pool_size'])}\n"
            f"cosine: r=1 sim={row['cosine_similarity']:.4f} "
            f"(2nd best={row['second_best_cosine']:.4f})"
        )
        plot_mirror(
            raw_ax, real_mz, real_intensity, iceberg_mz, iceberg_intensity, title
        )

        p_real_mz, p_real_intensity = simba_preprocess(
            real_prec_mz, real_mz, real_intensity
        )
        p_iceberg_mz, p_iceberg_intensity = simba_preprocess(
            cand_prec_mz, iceberg_mz, iceberg_intensity
        )
        plot_mirror(
            proc_ax,
            p_real_mz,
            p_real_intensity,
            p_iceberg_mz,
            p_iceberg_intensity,
            "SIMBA-preprocessed\n(precursor removed, top-100, sqrt+L2-norm)",
        )

        if col_i == 0:
            raw_ax.set_ylabel("low cosine hit\n(raw)", fontsize=8)
            proc_ax.set_ylabel("low cosine hit\n(preprocessed)", fontsize=8)

    fig.suptitle(
        "Cosine-correct examples with genuinely low (not floor-tied) similarity\n"
        "Real test spectrum (blue, up) vs ICEBERG-predicted true-candidate spectrum (red, down)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, top=0.85)
    out_path = out_dir / "low_cosine_hit_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidate_tsv", required=True)
    p.add_argument("--iceberg_preds", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
