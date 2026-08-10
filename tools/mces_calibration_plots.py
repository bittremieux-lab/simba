"""3e: GT-binned calibration boxplots for SIMBA-predicted vs GT MCES
(test-to-candidate and test-to-test), reusing the exact flat (pred, gt)
pairs from ood_generalization_check.py's per-spectrum scoring, same as the
MAE/Spearman numbers already reported there. The query is each individual
test spectrum, not one averaged per-molecule embedding — see that module's
docstring for why.

NO AVERAGING ANYWHERE, on either side — see ood_generalization_check.py's
module docstring for why (query-side averaging inflates accuracy;
candidate-side averaging across adducts blends two different in silico
spectra that were never meant to be compared to the same query).

EXCLUSION MODE: test-to-candidate includes self (the true candidate, GT=0,
via add_self_pairs, matched to the query's own adduct); test-to-test
excludes only the literal self spectrum and includes same-molecule-
different-spectrum pairs (GT=0) — see ood_generalization_check.py's module
docstring for the full explanation.

GT MCES binned on the x-axis, SIMBA-predicted MCES boxplotted per bin,
whis=(5, 95) and outlier points hidden (showfliers=False) so a few extreme
predictions per bin don't dominate the picture. Each box is annotated with
its n. GT is clipped to --gt_clip_max (default 40) on both plots so the two
populations share the same x-axis scale (test-to-candidate's GT otherwise
runs past 40, unlike test-to-test's, which is capped there by construction).

Usage:
    uv run python tools/mces_calibration_plots.py \\
        --intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --test_to_test_prepro_dir /path/to/preprocessing_msg_exact_mces_1020 \\
        --output_dir /path/to/output \\
        --mces_max_value 40
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ood_generalization_check import (
    add_self_pairs,
    build_candidate_embeddings_by_smi_adduct,
    expand_and_score_ragged,
    score_test_to_test_no_averaging,
)


def plot_binned_box(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    out_path: Path,
    bin_width: float = 2.0,
    min_n: int = 5,
    gt_clip_max: float = 40.0,
) -> None:
    keep = gt <= gt_clip_max
    pred, gt = pred[keep], gt[keep]

    edges = np.arange(0, gt_clip_max + bin_width, bin_width)
    bin_idx = np.digitize(gt, edges) - 1

    groups, positions, widths, ns = [], [], [], []
    for b in range(len(edges) - 1):
        vals = pred[bin_idx == b]
        if len(vals) < min_n:
            continue
        groups.append(vals)
        positions.append((edges[b] + edges[b + 1]) / 2)
        widths.append(bin_width * 0.8)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        groups, positions=positions, widths=widths, whis=(5, 95), showfliers=False
    )
    ax.plot(
        [0, gt_clip_max],
        [0, gt_clip_max],
        color="red",
        linestyle="--",
        linewidth=1,
        label="pred = GT",
    )

    ymax = max(np.percentile(v, 95) for v in groups)
    label_y = ymax * 1.03
    for pos, n in zip(positions, ns):
        ax.text(
            pos, label_y, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90
        )
    ax.set_ylim(top=label_y * 1.25)

    ax.set_xlabel("GT MCES (binned)")
    ax.set_ylabel("SIMBA-predicted MCES")
    n_ticks = min(len(positions), 15)
    step = max(1, len(positions) // n_ticks)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels([f"{p:.0f}" for p in positions[::step]])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved {out_path}  ({len(groups)} bins with >= {min_n} points, {len(pred):,}/{len(keep):,} pairs kept after GT<={gt_clip_max} clip)"
    )


def run(
    intermediates_dir: str,
    gt_mces_dir: str,
    test_to_test_prepro_dir: str,
    output_dir: str,
    mces_max_value: float = 40.0,
    gt_clip_max: float = 40.0,
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

    print("\n--- test-to-candidate (self included, own-adduct match) ---")
    gt_dir = Path(gt_mces_dir)
    tc_idx_to_smiles = gt_dir.joinpath("smiles.txt").read_text().splitlines()
    tc_pairs = np.load(gt_dir / "mces_exact.npy")
    valid = (tc_pairs[:, 2] >= 0) & ~np.isnan(tc_pairs[:, 2])
    tc_pairs = tc_pairs[valid]
    tc_pairs = add_self_pairs(tc_pairs)
    _, _, tc_gt, tc_pred = expand_and_score_ragged(
        tc_pairs,
        tc_idx_to_smiles,
        test_smiles,
        test_adducts,
        test_embeddings,
        cand_smi_adduct_to_emb,
        mces_max_value,
    )
    print(f"  {len(tc_gt):,} scored pairs (incl. self)")

    plot_binned_box(
        tc_pred,
        tc_gt,
        "Test-to-candidate: SIMBA-predicted MCES by GT bin (self included)",
        out_dir / "test_to_candidate_binned_box.png",
        gt_clip_max=gt_clip_max,
    )

    print(
        "\n--- test-to-test (self spectrum excluded, same-molecule spectra included, no averaging) ---"
    )
    with open(Path(test_to_test_prepro_dir) / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    tt_idx_to_smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()
    tt_pairs = np.load(
        Path(test_to_test_prepro_dir)
        / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy"
    )
    tt_pred_matrix, tt_gt_matrix, tt_valid_mask, _ = score_test_to_test_no_averaging(
        tt_pairs, tt_idx_to_smiles, test_smiles, test_embeddings, mces_max_value
    )
    tt_pred, tt_gt = tt_pred_matrix[tt_valid_mask], tt_gt_matrix[tt_valid_mask]
    print(f"  {len(tt_gt):,} scored individual-spectrum pairs")

    plot_binned_box(
        tt_pred,
        tt_gt,
        "Test-to-test: SIMBA-predicted MCES by GT bin (self spectrum excluded, same-molecule spectra included, no averaging)",
        out_dir / "test_to_test_binned_box.png",
        gt_clip_max=gt_clip_max,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--intermediates_dir", required=True)
    p.add_argument("--gt_mces_dir", required=True)
    p.add_argument("--test_to_test_prepro_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mces_max_value", type=float, default=40.0)
    p.add_argument("--gt_clip_max", type=float, default=40.0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
