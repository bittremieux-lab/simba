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

MASS-RESTRICTED COMPARISON (item 8c): re-filters the SAME already-scored
pairs (no re-scoring, no re-embedding) by max(mass_a, mass_b) < cutoff, for
--mass_cutoffs (default 300/500/750/1000 Da) plus "no limit", to see
whether calibration/error improves when restricted to a mass range closer
to what training actually saw. Produces `binned_box_by_mass_cutoff.png` (a
2-column x len(cutoffs)+1-row grid, same boxplot style as above, "no limit"
row reproducing the two standalone plots) and
`mae_spearman_by_mass_cutoff.png` (2 line plots — MAE and Spearman vs
cutoff, one line per population).

`mae_spearman_by_mass_cutoff.png` also overlays a GT-MCES-balanced version
of both metrics (dashed) alongside the raw ones (solid): at each cutoff,
every non-empty GT bin (0-5, 5-10, ..., up to gt_clip_max) is resampled to
exactly --gt_balance_target_n pairs (default 10,000) before computing
MAE/Spearman, so the metric isn't dominated by whichever GT range happens
to have the most pairs (typically the near-0 bin). A bin with fewer than
that available is oversampled WITH replacement rather than left thin --
plain min-bin-count balancing left test-to-candidate with as few as
91-1936 pairs/bin at some cutoffs, too small to trust. The run log reports
each bin's thinnest available count and how much oversampling (if any) it
took to reach the target.

CACHING: the dominant cost here is build_candidate_embeddings_by_smi_adduct
canonicalizing ~600k raw candidate SMILES with RDKit (same cost every other
8b script pays) plus the dense test-to-test matmul -- neither depends on
--mass_cutoffs at all, only the final filtering/plotting step does. So the
scored (tc_*, tt_*) arrays are pickled to
`<output_dir>/scored_pairs_cache.pkl` after computing, and reloaded instead
of recomputed on any later run that finds it -- re-running with a different
--mass_cutoffs is then just the cheap filtering + plotting step. Pass
--force_recompute to ignore an existing cache (e.g. after the underlying
intermediates change).

Usage:
    uv run python tools/mces_calibration_plots.py \\
        --intermediates_dir /path/to/008_2_.../retrieval_iceberg \\
        --gt_mces_dir /path/to/gt_mces_retrieval_candidates \\
        --test_to_test_prepro_dir /path/to/preprocessing_msg_exact_mces_1020 \\
        --output_dir /path/to/output \\
        --mces_max_value 40 \\
        --mass_cutoffs 300,500,750,1000
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
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from scipy.stats import spearmanr


_MASS_CACHE: dict[str, float] = {}


def mass_from_smiles(smi: str) -> float:
    """RDKit ExactMolWt, cached by (already-canonical) SMILES string."""
    if smi not in _MASS_CACHE:
        mol = Chem.MolFromSmiles(smi)
        _MASS_CACHE[smi] = ExactMolWt(mol) if mol is not None else float("nan")
    return _MASS_CACHE[smi]


def mae_spearman(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if len(gt) == 0:
        return float("nan"), float("nan")
    mae = float(np.abs(pred - gt).mean())
    rho, _ = spearmanr(pred, gt)
    return mae, float(rho)


def balance_by_gt_bin(
    pred: np.ndarray,
    gt: np.ndarray,
    bin_width: float = 5.0,
    gt_clip_max: float = 40.0,
    target_n_per_bin: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Resamples pred/gt so every non-empty GT bin (0-5, 5-10, ..., up to
    gt_clip_max) contributes exactly target_n_per_bin pairs -- without
    replacement for a bin with enough pairs, WITH replacement (oversampled)
    for a bin with fewer -- so MAE/Spearman isn't dominated by whichever GT
    range happens to have the most pairs (typically the near-0 bin), and
    isn't crippled down to a handful of points by whichever range has the
    fewest (min-bin-count balancing left test-to-candidate with as few as
    91-1936 pairs/bin at some cutoffs). Bins with zero pairs are dropped
    entirely. Returns (balanced_pred, balanced_gt, target_n_per_bin,
    n_bins_used, min_available_in_any_bin) -- that last value tells the
    caller how much oversampling the thinnest bin needed."""
    keep = gt <= gt_clip_max
    pred, gt = pred[keep], gt[keep]
    edges = np.arange(0, gt_clip_max + bin_width, bin_width)
    bin_idx = np.digitize(gt, edges) - 1

    bin_indices = [
        np.flatnonzero(bin_idx == b)
        for b in range(len(edges) - 1)
        if np.count_nonzero(bin_idx == b) > 0
    ]
    if not bin_indices:
        return pred[:0], gt[:0], 0, 0, 0

    rng = np.random.default_rng(seed)
    min_available = min(len(idx) for idx in bin_indices)
    chosen = np.concatenate(
        [
            rng.choice(idx, size=target_n_per_bin, replace=len(idx) < target_n_per_bin)
            for idx in bin_indices
        ]
    )
    return pred[chosen], gt[chosen], target_n_per_bin, len(bin_indices), min_available


def binned_box_on_ax(
    ax,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    bin_width: float = 2.0,
    min_n: int = 5,
    gt_clip_max: float = 40.0,
) -> tuple[int, int, int]:
    """Draws the GT-binned SIMBA-predicted-MCES boxplot onto an existing ax.
    Returns (n_bins, n_kept, n_total) for the caller to log."""
    keep = gt <= gt_clip_max
    pred, gt = pred[keep], gt[keep]
    n_total, n_kept = len(keep), len(gt)

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

    if not groups:
        ax.set_title(f"{title}\n(no bins with >= {min_n} points)")
        ax.axis("off")
        return 0, n_kept, n_total

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
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    return len(groups), n_kept, n_total


def plot_binned_box(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    out_path: Path,
    bin_width: float = 2.0,
    min_n: int = 5,
    gt_clip_max: float = 40.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_bins, n_kept, n_total = binned_box_on_ax(
        ax, pred, gt, title, bin_width, min_n, gt_clip_max
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved {out_path}  ({n_bins} bins with >= {min_n} points, {n_kept:,}/{n_total:,} pairs kept after GT<={gt_clip_max} clip)"
    )


def run(
    intermediates_dir: str,
    gt_mces_dir: str,
    test_to_test_prepro_dir: str,
    output_dir: str,
    mces_max_value: float = 40.0,
    gt_clip_max: float = 40.0,
    mass_cutoffs: str = "300,500,750,1000",
    force_recompute: bool = False,
    gt_balance_target_n: int = 10_000,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "scored_pairs_cache.pkl"

    if cache_path.exists() and not force_recompute:
        print(
            f"Loading cached scored pairs from {cache_path} (pass --force_recompute to ignore) ..."
        )
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        test_smiles = cache["test_smiles"]
        tc_idx_to_smiles = cache["tc_idx_to_smiles"]
        tc_spec_idx = cache["tc_spec_idx"]
        tc_other_idx = cache["tc_other_idx"]
        tc_gt = cache["tc_gt"]
        tc_pred = cache["tc_pred"]
        tt_idx_to_smiles = cache["tt_idx_to_smiles"]
        tt_pred_matrix = cache["tt_pred_matrix"]
        tt_gt_matrix = cache["tt_gt_matrix"]
        tt_valid_mask = cache["tt_valid_mask"]
        tt_spec_molidx = cache["tt_spec_molidx"]
        tt_pred, tt_gt = tt_pred_matrix[tt_valid_mask], tt_gt_matrix[tt_valid_mask]
        print(
            f"  {len(tc_gt):,} test-to-candidate pairs, {len(tt_gt):,} test-to-test pairs (from cache)"
        )
    else:
        inter = Path(intermediates_dir)

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
        tc_spec_idx, tc_other_idx, tc_gt, tc_pred = expand_and_score_ragged(
            tc_pairs,
            tc_idx_to_smiles,
            test_smiles,
            test_adducts,
            test_embeddings,
            cand_smi_adduct_to_emb,
            mces_max_value,
        )
        print(f"  {len(tc_gt):,} scored pairs (incl. self)")

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
        tt_pred_matrix, tt_gt_matrix, tt_valid_mask, tt_spec_molidx = (
            score_test_to_test_no_averaging(
                tt_pairs, tt_idx_to_smiles, test_smiles, test_embeddings, mces_max_value
            )
        )
        tt_pred, tt_gt = tt_pred_matrix[tt_valid_mask], tt_gt_matrix[tt_valid_mask]
        print(f"  {len(tt_gt):,} scored individual-spectrum pairs")

        print(f"Caching scored pairs to {cache_path} ...")
        with open(cache_path, "wb") as fh:
            pickle.dump(
                {
                    "test_smiles": test_smiles,
                    "tc_idx_to_smiles": tc_idx_to_smiles,
                    "tc_spec_idx": tc_spec_idx,
                    "tc_other_idx": tc_other_idx,
                    "tc_gt": tc_gt,
                    "tc_pred": tc_pred,
                    "tt_idx_to_smiles": tt_idx_to_smiles,
                    "tt_pred_matrix": tt_pred_matrix,
                    "tt_gt_matrix": tt_gt_matrix,
                    "tt_valid_mask": tt_valid_mask,
                    "tt_spec_molidx": tt_spec_molidx,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    plot_binned_box(
        tc_pred,
        tc_gt,
        "Test-to-candidate: SIMBA-predicted MCES by GT bin (self included)",
        out_dir / "test_to_candidate_binned_box.png",
        gt_clip_max=gt_clip_max,
    )

    plot_binned_box(
        tt_pred,
        tt_gt,
        "Test-to-test: SIMBA-predicted MCES by GT bin (self spectrum excluded, same-molecule spectra included, no averaging)",
        out_dir / "test_to_test_binned_box.png",
        gt_clip_max=gt_clip_max,
    )

    print("\n--- Mass-restricted comparison (item 8c) ---")
    cutoffs = [float(x) for x in mass_cutoffs.split(",") if x.strip()]
    run_mass_restricted(
        tc_spec_idx,
        tc_other_idx,
        tc_gt,
        tc_pred,
        test_smiles,
        tc_idx_to_smiles,
        tt_pred_matrix,
        tt_gt_matrix,
        tt_valid_mask,
        tt_spec_molidx,
        tt_idx_to_smiles,
        cutoffs,
        gt_clip_max,
        out_dir,
        target_n_per_bin=gt_balance_target_n,
    )


def run_mass_restricted(
    tc_spec_idx: np.ndarray,
    tc_other_idx: np.ndarray,
    tc_gt: np.ndarray,
    tc_pred: np.ndarray,
    test_smiles: list[str],
    tc_idx_to_smiles: list[str],
    tt_pred_matrix: np.ndarray,
    tt_gt_matrix: np.ndarray,
    tt_valid_mask: np.ndarray,
    tt_spec_molidx: np.ndarray,
    tt_idx_to_smiles: list[str],
    cutoffs: list[float],
    gt_clip_max: float,
    out_dir: Path,
    target_n_per_bin: int = 10_000,
) -> None:
    """Re-filters the SAME already-scored pairs (no re-scoring) by
    max(mass_a, mass_b) < cutoff, for a small set of cutoffs plus "no
    limit" (reproducing exactly what the two standalone plots above show),
    to see whether calibration improves when restricted to a mass range
    closer to what training actually saw."""
    print("Computing per-pair molecule masses (RDKit ExactMolWt, cached) ...")
    tc_query_mass = np.array([mass_from_smiles(test_smiles[i]) for i in tc_spec_idx])
    tc_other_mass = np.array(
        [mass_from_smiles(tc_idx_to_smiles[i]) for i in tc_other_idx]
    )
    tc_max_mass = np.maximum(tc_query_mass, tc_other_mass)

    tt_mass_by_spec = np.array(
        [mass_from_smiles(tt_idx_to_smiles[i]) for i in tt_spec_molidx]
    )
    tt_max_mass_matrix = np.maximum(tt_mass_by_spec[:, None], tt_mass_by_spec[None, :])

    labels = [f"< {c:g} Da" for c in cutoffs] + ["no limit"]
    n_rows = len(labels)

    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows), squeeze=False)
    mae_tt, mae_tc, rho_tt, rho_tc = [], [], [], []
    mae_tt_bal, mae_tc_bal, rho_tt_bal, rho_tc_bal = [], [], [], []

    for row, label in enumerate(labels):
        cutoff = cutoffs[row] if row < len(cutoffs) else None

        if cutoff is None:
            tc_mask = np.ones_like(tc_max_mass, dtype=bool)
            tt_mask = tt_valid_mask
        else:
            tc_mask = tc_max_mass < cutoff
            tt_mask = (tt_max_mass_matrix < cutoff) & tt_valid_mask

        tc_pred_f, tc_gt_f = tc_pred[tc_mask], tc_gt[tc_mask]
        tt_pred_f, tt_gt_f = tt_pred_matrix[tt_mask], tt_gt_matrix[tt_mask]

        binned_box_on_ax(
            axes[row][0],
            tt_pred_f,
            tt_gt_f,
            f"test-to-test, {label} (n={len(tt_gt_f):,})",
            gt_clip_max=gt_clip_max,
        )
        binned_box_on_ax(
            axes[row][1],
            tc_pred_f,
            tc_gt_f,
            f"test-to-candidate, {label} (n={len(tc_gt_f):,})",
            gt_clip_max=gt_clip_max,
        )

        tt_mae, tt_rho = mae_spearman(tt_pred_f, tt_gt_f)
        tc_mae, tc_rho = mae_spearman(tc_pred_f, tc_gt_f)
        mae_tt.append(tt_mae)
        rho_tt.append(tt_rho)
        mae_tc.append(tc_mae)
        rho_tc.append(tc_rho)
        print(
            f"  {label}: test-to-test n={len(tt_gt_f):,} mae={tt_mae:.3f} rho={tt_rho:.3f} | "
            f"test-to-candidate n={len(tc_gt_f):,} mae={tc_mae:.3f} rho={tc_rho:.3f}"
        )

        tt_pred_bal, tt_gt_bal, tt_n_per_bin, tt_n_bins, tt_min_avail = (
            balance_by_gt_bin(
                tt_pred_f,
                tt_gt_f,
                gt_clip_max=gt_clip_max,
                target_n_per_bin=target_n_per_bin,
            )
        )
        tc_pred_bal, tc_gt_bal, tc_n_per_bin, tc_n_bins, tc_min_avail = (
            balance_by_gt_bin(
                tc_pred_f,
                tc_gt_f,
                gt_clip_max=gt_clip_max,
                target_n_per_bin=target_n_per_bin,
            )
        )
        tt_mae_bal, tt_rho_bal = mae_spearman(tt_pred_bal, tt_gt_bal)
        tc_mae_bal, tc_rho_bal = mae_spearman(tc_pred_bal, tc_gt_bal)
        mae_tt_bal.append(tt_mae_bal)
        rho_tt_bal.append(tt_rho_bal)
        mae_tc_bal.append(tc_mae_bal)
        rho_tc_bal.append(tc_rho_bal)
        print(
            f"    GT-balanced: test-to-test n={tt_n_per_bin}/bin x {tt_n_bins} bins "
            f"(thinnest bin had {tt_min_avail}, "
            f"{'oversampled ' + f'{target_n_per_bin / tt_min_avail:.1f}x' if tt_min_avail < target_n_per_bin else 'no oversampling'}) "
            f"mae={tt_mae_bal:.3f} rho={tt_rho_bal:.3f} | "
            f"test-to-candidate n={tc_n_per_bin}/bin x {tc_n_bins} bins "
            f"(thinnest bin had {tc_min_avail}, "
            f"{'oversampled ' + f'{target_n_per_bin / tc_min_avail:.1f}x' if tc_min_avail < target_n_per_bin else 'no oversampling'}) "
            f"mae={tc_mae_bal:.3f} rho={tc_rho_bal:.3f}"
        )

    fig.tight_layout()
    out_path = out_dir / "binned_box_by_mass_cutoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    x_pos = list(range(len(cutoffs) + 1))
    x_labels = [f"{c:g}" for c in cutoffs] + ["no limit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x_pos, mae_tt, "o-", label="test-to-test (raw)", color="tab:blue")
    ax1.plot(
        x_pos,
        mae_tt_bal,
        "o--",
        label="test-to-test (GT-balanced)",
        color="tab:blue",
        alpha=0.55,
    )
    ax1.plot(x_pos, mae_tc, "o-", label="test-to-candidate (raw)", color="tab:orange")
    ax1.plot(
        x_pos,
        mae_tc_bal,
        "o--",
        label="test-to-candidate (GT-balanced)",
        color="tab:orange",
        alpha=0.55,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right")
    ax1.set_xlabel("mass cutoff (max mass in pair, Da)")
    ax1.set_ylabel("MAE (SIMBA-predicted vs GT MCES)")
    ax1.set_title("MAE vs mass cutoff")
    ax1.legend(fontsize=8)

    ax2.plot(x_pos, rho_tt, "o-", label="test-to-test (raw)", color="tab:blue")
    ax2.plot(
        x_pos,
        rho_tt_bal,
        "o--",
        label="test-to-test (GT-balanced)",
        color="tab:blue",
        alpha=0.55,
    )
    ax2.plot(x_pos, rho_tc, "o-", label="test-to-candidate (raw)", color="tab:orange")
    ax2.plot(
        x_pos,
        rho_tc_bal,
        "o--",
        label="test-to-candidate (GT-balanced)",
        color="tab:orange",
        alpha=0.55,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right")
    ax2.set_xlabel("mass cutoff (max mass in pair, Da)")
    ax2.set_ylabel("Spearman correlation")
    ax2.set_title("Spearman vs mass cutoff")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "mae_spearman_by_mass_cutoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


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
    p.add_argument(
        "--mass_cutoffs",
        default="300,500,750,1000",
        help="Comma-separated max(mass_a, mass_b) cutoffs (Da); 'no limit' is always added as the last one.",
    )
    p.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore any existing scored_pairs_cache.pkl and recompute from the intermediates.",
    )
    p.add_argument(
        "--gt_balance_target_n",
        type=int,
        default=10_000,
        help="Target pairs per GT bin for the balanced MAE/Spearman overlay -- oversampled (with replacement) if a bin has fewer.",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
