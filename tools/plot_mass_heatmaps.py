"""8d: mass1 x mass2 heatmaps -- how does calibration/task-difficulty vary
with the (mass, mass) of a pair, for both test-to-test and test-to-candidate.

One combined figure, 2 rows (population) x 4 columns (color metric), each
subplot its own heatmap with its own colorbar (the 4 metrics have
different natural scales, so a shared colorbar wouldn't make sense):

  - population (rows): test-to-test (axes = min(mass_a, mass_b) / max(mass_a,
    mass_b), since pairs are unordered -- avoids a redundant symmetric
    square) and test-to-candidate (axes = query mass / candidate mass).
    Candidates come from a formula-matched pool, so query and candidate
    mass coincide almost exactly (checked earlier: shared to ~1e-13 Da for
    the true candidate, and confirmed visually -- the heatmap concentrates
    on the diagonal) -- built anyway to show that directly.
  - color metric (columns): MAE (mean |pred-GT|), signed bias
    (mean(pred-GT), so over- vs under-prediction is visible -- MAE alone
    hides direction), mean GT MCES (task difficulty alone, independent of
    SIMBA), and per-cell Spearman correlation (SIMBA-predicted vs GT MCES,
    within that cell only).

Fixed-width bins only (--step Da, default 100) -- coarse enough that every
occupied cell can be labeled directly with its value and n. Cells below
--min_n (or with too few points for a meaningful Spearman, < 2) are left
blank.

Reuses mces_calibration_plots.py's scored_pairs_cache.pkl (same already-
scored (pred, GT) pairs, no re-scoring/re-embedding) -- run that script
first if the cache doesn't exist yet.

Usage:
    uv run python tools/plot_mass_heatmaps.py \\
        --cache_path /path/to/scored_pairs_cache.pkl \\
        --output_dir /path/to/output \\
        --gt_clip_max 40 \\
        --step 100 \\
        --min_n 1000
"""

import argparse
import pickle
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mces_calibration_plots import mass_from_smiles
from scipy.stats import spearmanr


# (key, title, cmap, center_zero, fixed_vlim) -- fixed_vlim overrides a
# data-driven +/-vlim (e.g. Spearman is naturally bounded to [-1, 1]).
METRICS = [
    ("mae", "MAE (|pred - GT MCES|)", "viridis", False, None),
    ("bias", "Signed bias (pred - GT MCES)", "coolwarm", True, None),
    ("gt_mces", "Mean GT MCES", "viridis", False, None),
    ("spearman", "Spearman (pred vs GT), per cell", "coolwarm", True, 1.0),
]


def format_n(n: int) -> str:
    """Compact cell-count label: 234,861,942 -> '235M', 79,180 -> '79K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def fixed_edges(values: np.ndarray, step: float) -> np.ndarray:
    hi = np.ceil(float(values.max()) / step) * step
    return np.arange(0, hi + step, step)


def compute_cell_stats(
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Bins (x, y) into a (nx, ny) grid and computes, per cell: MAE, signed
    bias, mean GT MCES (all via np.bincount -- fast even at ~280M points)
    and Spearman correlation (needs per-cell ranking, so pred/gt are
    sorted once by cell and split at the boundaries rather than looping a
    boolean mask per cell, which would be O(n_cells * n) instead of
    O(n log n)). Returns ({metric: grid (nx, ny), NaN where empty/too
    small}, count_grid (nx, ny))."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    flat_idx = xi * ny + yi
    minlen = nx * ny

    count = np.bincount(flat_idx, minlength=minlen).astype(np.int64)
    count_grid = count.reshape(nx, ny)

    grids = {}
    for name, vals in [
        ("mae", np.abs(pred - gt)),
        ("bias", pred - gt),
        ("gt_mces", gt),
    ]:
        sums = np.bincount(flat_idx, weights=vals, minlength=minlen)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = (sums / count).reshape(nx, ny)
        grid[count_grid == 0] = np.nan
        grids[name] = grid

    order = np.argsort(flat_idx, kind="stable")
    sorted_idx = flat_idx[order]
    pred_sorted, gt_sorted = pred[order], gt[order]
    boundaries = np.flatnonzero(np.diff(sorted_idx)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(sorted_idx)]])

    rho_grid = np.full(minlen, np.nan)
    for s, e in zip(starts, ends):
        if e - s >= 2:
            rho, _ = spearmanr(pred_sorted[s:e], gt_sorted[s:e])
            rho_grid[sorted_idx[s]] = rho
    grids["spearman"] = rho_grid.reshape(nx, ny)

    return grids, count_grid


def plot_heatmap_on_ax(
    fig,
    ax,
    grid: np.ndarray,
    count_grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
    min_n: int = 1_000,
    center_zero: bool = False,
    fixed_vlim: float | None = None,
) -> int:
    """Draws one heatmap (with its own colorbar) onto ax, annotating every
    occupied cell with its value and n. Returns the number of occupied
    cells, for the caller to log."""
    ok = (count_grid >= min_n) & ~np.isnan(grid)
    plot_grid = np.ma.masked_where(~ok, grid)

    vmin = vmax = None
    if center_zero:
        if fixed_vlim is not None:
            vmin, vmax = -fixed_vlim, fixed_vlim
        else:
            finite = plot_grid[~np.ma.getmaskarray(plot_grid)]
            vlim = float(np.abs(finite).max()) if finite.size else 1.0
            vmin, vmax = -vlim, vlim

    im = ax.pcolormesh(
        x_edges, y_edges, plot_grid.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
    )
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    stroke = [pe.withStroke(linewidth=1.5, foreground="black")]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if ok[i, j]:
                ax.text(
                    x_centers[i],
                    y_centers[j],
                    f"{grid[i, j]:.2f}\nn={format_n(int(count_grid[i, j]))}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                    path_effects=stroke,
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return int(ok.sum())


def run(
    cache_path: str,
    output_dir: str,
    gt_clip_max: float = 40.0,
    step: float = 100.0,
    min_n: int = 1_000,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cache_path} ...")
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
    print(
        f"  {len(tc_gt):,} test-to-candidate pairs, "
        f"{int(tt_valid_mask.sum()):,} test-to-test pairs"
    )

    print("Computing per-molecule masses (RDKit ExactMolWt, cached) ...")
    tc_query_mass = np.array([mass_from_smiles(test_smiles[i]) for i in tc_spec_idx])
    tc_other_mass = np.array(
        [mass_from_smiles(tc_idx_to_smiles[i]) for i in tc_other_idx]
    )

    tt_mass_by_spec = np.array(
        [mass_from_smiles(tt_idx_to_smiles[i]) for i in tt_spec_molidx]
    )
    tt_mass_a = np.broadcast_to(tt_mass_by_spec[:, None], tt_valid_mask.shape)
    tt_mass_b = np.broadcast_to(tt_mass_by_spec[None, :], tt_valid_mask.shape)

    print("Filtering by GT clip + validity, flattening ...")
    tc_keep = tc_gt <= gt_clip_max
    tc_x, tc_y = tc_query_mass[tc_keep], tc_other_mass[tc_keep]
    tc_pred_k, tc_gt_k = tc_pred[tc_keep], tc_gt[tc_keep]

    tt_keep = tt_valid_mask & (tt_gt_matrix <= gt_clip_max)
    tt_a, tt_b = tt_mass_a[tt_keep], tt_mass_b[tt_keep]
    tt_x, tt_y = np.minimum(tt_a, tt_b), np.maximum(tt_a, tt_b)
    tt_pred_k, tt_gt_k = tt_pred_matrix[tt_keep], tt_gt_matrix[tt_keep]

    populations = [
        (
            "test_to_test",
            {
                "x": tt_x,
                "y": tt_y,
                "pred": tt_pred_k,
                "gt": tt_gt_k,
                "xlabel": "min(mass_a, mass_b) (Da)",
                "ylabel": "max(mass_a, mass_b) (Da)",
            },
        ),
        (
            "test_to_candidate",
            {
                "x": tc_x,
                "y": tc_y,
                "pred": tc_pred_k,
                "gt": tc_gt_k,
                "xlabel": "query mass (Da)",
                "ylabel": "candidate mass (Da)",
            },
        ),
    ]

    fig, axes = plt.subplots(2, len(METRICS), figsize=(6.5 * len(METRICS), 13))

    for row, (pop_name, pop) in enumerate(populations):
        x, y, pred, gt = pop["x"], pop["y"], pop["pred"], pop["gt"]
        print(f"\n=== {pop_name} ({len(gt):,} pairs after GT<={gt_clip_max} clip) ===")

        x_edges, y_edges = fixed_edges(x, step), fixed_edges(y, step)
        print(f"  {len(x_edges) - 1} x {len(y_edges) - 1} cells (step={step} Da)")
        grids, count = compute_cell_stats(x, y, pred, gt, x_edges, y_edges)

        for col, (metric_key, title, cmap, center_zero, fixed_vlim) in enumerate(
            METRICS
        ):
            n_ok = plot_heatmap_on_ax(
                fig,
                axes[row, col],
                grids[metric_key],
                count,
                x_edges,
                y_edges,
                pop["xlabel"],
                pop["ylabel"],
                f"{pop_name}\n{title}",
                cmap=cmap,
                min_n=min_n,
                center_zero=center_zero,
                fixed_vlim=fixed_vlim,
            )
            print(f"  {metric_key}: {n_ok}/{count.size} cells with >= {min_n}")

    fig.suptitle(
        f"Mass1 x mass2 heatmaps (step={step:g} Da, min_n={min_n})", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "mass_heatmaps.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gt_clip_max", type=float, default=40.0)
    p.add_argument("--step", type=float, default=100.0)
    p.add_argument("--min_n", type=int, default=1_000)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
