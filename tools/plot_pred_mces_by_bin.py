"""Predicted-MCES distribution per GT-MCES bin, overlaid.

Diagnostic for telling apart "systematically shifted" vs. "genuinely
diffuse" separation failures -- e.g. why does raw cosine similarity
separate self-pairs from near-neighbors better than SIMBA's own
predictions (tools/compute_val_cosine.py), even when SIMBA wins on broader
overlap metrics? A self-pair distribution that's centered near 0 but just
wide/overlapping with its neighbors points at a resolution/architectural
limit; one that's clearly shifted away from 0 points at a calibration/bias
problem instead (e.g. training-time augmentation teaching the model that
"the same molecule" doesn't have to score near 0).

Reads one experiment's most recent validation check directly from its
consolidated parquet (mces_bin + that step's pred_mces_stepNNNNNN column) --
same file tools/dashboard_app.py reads, no SIMBA re-inference needed.
Lightweight (a few columns from an existing file); run directly, no SLURM
needed.

Optionally pass --cosine_parquet (tools/compute_val_cosine.py's output) to
add a second panel showing the same bin split on raw cosine similarity, for
a direct side-by-side comparison of how each scoring method separates the
same populations.

Usage:
    uv run python tools/plot_pred_mces_by_bin.py \\
        --exp_dir /path/to/experiments/training/012_..._1gpu \\
        --val_name val \\
        --cosine_parquet /path/to/preprocessing_dir/val_cosine_val.parquet \\
        --bins "self (MCES=0)" "(0,5]" "(5,10]" "(10,15]"
"""

import argparse
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_PARQUET_PRED_COL_RE = re.compile(r"^pred_mces_step(\d+)$")
DEFAULT_BINS = ["self (MCES=0)", "(0,5]", "(5,10]", "(10,15]"]
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def latest_step(consolidated_path: Path) -> int:
    steps = [
        int(m.group(1))
        for name in pq.ParquetFile(consolidated_path).schema.names
        if (m := _PARQUET_PRED_COL_RE.match(name))
    ]
    if not steps:
        raise ValueError(f"No pred_mces_step* columns found in {consolidated_path}")
    return max(steps)


def split_by_bin(
    df: pd.DataFrame, bins: list[str], value_col: str, split_self: bool
) -> list[tuple[str, np.ndarray]]:
    """[(label, values)] for each requested bin -- splits 'self (MCES=0)'
    into true same-spectrum vs same-molecule-different-spectrum pairs when
    split_self is set and spec_idx_0/1 are present."""
    series = []
    for label in bins:
        if label == "self (MCES=0)" and split_self and "spec_idx_0" in df.columns:
            self_df = df.loc[df["mces_bin"] == label]
            same_spec = self_df["spec_idx_0"] == self_df["spec_idx_1"]
            series.append(
                ("self, same spectrum", self_df.loc[same_spec, value_col].to_numpy())
            )
            series.append(
                (
                    "self, different spectrum",
                    self_df.loc[~same_spec, value_col].to_numpy(),
                )
            )
        else:
            series.append(
                (label, df.loc[df["mces_bin"] == label, value_col].to_numpy())
            )
    return series


def plot_series(
    ax, series: list[tuple[str, np.ndarray]], value_range, xlabel: str
) -> list[str]:
    stats_lines = []
    for i, (label, vals) in enumerate(series):
        if len(vals) == 0:
            print(f"  (no pairs for {label!r}, skipping)")
            continue
        ax.hist(
            vals,
            bins=80,
            range=value_range,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"{label} (n={len(vals):,})",
            color=_TAB10[i % len(_TAB10)],
        )
        line = (
            f"{label}: median={np.median(vals):.3f}, mean={vals.mean():.3f}, "
            f"std={vals.std():.3f}, n={len(vals):,}"
        )
        stats_lines.append(line)
        print(f"  {line}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    return stats_lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--val_name", default="val")
    parser.add_argument("--bins", nargs="+", default=DEFAULT_BINS)
    parser.add_argument(
        "--split_self_by_spectrum",
        action="store_true",
        default=True,
        help="If 'self (MCES=0)' is among --bins, additionally split it into "
        "true same-spectrum pairs (spec_idx_0==spec_idx_1, literally "
        "identical measurement) vs same-molecule-different-spectrum pairs -- "
        "these can behave very differently (see module docstring).",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Defaults to the most recent check."
    )
    parser.add_argument(
        "--cosine_parquet",
        default=None,
        help="tools/compute_val_cosine.py's output -- if given, adds a second "
        "panel showing the same bin split scored by raw cosine similarity "
        "instead of pred_mces, for a direct side-by-side comparison.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    consolidated_path = exp_dir / f"val_pairs_{args.val_name}_consolidated.parquet"
    step = args.step or latest_step(consolidated_path)
    pred_col = f"pred_mces_step{step:06d}"
    print(f"Reading step {step} from {consolidated_path} ...")
    needs_spec_idx = args.split_self_by_spectrum and "self (MCES=0)" in args.bins
    columns = ["mces_bin", pred_col]
    if needs_spec_idx or args.cosine_parquet:
        columns += ["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"]
    df = pd.read_parquet(consolidated_path, columns=list(dict.fromkeys(columns)))
    df = df.rename(columns={pred_col: "pred_mces"})

    ncols = 2 if args.cosine_parquet else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6))
    axes = np.atleast_1d(axes)

    print("Predicted MCES (SIMBA):")
    series = split_by_bin(df, args.bins, "pred_mces", args.split_self_by_spectrum)
    plot_series(axes[0], series, (0, 40), "Predicted MCES")
    axes[0].set_title("SIMBA prediction")

    if args.cosine_parquet:
        print("\nCosine similarity (raw spectral baseline):")
        cos = pd.read_parquet(args.cosine_parquet)
        merged = df.merge(
            cos, on=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"], how="inner"
        )
        cos_series = split_by_bin(
            merged, args.bins, "cosine", args.split_self_by_spectrum
        )
        plot_series(axes[1], cos_series, (0, 1), "Cosine similarity")
        axes[1].set_title("Cosine baseline")

    fig.suptitle(f"{exp_dir.name}\nScore distribution by GT-MCES bin (step {step})")
    fig.tight_layout()

    output = Path(args.output or exp_dir / f"pred_mces_by_bin_step{step:06d}.png")
    fig.savefig(output, dpi=150)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
