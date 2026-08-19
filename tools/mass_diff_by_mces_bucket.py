"""Molecule mass-difference distribution per MCES sampling bucket.

Diagnostic for calibrating a future mass-difference-based sampling weight on
top of the MCES-bucket inverse-frequency scheme in
simba/workflows/training.py's prepare_data (use_mces_sampling branch, see
MCES_SAMPLING_EDGES/MCES_SAMPLING_BIN_LABELS there).

Loads the real training pair pool exactly as `simba train` would build it
(same load_dataset + prepare_data call, same relevant config overrides), so
pair_distances/extra_distances reflect the actual post-remap, post-exact-
MCES-filter, post-identity-pairs pool the weighted sampler operates over --
not a hand-rolled reimplementation of that pipeline. Buckets pairs by the
same MCES_SAMPLING_EDGES scheme used for sampling weights, computes each
pair's molecule mass difference (RDKit ExactMolWt, looked up once per unique
molecule), and plots one histogram subplot per bucket in a single figure.

This does the same preprocessing work (spectrum preprocessing, spectra
loading) a real training run's data-prep phase does, so it's CPU/IO-bound
similarly -- run via the accompanying SLURM script for uncontended storage
I/O rather than directly on the login node.

Usage:
    uv run python tools/mass_diff_by_mces_bucket.py \\
        --preprocessing_dir /path/to/preprocessing_gaetan_split_max_lb_hdf5_v2 \\
        --mgf_path /path/to/MassSpecGym.mgf \\
        --output /path/to/mass_diff_by_mces_bucket.png
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir

from simba.core.chemistry.chem_utils import mass_lookup_from_df_smiles
from simba.utils.config_utils import get_config_path
from simba.workflows.training import (
    MCES_SAMPLING_BIN_LABELS,
    MCES_SAMPLING_EDGES,
    load_dataset,
    prepare_data,
)


DEFAULT_PREPRO_DIR = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "preprocessing_gaetan_split_max_lb_hdf5_v2"
)
DEFAULT_MGF = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "data/auxiliary/MassSpecGym.mgf"
)
DEFAULT_OUTPUT = (
    "/sofia/projects/2026_053/simba_project/experiments/"
    "mass_diff_by_mces_bucket_gaetan_split_v2.png"
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preprocessing_dir", default=DEFAULT_PREPRO_DIR)
    parser.add_argument("--mgf_path", default=DEFAULT_MGF)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Same overrides as the 010/011 SLURM scripts, restricted to what affects
    # the training pair pool's construction (identity pairs, ED head off so
    # use_mces_sampling's branch is taken, same as those real runs).
    overrides = [
        f"paths.preprocessing_dir={args.preprocessing_dir}",
        f"paths.preprocessing_dir_train={args.preprocessing_dir}",
        "paths.preprocessing_pickle_file=mapping.pkl",
        f"paths.mgf_path={args.mgf_path}",
        "sampling.add_identity_pairs=true",
        "model.tasks.edit_distance.enabled=false",
    ]
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

    print("Loading dataset ...")
    (
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_val_official,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    ) = load_dataset(cfg)

    print("Running prepare_data (builds the real training pair pool) ...")
    prepare_data(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
        cfg,
        molecule_pairs_val_official=molecule_pairs_val_official,
    )

    pair_distances = molecule_pairs_train.pair_distances
    extra_distances = molecule_pairs_train.extra_distances
    mol_idx_0 = pair_distances[:, 0].astype(int)
    mol_idx_1 = pair_distances[:, 1].astype(int)
    mces_raw = (1.0 - extra_distances) * 40.0

    print(
        f"Computing molecule masses (RDKit ExactMolWt) for "
        f"{len(molecule_pairs_train.df_smiles):,} unique molecules ..."
    )
    masses = mass_lookup_from_df_smiles(molecule_pairs_train.df_smiles)
    mass_diff = np.abs(masses[mol_idx_0] - masses[mol_idx_1])

    bin_idx = np.clip(
        np.searchsorted(MCES_SAMPLING_EDGES, mces_raw).astype(int),
        0,
        len(MCES_SAMPLING_BIN_LABELS) - 1,
    )

    n_labels = len(MCES_SAMPLING_BIN_LABELS)
    ncols = 4
    nrows = -(-n_labels // ncols)  # ceil div
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, label in enumerate(MCES_SAMPLING_BIN_LABELS):
        ax = axes[i]
        vals = mass_diff[bin_idx == i]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ax.set_title(f"{label} (n=0)")
            ax.axis("off")
            continue
        ax.hist(vals, bins=60, color="#1f77b4")
        ax.set_title(f"{label} (n={len(vals):,})")
        ax.set_xlabel("|mass_0 - mass_1| (Da)")
        ax.set_ylabel("count")
        ax.text(
            0.98,
            0.95,
            f"median={np.median(vals):.1f}\nmean={vals.mean():.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    for j in range(n_labels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Molecule mass-difference distribution per MCES sampling bucket "
        "(real training pair pool)"
    )
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
