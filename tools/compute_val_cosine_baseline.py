"""Raw spectral cosine similarity baseline for every validation pair, plus
the same overlap-coefficient/Hit@k summary metrics real training logs --
logged into a separate "cosine" TensorBoard run so it overlays as a
constant reference line on the real run's val_overlap*/val_hit_at* charts.

Uses each spectrum's RAW peaks (before SIMBA's own Preprocessor runs) and
the same binning/L2-normalize recipe as tools/cosine_baseline_iceberg.py on
exp/mces-pipeline, so results match that project's "cosine baseline"
convention. Reuses the exact overlap/Hit@k functions from
simba.core.training.callbacks (no reimplementation) so the numbers are
directly comparable to what training logs for the real model.

Usage:
    uv run python tools/compute_val_cosine_baseline.py \\
        --tb_logdir /path/to/experiment_checkpoint_dir
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from hydra import compose, initialize_config_dir
from sklearn.preprocessing import normalize
from tqdm import tqdm

from simba.core.training.callbacks import (
    _bin_index,
    _bin_labels,
    _build_pool_and_queries,
    _build_score_matrix,
    _hit_at_k,
    _local_index_lookup,
    _overlap_metrics,
    _true_match_scores,
)
from simba.utils.config_utils import get_config_path
from simba.workflows.training import load_dataset, prepare_data


# Same convention as exp/mces-pipeline's tools/cosine_baseline_iceberg.py /
# compute_val_cosine.py -- kept identical so "cosine baseline" means the
# same thing in both places.
BIN_WIDTH = 0.01
MAX_MZ = 1100.0

DEFAULT_PREPRO_DIR = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "preprocessing_gaetan_split_max_lb_hdf5_v2"
)
DEFAULT_MGF = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "data/auxiliary/MassSpecGym.mgf"
)


def bin_spectra(spectra, bin_width: float, max_mz: float) -> sp.csr_matrix:
    """Bin (mz, intensity) peak lists onto a fixed grid, sqrt-compress,
    L2-normalize. One sparse row per spectrum; a plain dot product between
    two rows is then exactly their cosine similarity."""
    n_bins = int(np.ceil(max_mz / bin_width)) + 1
    rows, cols, data = [], [], []
    for i, spec in enumerate(tqdm(spectra, desc="Binning spectra")):
        mz = np.asarray(spec.mz, dtype=np.float64)
        intensity = np.asarray(spec.intensity, dtype=np.float64)
        keep = mz <= max_mz
        if not keep.all():
            mz, intensity = mz[keep], intensity[keep]
        if len(mz) == 0:
            continue
        bins = np.clip(np.round(mz / bin_width).astype(np.int64), 0, n_bins - 1)
        rows.append(np.full(len(bins), i, dtype=np.int64))
        cols.append(bins)
        data.append(intensity)

    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    data = np.concatenate(data) if data else np.zeros(0, dtype=np.float64)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(len(spectra), n_bins)).tocsr()
    mat.data = np.sqrt(mat.data)
    return normalize(mat, norm="l2", axis=1, copy=False)


def resolve_val_pairs(molecule_pairs_val):
    """(mol_idx_0, mol_idx_1, spec_idx_0, spec_idx_1, gt_sim) for every
    validation pair. spec_idx resolution matches
    CustomDatasetMultitasking.__getitem__'s non-training branch: first
    spectrum for side 0, last spectrum for side 1."""
    mol_idx_0 = molecule_pairs_val.pair_distances[:, 0].astype(np.int64)
    mol_idx_1 = molecule_pairs_val.pair_distances[:, 1].astype(np.int64)
    gt_sim = molecule_pairs_val.extra_distances.astype(np.float64)

    df_smiles = molecule_pairs_val.df_smiles
    n_mol = len(df_smiles)
    first_spec = np.array([df_smiles.loc[i, "indexes"][0] for i in range(n_mol)])
    last_spec = np.array([df_smiles.loc[i, "indexes"][-1] for i in range(n_mol)])
    spec_idx_0 = first_spec[mol_idx_0]
    spec_idx_1 = last_spec[mol_idx_1]
    return mol_idx_0, mol_idx_1, spec_idx_0, spec_idx_1, gt_sim


def log_cosine_baseline_to_tensorboard(
    checkpoint_dir, pairwise, skip_avg, hits, max_step: int
) -> None:
    """Writes the cosine baseline as a constant value at two steps (0 and
    max_step) so it renders as a flat horizontal reference line. max_step
    should be the real run's own total step count -- padding it far beyond
    that (e.g. a huge sentinel) forces every chart's x-axis to auto-scale
    to that range too, squashing the real run's curve into a sliver. Logged
    under both the raw and CORN-corrected tag families -- cosine has no
    CORN-corrected variant of its own, so the same value goes to both."""
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(
        save_dir=str(checkpoint_dir), name="tb_logs", version="cosine"
    )
    steps = (0, max_step)
    for prefix in ("val_overlap", "val_overlap_corrected"):
        for suffix, v in pairwise.items():
            for step in steps:
                logger.experiment.add_scalar(f"{prefix}/{suffix}", v, global_step=step)
        for skip, v in skip_avg.items():
            for step in steps:
                logger.experiment.add_scalar(
                    f"{prefix}_avg/skip{skip}", v, global_step=step
                )
    for prefix in ("val_hit_at", "val_hit_at_corrected"):
        for k, v in hits.items():
            for step in steps:
                logger.experiment.add_scalar(f"{prefix}_{k}", v, global_step=step)
    logger.experiment.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preprocessing_dir", default=DEFAULT_PREPRO_DIR)
    parser.add_argument("--mgf_path", default=DEFAULT_MGF)
    parser.add_argument(
        "--tb_logdir",
        required=True,
        help="An experiment's checkpoint_dir (e.g. .../full_run_diagnostics) -- "
        "the cosine baseline is logged as a sibling TensorBoard run under "
        "<tb_logdir>/tb_logs/cosine so it overlays the real run's charts.",
    )
    parser.add_argument("--mces_max_value", type=float, default=40.0)
    parser.add_argument(
        "--mae_bin_edges",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 35, 40],
    )
    parser.add_argument("--max_skip", type=int, default=2)
    parser.add_argument("--hit_at_k_n_decoys", type=int, default=255)
    parser.add_argument("--hit_at_k_ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Step at which the flat reference line's second point is "
        "logged -- should be close to the real run's own total step count "
        "so TensorBoard's x-axis doesn't stretch to accommodate it. "
        "Defaults to training.epochs * training.limit_train_batches from "
        "the composed config.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["sampling.add_identity_pairs=true"],
        help="Extra Hydra overrides for dataset construction (which pairs "
        "exist in validation) -- doesn't affect the cosine computation "
        "itself, which always uses raw, unpreprocessed spectra.",
    )
    args = parser.parse_args()

    overrides = [
        f"paths.preprocessing_dir={args.preprocessing_dir}",
        f"paths.preprocessing_dir_train={args.preprocessing_dir}",
        "paths.preprocessing_pickle_file=mapping.pkl",
        f"paths.mgf_path={args.mgf_path}",
        *args.overrides,
    ]
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

    max_step = args.max_step
    if max_step is None:
        max_step = int(cfg.training.epochs) * int(cfg.training.limit_train_batches)
    print(f"Reference line will span step 0 to {max_step:,}")

    print("Loading dataset ...")
    molecule_pairs_train, molecule_pairs_val, molecule_pairs_test, uniformed_test = (
        load_dataset(cfg)
    )
    prepare_data(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_test,
        cfg,
    )

    print("Resolving validation pair identity ...")
    mol_idx_0, mol_idx_1, spec_idx_0, spec_idx_1, gt_sim = resolve_val_pairs(
        molecule_pairs_val
    )
    print(f"{len(mol_idx_0):,} validation pairs")

    spectra = molecule_pairs_val.original_spectra
    print(f"Binning {len(spectra):,} raw validation spectra ...")
    mat = bin_spectra(spectra, BIN_WIDTH, MAX_MZ)

    print("Computing cosine similarity ...")
    cosine = np.asarray(mat[spec_idx_0].multiply(mat[spec_idx_1]).sum(axis=1)).flatten()

    out = pd.DataFrame(
        {
            "mol_idx_0": mol_idx_0,
            "mol_idx_1": mol_idx_1,
            "spec_idx_0": spec_idx_0,
            "spec_idx_1": spec_idx_1,
            "cosine": cosine,
        }
    )
    parquet_path = Path(args.preprocessing_dir) / "val_cosine_val.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"Saved {len(out):,} rows to {parquet_path}")

    print("Computing overlap/Hit@k summary metrics ...")
    gt_mces = (1.0 - gt_sim) * args.mces_max_value
    is_self = mol_idx_0 == mol_idx_1
    bin_edges = np.array(args.mae_bin_edges, dtype=float)
    labels = _bin_labels(bin_edges)
    bin_idx = _bin_index(gt_mces, is_self, bin_edges)
    pairwise, skip_avg = _overlap_metrics(cosine, bin_idx, labels, args.max_skip)

    pool_mols, query_mols = _build_pool_and_queries(
        mol_idx_0, spec_idx_0, spec_idx_1, is_self
    )
    local_of = _local_index_lookup(pool_mols)
    gt_matrix = _build_score_matrix(mol_idx_0, mol_idx_1, gt_mces, pool_mols, local_of)
    score_matrix = _build_score_matrix(
        mol_idx_0, mol_idx_1, cosine, pool_mols, local_of
    )
    true_scores = _true_match_scores(mol_idx_0, spec_idx_0, spec_idx_1, is_self, cosine)
    hits = _hit_at_k(
        gt_matrix,
        score_matrix,
        true_scores,
        local_of,
        query_mols,
        args.hit_at_k_n_decoys,
        tuple(args.hit_at_k_ks),
        higher_is_better=True,
    )

    print("Overlap (skip averages):")
    for skip, v in skip_avg.items():
        print(f"  skip{skip}: {v:.4f}")
    print("Hit@k:")
    for k, v in hits.items():
        print(f"  Hit@{k}: {v:.4f}")

    print(f"Logging to TensorBoard under {args.tb_logdir}/tb_logs/cosine ...")
    log_cosine_baseline_to_tensorboard(
        args.tb_logdir, pairwise, skip_avg, hits, max_step
    )
    print("Done.")


if __name__ == "__main__":
    main()
