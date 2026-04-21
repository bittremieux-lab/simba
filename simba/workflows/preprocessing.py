"""Preprocessing workflow for SIMBA.

This module contains the main preprocessing logic for MS/MS spectral data.
It computes structural similarity metrics and prepares data for training.
"""

import contextlib
import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn", force=True)

from simba.core.chemistry.edit_distance.edit_distance import (
    load_precomputed_distances_cache,
)
from simba.core.chemistry.mces.fast_compute import compute_pairs_for_split
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger
from simba.workflows.utils import load_spectra


def write_data(
    file_path: str,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    mgf_path: str = None,
):
    """Write preprocessed data to pickle file.

    Args:
        file_path: Path to output pickle file
        molecule_pairs_train: Training molecule pairs
        molecule_pairs_val: Validation molecule pairs
        molecule_pairs_test: Test molecule pairs
        mgf_path: Path to original MGF file
    """
    # Save only df_smiles, original MGF indexes, and mgf path
    # Spectra will be loaded at training time from mgf file using absolute indexes
    dataset = {
        "df_smiles_train": (
            molecule_pairs_train.df_smiles if molecule_pairs_train is not None else None
        ),
        "df_smiles_val": (
            molecule_pairs_val.df_smiles if molecule_pairs_val is not None else None
        ),
        "df_smiles_test": (
            molecule_pairs_test.df_smiles if molecule_pairs_test is not None else None
        ),
        "spectrum_indexes_train": (
            [s.mgf_index for s in molecule_pairs_train.original_spectra]
            if molecule_pairs_train is not None
            else None
        ),
        "spectrum_indexes_val": (
            [s.mgf_index for s in molecule_pairs_val.original_spectra]
            if molecule_pairs_val is not None
            else None
        ),
        "spectrum_indexes_test": (
            [s.mgf_index for s in molecule_pairs_test.original_spectra]
            if molecule_pairs_test is not None
            else None
        ),
        "mgf_path": mgf_path,
        "format_version": "lightweight",
    }

    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)


def preprocess(cfg: DictConfig) -> None:
    """Run the full preprocessing pipeline.

    This function:
    1. Loads spectra from MGF file
    2. Splits data into train/val/test sets
    3. Computes edit distance and MCES between molecules
    4. Saves preprocessed data to disk

    Args:
        cfg: Hydra configuration object with preprocessing parameters
    """
    logger.info("Starting SIMBA preprocessing workflow...")

    workspace = Path(cfg.paths.preprocessing_dir)

    # Create workspace directory
    workspace.mkdir(parents=True, exist_ok=True)

    # Clean workspace if overwrite is enabled
    if cfg.preprocessing.overwrite and workspace.exists():
        logger.info("Removing existing preprocessing files...")
        for file in workspace.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting file {file}: {e}")

    # Load and preprocess spectra
    logger.info(f"Loading spectra from {cfg.paths.spectra_path}...")
    # Use max_spectra_load if specified, otherwise load all spectra (-1)
    n_samples = getattr(cfg.preprocessing, "max_spectra_load", -1)
    logger.info(
        f"Loading up to {n_samples if n_samples > 0 else 'all'} spectra from MGF file..."
    )
    all_spectra = load_spectra(
        str(cfg.paths.spectra_path),
        cfg,
        n_samples=n_samples,
        use_gnps_format=False,
        use_only_protonized_adducts=cfg.preprocessing.use_only_protonized_adducts,
    )

    unique_molecules_total = len({s.params.get("smiles", "N/A") for s in all_spectra})
    logger.info(
        f"Loaded {len(all_spectra)} spectra with {unique_molecules_total} unique molecules"
    )

    # Split data into train, validation, and test sets
    logger.info("Splitting data into train/val/test sets...")
    all_spectra_train, all_spectra_val, all_spectra_test = (
        TrainUtils.train_val_test_split_bms(
            all_spectra,
            n_buckets=cfg.preprocessing.n_buckets,
            train_buckets=list(cfg.preprocessing.train_buckets),
            val_buckets=list(cfg.preprocessing.val_buckets),
            test_buckets=list(cfg.preprocessing.test_buckets),
        )
    )

    # Limit to max spectra
    all_spectra_train = all_spectra_train[0 : cfg.preprocessing.max_spectra_train]
    all_spectra_val = all_spectra_val[0 : cfg.preprocessing.max_spectra_val]
    all_spectra_test = all_spectra_test[0 : cfg.preprocessing.max_spectra_test]

    logger.info(
        f"Split sizes - Train: {len(all_spectra_train)}, "
        f"Val: {len(all_spectra_val)}, Test: {len(all_spectra_test)}"
    )

    # Log unique molecules per split
    train_unique = len({s.params.get("smiles", "N/A") for s in all_spectra_train})
    val_unique = len({s.params.get("smiles", "N/A") for s in all_spectra_val})
    test_unique = len({s.params.get("smiles", "N/A") for s in all_spectra_test})
    logger.info(
        f"Unique molecules per split - Train: {train_unique}, Val: {val_unique}, Test: {test_unique}"
    )

    logger.info("Pre-computing SMILES mappings for early save...")
    molecule_pairs_metadata = {}
    for type_data, spectra in [
        ("_train", all_spectra_train),
        ("_val", all_spectra_val),
        ("_test", all_spectra_test),
    ]:
        spectra_unique, df_smiles = TrainUtils.get_unique_spectra(spectra)
        molecule_pairs_metadata[type_data] = {
            "df_smiles": df_smiles,
            "spectra_unique": spectra_unique,
            "original_spectra": spectra,
        }

        # Log statistics about molecule sizes
        from rdkit import Chem

        large_mols = []
        for smiles in df_smiles["canon_smiles"]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetNumAtoms() > 40:
                    large_mols.append((smiles, mol.GetNumAtoms()))
            except Exception as e:
                logger.warning(f"Error processing SMILES '{smiles}': {e}")
                pass

        if large_mols:
            logger.info(
                f"  {type_data[1:]} set: {len(large_mols)} molecules with >40 atoms "
                f"(will produce NaN in distance calculations)"
            )
            if len(large_mols) <= 5:
                for smiles, natoms in large_mols:
                    logger.debug(
                        f"    Large molecule: {natoms} atoms - {smiles[:60]}..."
                    )

    output_file = workspace / cfg.paths.preprocessing_pickle_file
    logger.info(f"Saving mapping file early to {output_file}...")

    dataset = {
        "df_smiles_train": molecule_pairs_metadata["_train"]["df_smiles"],
        "df_smiles_val": molecule_pairs_metadata["_val"]["df_smiles"],
        "df_smiles_test": molecule_pairs_metadata["_test"]["df_smiles"],
        "spectrum_indexes_train": [
            s.mgf_index for s in molecule_pairs_metadata["_train"]["original_spectra"]
        ],
        "spectrum_indexes_val": [
            s.mgf_index for s in molecule_pairs_metadata["_val"]["original_spectra"]
        ],
        "spectrum_indexes_test": [
            s.mgf_index for s in molecule_pairs_metadata["_test"]["original_spectra"]
        ],
        "mgf_path": str(cfg.paths.spectra_path),
        "format_version": "lightweight",
    }
    with open(output_file, "wb") as file:
        pickle.dump(dataset, file)

    # Load precomputed distances cache if configured
    precomputed_cache = {}
    if (
        hasattr(cfg.preprocessing, "precomputed_distances")
        and cfg.preprocessing.precomputed_distances
    ):
        dirs = cfg.preprocessing.precomputed_distances
        if dirs and len(dirs) > 0:
            logger.info(
                f"Loading precomputed distances from {len(dirs)} directory(ies)..."
            )
            precomputed_cache = load_precomputed_distances_cache(dirs)

            # Filter cache for SMILES in current dataset
            if precomputed_cache:
                all_smiles = set()
                for spectra_list in [
                    all_spectra_train,
                    all_spectra_val,
                    all_spectra_test,
                ]:
                    if spectra_list:
                        all_smiles.update(s.params["smiles"] for s in spectra_list)

                if all_smiles:
                    from rdkit import Chem as _Chem

                    from simba.core.chemistry.edit_distance.edit_distance import (
                        filter_cache_by_smiles,
                    )

                    canon_smiles = set()
                    for s in all_smiles:
                        mol = _Chem.MolFromSmiles(s)
                        if mol is not None:
                            canon_smiles.add(_Chem.MolToSmiles(mol))
                        else:
                            canon_smiles.add(s)

                    precomputed_cache = filter_cache_by_smiles(
                        precomputed_cache, list(canon_smiles)
                    )
        else:
            logger.info(
                "No precomputed distances configured, computing all from scratch"
            )
    else:
        logger.info("No precomputed distances configured, computing all from scratch")

    # Compute distances for each partition
    molecule_pairs = {}
    for type_data, spectra in [
        ("_train", all_spectra_train),
        ("_val", all_spectra_val),
        ("_test", all_spectra_test),
    ]:
        logger.info(
            f"Computing distances for {type_data[1:]} set with {len(spectra)} spectra..."
        )
        molecule_pairs[type_data] = compute_pairs_for_split(
            spectra,
            preprocessing_dir=str(workspace) + "/",
            identifier=type_data,
            num_workers=cfg.preprocessing.num_workers,
            current_node=cfg.preprocessing.current_node,
            num_nodes=cfg.preprocessing.num_nodes,
            precomputed_cache=precomputed_cache if len(precomputed_cache) > 0 else None,
            threshold_mces=cfg.model.tasks.mces.threshold,
        )

        # Log statistics about the computed pairs
        pairs_obj = molecule_pairs[type_data]
        num_unique_mols = (
            len(pairs_obj.df_smiles)
            if hasattr(pairs_obj, "df_smiles") and pairs_obj.df_smiles is not None
            else 0
        )
        num_pairs = (
            pairs_obj.pair_distances.shape[0]
            if hasattr(pairs_obj, "pair_distances")
            and pairs_obj.pair_distances is not None
            else 0
        )
        logger.info(
            f"{type_data[1:]} set completed: {num_unique_mols} unique molecules, {num_pairs} molecule pairs"
        )

    # Combine edit distance and MCES files
    logger.info("Combining edit distance and MCES files...")

    all_files = os.listdir(str(workspace))
    for partition in ["train", "val", "test"]:
        # Filter only the files ending with '.npy'
        npy_files = [f for f in all_files if f.endswith(".npy")]
        npy_files = [f for f in npy_files if partition in f]

        # Extract unique file identifiers for both single-node and multi-node patterns
        # Single-node: edit_distance_indexes_tani_incremental_train_0.npy
        # Multi-node: edit_distance_indexes_tani_incremental_train_node0_chunk1.npy
        file_identifiers = set()
        for file_loaded in npy_files:
            # Only process edit_distance files to get unique identifiers
            if not file_loaded.startswith("edit_distance_"):
                continue

            # Extract the suffix after the partition name
            suffix_pattern = f"indexes_tani_incremental_{partition}_"
            if suffix_pattern in file_loaded:
                suffix_part = file_loaded.split(suffix_pattern)[1].replace(".npy", "")
                file_identifiers.add(suffix_part)

        for identifier in sorted(file_identifiers):
            # Define file names using the extracted identifier
            suffix = f"indexes_tani_incremental_{partition}_"
            file_ed = workspace / f"edit_distance_{suffix}{identifier}.npy"
            file_mces = workspace / f"mces_{suffix}{identifier}.npy"
            file_output = workspace / f"ed_mces_{suffix}{identifier}.npy"

            if not file_ed.exists() or not file_mces.exists():
                continue

            # Load data
            ed_data = np.load(file_ed)
            mces_data = np.load(file_mces)

            logger.info(
                f"  {partition} partition '{identifier}': Loaded {ed_data.shape[0]} pairs from files"
            )

            # Check that the indexes are the same
            if np.all(ed_data[:, 0] == mces_data[:, 0]) and np.all(
                ed_data[:, 1] == mces_data[:, 1]
            ):
                # Combine ED and MCES data
                all_distance_data = np.column_stack((ed_data, mces_data[:, 2]))

                pairs_before_filter = all_distance_data.shape[0]

                # Filter out invalid values (NaN and 666)
                ed = all_distance_data[:, cfg.model.data_columns.edit_distance]
                mces = all_distance_data[:, cfg.model.data_columns.mces20]
                indexes_invalid = np.isnan(ed) | np.isnan(mces)
                ed_exceeded = ed == 666
                mces_exceeded = mces == 666

                # Count failures (with overlap-aware breakdown)
                nan_count = int(np.sum(indexes_invalid))
                both_timeout_mask = ed_exceeded & mces_exceeded & (~indexes_invalid)
                ed_only_timeout_mask = (
                    ed_exceeded & (~mces_exceeded) & (~indexes_invalid)
                )
                mces_only_timeout_mask = (
                    mces_exceeded & (~ed_exceeded) & (~indexes_invalid)
                )
                any_timeout_mask = (ed_exceeded | mces_exceeded) & (~indexes_invalid)

                both_timeout_count = int(np.sum(both_timeout_mask))
                ed_only_timeout_count = int(np.sum(ed_only_timeout_mask))
                mces_only_timeout_count = int(np.sum(mces_only_timeout_mask))
                any_timeout_count = int(np.sum(any_timeout_mask))

                # Log specific problem pairs if there are NaN values (usually large molecules)
                if nan_count > 0:
                    nan_pairs = all_distance_data[indexes_invalid][
                        : min(5, nan_count)
                    ]  # show first 5
                    logger.debug(
                        f"    Example NaN pairs (large molecules >40 atoms): "
                        f"{nan_pairs[:, :2].astype(int)}"
                    )

                # Log specific problem pairs if there are 666 values (low Tanimoto similarity)
                if any_timeout_count > 0:
                    timeout_pairs = all_distance_data[any_timeout_mask][
                        : min(5, any_timeout_count)
                    ]
                    logger.debug(
                        f"    Example 666 (low similarity) pairs: "
                        f"{timeout_pairs[:, :2].astype(int)}"
                    )

                # Apply filter
                all_distance_data = all_distance_data[
                    (~indexes_invalid) & (~ed_exceeded) & (~mces_exceeded)
                ]

                pairs_after_filter = all_distance_data.shape[0]
                pairs_removed = pairs_before_filter - pairs_after_filter
                removal_pct = (
                    100.0 * pairs_removed / pairs_before_filter
                    if pairs_before_filter > 0
                    else 0
                )

                logger.info(
                    f"  {partition} partition '{identifier}' filtering: "
                    f"{pairs_before_filter} pairs → {pairs_after_filter} pairs "
                    f"({pairs_removed} removed, {removal_pct:.1f}% loss)"
                )
                if nan_count > 0 or any_timeout_count > 0:
                    nan_pct = (
                        100.0 * nan_count / pairs_before_filter
                        if pairs_before_filter > 0
                        else 0.0
                    )
                    any_timeout_pct = (
                        100.0 * any_timeout_count / pairs_before_filter
                        if pairs_before_filter > 0
                        else 0.0
                    )
                    logger.info(
                        f"    Filtered out (non-overlapping): "
                        f"{nan_count} NaN ({nan_pct:.1f}%), "
                        f"{any_timeout_count} timeout(666) ({any_timeout_pct:.1f}%)"
                    )
                    logger.info(
                        f"    Timeout breakdown: "
                        f"ED-only={ed_only_timeout_count}, "
                        f"MCES-only={mces_only_timeout_count}, "
                        f"both={both_timeout_count}"
                    )

                np.save(file_output, all_distance_data)
                logger.info(
                    f"  Combined {partition} partition identifier '{identifier}': {all_distance_data.shape[0]} pairs"
                )

    logger.info("✅ Preprocessing completed successfully!")
