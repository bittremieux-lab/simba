"""Training workflow for SIMBA.

This module contains the main training logic adapted to work with Hydra configuration.
Refactored from legacy/training_scripts/final_training.py to use DictConfig.
"""

import sys
from pathlib import Path

import dill
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.data.weighted_sampling import (
    CustomWeightedRandomSampler,
    SimilarityWeightSampler,
)
from simba.core.models.similarity_models import SimilarityModelMultitask
from simba.core.training.callbacks import (
    LossCallback,
    ProgressLogCallback,
    ValMetricsCallback,
)
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger
from simba.utils.sanity_checks import SanityChecks
from simba.workflows.utils import load_spectra


# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs
sys.modules["simba.molecular_pairs"] = simba.core.data.molecule_pairs
sys.modules["simba.spectrum"] = simba.core.data.spectrum
sys.modules["simba.spectrum_ext"] = simba.core.data.spectrum


def load_dataset(cfg: DictConfig):
    """Load training dataset from pickle file.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (molecule_pairs_train, molecule_pairs_val,
                  molecule_pairs_test, uniformed_molecule_pairs_test)

    Raises:
        FileNotFoundError: If the mapping file does not exist
        ValueError: If the mapping file is corrupted or missing required keys
    """
    preprocessing_dir = cfg.paths.preprocessing_dir_train or cfg.paths.preprocessing_dir
    mapping_path = Path(preprocessing_dir) / cfg.paths.preprocessing_pickle_file

    # Check file existence
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {mapping_path}\n"
            f"Expected '{cfg.paths.preprocessing_pickle_file}' in preprocessing directory.\n"
            f"Run preprocessing first with 'simba preprocess' command."
        )

    logger.info(f"Loading dataset from {mapping_path}...")

    # Load and validate the mapping
    try:
        with open(mapping_path, "rb") as file:
            mapping = dill.load(file)
    except Exception as e:
        raise ValueError(
            f"Failed to deserialize dataset from {mapping_path}.\n"
            f"The file may be corrupted or incompatible.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    # Check if lightweight format
    if mapping.get("format_version") == "lightweight":
        logger.info(
            "Detected lightweight format - loading spectra dynamically from MGF"
        )

        mgf_path = mapping["mgf_path"].replace("/mnt/data2/", "/mnt/data/")

        # Use preprocessing config values (if available) to ensure consistent filtering
        use_only_protonized = getattr(
            cfg.preprocessing, "use_only_protonized_adducts", True
        )

        all_spectra = load_spectra(
            mgf_path,
            cfg,
            n_samples=-1,  # Load all spectra during training
            use_only_protonized_adducts=use_only_protonized,
        )

        # Create spectrum lookup by MGF index
        spectra_by_idx = {s.mgf_index: s for s in all_spectra}

        # Reconstruct molecule pairs with spectra
        for split in ["train", "val", "val_official", "test"]:
            df_smiles_key = f"df_smiles_{split}"
            spectrum_indexes_key = f"spectrum_indexes_{split}"

            if df_smiles_key in mapping and spectrum_indexes_key in mapping:
                df_smiles = mapping[df_smiles_key]
                spectrum_indexes = mapping[spectrum_indexes_key]

                # Load original spectra, handling missing ones
                original_spectra = []
                idx_map = {}  # old_idx -> new_idx
                missing = []

                for old_idx, mgf_idx in enumerate(spectrum_indexes):
                    if mgf_idx in spectra_by_idx:
                        idx_map[old_idx] = len(original_spectra)
                        original_spectra.append(spectra_by_idx[mgf_idx])
                    else:
                        missing.append(mgf_idx)

                mol_idx_remap = None
                if missing:
                    # Some spectra failed the training loader's validity check
                    # (filter mismatch between preprocessing and training).
                    # Drop molecules whose spectra are all missing; keep molecules
                    # that have at least one valid spectrum.
                    logger.warning(
                        f"[{split}] {len(missing)} spectra missing from loaded set "
                        f"(e.g., MGF index {missing[0]}). Dropping affected molecules."
                    )
                    # Graceful remap: skip missing entries
                    for i in df_smiles.index:
                        df_smiles.at[i, "indexes"] = [
                            idx_map[idx]
                            for idx in df_smiles.loc[i, "indexes"]
                            if idx in idx_map
                        ]
                    # Drop molecules with no valid spectra remaining
                    valid_rows = [
                        i for i in df_smiles.index if df_smiles.loc[i, "indexes"]
                    ]
                    if len(valid_rows) < len(df_smiles):
                        dropped = len(df_smiles) - len(valid_rows)
                        logger.warning(
                            f"[{split}] Dropping {dropped} molecules with no valid spectra"
                        )
                        # mol_idx_remap: old 0-based pos → new 0-based pos (for pair_distances)
                        mol_idx_remap = {old: new for new, old in enumerate(valid_rows)}
                        df_smiles = df_smiles.loc[valid_rows].reset_index(drop=True)
                else:
                    for i in df_smiles.index:
                        df_smiles.at[i, "indexes"] = [
                            idx_map[idx] for idx in df_smiles.loc[i, "indexes"]
                        ]

                # Build unique_spectra from df_smiles indexes (index is now 0..n-1)
                unique_spectra = [
                    original_spectra[df_smiles.loc[i, "indexes"][0]]
                    for i in df_smiles.index
                ]

                # Create MoleculePairsOpt object
                molecule_pairs = MoleculePairsOpt(
                    original_spectra=original_spectra,
                    unique_spectra=unique_spectra,
                    df_smiles=df_smiles,
                    pair_distances=None,  # Will be loaded separately
                )
                if mol_idx_remap is not None:
                    molecule_pairs._mol_idx_remap = mol_idx_remap

                # Store in mapping dict
                mapping[f"molecule_pairs_{split}"] = molecule_pairs

        return (
            mapping.get("molecule_pairs_train"),
            mapping.get("molecule_pairs_val"),
            mapping.get("molecule_pairs_val_official"),
            mapping.get("molecule_pairs_test"),
            None,
        )

    # Original full format validation
    logger.info("Loading full format")

    # Validate required keys
    required_keys = [
        "molecule_pairs_train",
        "molecule_pairs_val",
        "molecule_pairs_test",
        "uniformed_molecule_pairs_test",
    ]
    actual_keys = set(mapping.keys())
    missing_keys = [key for key in required_keys if key not in actual_keys]

    if missing_keys:
        raise ValueError(
            f"Dataset mapping is missing required keys: {missing_keys}\n"
            f"Found keys: {sorted(actual_keys)}\n"
            f"The preprocessing output may be from an incompatible version.\n"
            f"Re-run preprocessing with 'simba preprocess' command."
        )

    return (
        mapping["molecule_pairs_train"],
        mapping["molecule_pairs_val"],
        None,  # val_official not present in full-format mappings
        mapping["molecule_pairs_test"],
        mapping["uniformed_molecule_pairs_test"],
    )


def prepare_data(
    molecule_pairs_train,
    molecule_pairs_val,
    molecule_pairs_test,
    uniformed_molecule_pairs_test,
    cfg: DictConfig,
    molecule_pairs_val_official=None,
) -> tuple:
    """Prepare training data from molecule pairs.

    Args:
        molecule_pairs_train: Training molecule pairs
        molecule_pairs_val: Validation molecule pairs (scaffold split)
        molecule_pairs_test: Test molecule pairs
        uniformed_molecule_pairs_test: Uniformed test pairs
        cfg: Hydra configuration
        molecule_pairs_val_official: Optional second val set (MSG official val fold)

    Returns:
        Tuple of (dataset_train, train_sampler, dataset_val, val_sampler,
                  dataset_val_official, val_official_sampler, weights_ed, bins_ed).
        dataset_val_official and val_official_sampler are None when not provided.
    """
    logger.info("Loading pairs data ...")

    # Load MCES indexes for training
    indexes_tani_multitasking_train = LoadMCES.merge_numpy_arrays(
        cfg.paths.preprocessing_dir_train,
        prefix="ed_mces_indexes_tani_incremental_train",
        use_edit_distance=cfg.model.tasks.edit_distance.enabled,
        use_multitask=cfg.model.multitasking.enabled,
        add_high_similarity_pairs=cfg.sampling.add_high_similarity_pairs,
        remove_percentage=0.0,
    )
    indexes_tani_multitasking_train = _remove_duplicates_array(
        indexes_tani_multitasking_train
    )

    # Load MCES indexes for validation
    indexes_tani_multitasking_val = LoadMCES.merge_numpy_arrays(
        cfg.paths.preprocessing_dir_train,  # Note: uses TRAIN dir (same as original)
        prefix="ed_mces_indexes_tani_incremental_val",
        use_edit_distance=cfg.model.tasks.edit_distance.enabled,
        use_multitask=cfg.model.multitasking.enabled,
        add_high_similarity_pairs=cfg.sampling.add_high_similarity_pairs,
    )
    indexes_tani_multitasking_val = _remove_duplicates_array(
        indexes_tani_multitasking_val
    )

    ed_col = cfg.model.data_columns.edit_distance
    mces_col = cfg.model.data_columns.mces20

    def _apply_remap(arr, mol_pairs):
        """Filter and remap molecule indices in pair array using mol_pairs._mol_idx_remap."""
        remap = getattr(mol_pairs, "_mol_idx_remap", None)
        if remap is None:
            return arr
        col0 = arr[:, 0].astype(int)
        col1 = arr[:, 1].astype(int)
        mask = np.array([c in remap and d in remap for c, d in zip(col0, col1)])
        arr = arr[mask].copy()
        arr[:, 0] = np.array([remap[int(x)] for x in arr[:, 0]])
        arr[:, 1] = np.array([remap[int(x)] for x in arr[:, 1]])
        logger.info(
            f"mol_idx_remap: kept {mask.sum()} / {len(mask)} pairs after filtering"
        )
        return arr

    indexes_tani_multitasking_train = _apply_remap(
        indexes_tani_multitasking_train, molecule_pairs_train
    )
    indexes_tani_multitasking_val = _apply_remap(
        indexes_tani_multitasking_val, molecule_pairs_val
    )

    molecule_pairs_train.pair_distances = indexes_tani_multitasking_train[
        :, [0, 1, ed_col]
    ]
    molecule_pairs_train.extra_distances = indexes_tani_multitasking_train[:, mces_col]
    molecule_pairs_val.pair_distances = indexes_tani_multitasking_val[:, [0, 1, ed_col]]
    molecule_pairs_val.extra_distances = indexes_tani_multitasking_val[:, mces_col]

    logger.info(f"Number of pairs for train: {len(molecule_pairs_train)}")
    logger.info(f"Number of pairs for val: {len(molecule_pairs_val)}")

    # Load pairs for the official MSG val fold if provided
    if molecule_pairs_val_official is not None:
        indexes_tani_multitasking_val_official = LoadMCES.merge_numpy_arrays(
            cfg.paths.preprocessing_dir_train,
            prefix="ed_mces_indexes_tani_incremental_val_official",
            use_edit_distance=cfg.model.tasks.edit_distance.enabled,
            use_multitask=cfg.model.multitasking.enabled,
            add_high_similarity_pairs=cfg.sampling.add_high_similarity_pairs,
        )
        indexes_tani_multitasking_val_official = _remove_duplicates_array(
            indexes_tani_multitasking_val_official
        )
        indexes_tani_multitasking_val_official = _apply_remap(
            indexes_tani_multitasking_val_official, molecule_pairs_val_official
        )
        molecule_pairs_val_official.pair_distances = (
            indexes_tani_multitasking_val_official[:, [0, 1, ed_col]]
        )
        molecule_pairs_val_official.extra_distances = (
            indexes_tani_multitasking_val_official[:, mces_col]
        )
        logger.info(
            f"Number of pairs for val_official: {len(molecule_pairs_val_official)}"
        )

    # Sanity checks
    sanity_check_ids = SanityChecks.sanity_checks_ids(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    )
    sanity_check_bms = SanityChecks.sanity_checks_bms(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    )
    logger.info(f"Sanity check ids. Passed? {sanity_check_ids}")
    logger.info(f"Sanity check bms. Passed? {sanity_check_bms}")

    # Calculate weights for the training set.
    # Use MCES-based sampling only when ED is disabled AND the ED column is all zeros
    # (i.e. the preprocessing didn't compute real edit distances). When real ED values
    # are present, use them for sampling even if the ED head is disabled — this gives
    # meaningful pair balancing without requiring the ED objective.
    # NOTE: pair_distances[:, 2] is NORMALIZED ED (normalize_ed maps raw=0 → 1.0,
    # raw>0 → <1.0). Check != 1.0 to detect real non-zero ED, not != 0.
    n_bins = cfg.model.tasks.edit_distance.n_classes - 1
    has_real_ed = bool((molecule_pairs_train.pair_distances[:, 2] != 1.0).any())
    use_mces_sampling = (
        not cfg.model.tasks.edit_distance.enabled
        and not has_real_ed
        and molecule_pairs_train.extra_distances is not None
    )
    if use_mces_sampling:
        # Non-uniform bins: fine resolution in [0,10], width-5 above.
        # Edges: [0,2.5), [2.5,5), [5,7.5), [7.5,10), [10,15), [15,20),
        #        [20,25), [25,30), [30,35), [35,40]  → 10 bins (n_classes=11)
        _mces_edges = np.array([2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])

        mces_sim_tr = molecule_pairs_train.extra_distances  # similarity = 1 - MCES/40
        mces_raw_tr = (1.0 - mces_sim_tr) * 40.0
        bin_idx_tr = np.clip(
            np.searchsorted(_mces_edges, mces_raw_tr).astype(int), 0, n_bins - 1
        )
        bin_counts = np.bincount(bin_idx_tr, minlength=n_bins)
        total = bin_counts.sum()
        weights_ed = np.where(bin_counts > 0, total / bin_counts.astype(float), 0.0)
        weights_ed = weights_ed / weights_ed.sum()
        bins_ed = np.arange(n_bins) / n_bins
        weights_tr = weights_ed[bin_idx_tr]
        weights_tr = weights_tr / weights_tr.sum()

        mces_sim_val = molecule_pairs_val.extra_distances
        mces_raw_val = (1.0 - mces_sim_val) * 40.0
        bin_idx_val = np.clip(
            np.searchsorted(_mces_edges, mces_raw_val).astype(int), 0, n_bins - 1
        )
        bin_counts_val = np.bincount(bin_idx_val, minlength=n_bins)
        total_val = bin_counts_val.sum()
        weights_ed_val = np.where(
            bin_counts_val > 0, total_val / bin_counts_val.astype(float), 0.0
        )
        weights_ed_val = weights_ed_val / weights_ed_val.sum()
        weights_val = weights_ed_val[bin_idx_val]
        weights_val = weights_val / weights_val.sum()

        logger.info(
            f"MCES-based sampling: {n_bins} bins, "
            f"train counts: {bin_counts.tolist()}, "
            f"val_scaffold counts: {bin_counts_val.tolist()}"
        )
    else:
        train_binned_list, ranges = TrainUtils.divide_data_into_bins_categories(
            molecule_pairs_train,
            n_bins,
            bin_sim_1=True,
        )
        weights_ed, bins_ed = SimilarityWeightSampler.compute_weights(train_binned_list)
        weights_tr = SimilarityWeightSampler.compute_sample_weights_categories(
            molecule_pairs_train, weights_ed
        )
        weights_val = SimilarityWeightSampler.compute_sample_weights_categories(
            molecule_pairs_val, weights_ed
        )

    # Create datasets from molecule pairs
    dataset_train = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_train,
        max_num_peaks=int(cfg.model.transformer.context_length),
        training=True,
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
        use_ion_mode=cfg.model.features.use_ion_mode,
    )

    dataset_val = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_val,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
        use_ion_mode=cfg.model.features.use_ion_mode,
    )

    # Create samplers
    train_sampler = CustomWeightedRandomSampler(
        weights=weights_tr, num_samples=len(dataset_train), replacement=True
    )
    val_sampler = CustomWeightedRandomSampler(
        weights=weights_val, num_samples=len(dataset_val), replacement=True
    )

    # Build optional official val dataset + sampler using the same train weights
    dataset_val_official = None
    val_official_sampler = None
    if molecule_pairs_val_official is not None:
        dataset_val_official = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
            molecule_pairs_val_official,
            max_num_peaks=int(cfg.model.transformer.context_length),
            use_adduct=cfg.model.features.use_adduct,
            use_ce=cfg.model.features.use_ce,
            use_ion_activation=cfg.model.features.use_ion_activation,
            use_ion_method=cfg.model.features.use_ion_method,
            use_ion_mode=cfg.model.features.use_ion_mode,
        )
        if use_mces_sampling:
            mces_sim_off = molecule_pairs_val_official.extra_distances
            mces_raw_off = (1.0 - mces_sim_off) * 40.0
            bin_idx_off = np.clip(
                np.searchsorted(_mces_edges, mces_raw_off).astype(int), 0, n_bins - 1
            )
            bin_counts_off = np.bincount(bin_idx_off, minlength=n_bins)
            total_off = bin_counts_off.sum()
            weights_ed_off = np.where(
                bin_counts_off > 0, total_off / bin_counts_off.astype(float), 0.0
            )
            weights_ed_off = weights_ed_off / weights_ed_off.sum()
            weights_off = weights_ed_off[bin_idx_off]
            weights_off = weights_off / weights_off.sum()
            logger.info(
                f"MCES-based sampling val_official counts: {bin_counts_off.tolist()}"
            )
        else:
            weights_off = SimilarityWeightSampler.compute_sample_weights_categories(
                molecule_pairs_val_official, weights_ed
            )
        val_official_sampler = CustomWeightedRandomSampler(
            weights=weights_off,
            num_samples=len(dataset_val_official),
            replacement=True,
        )

    return (
        dataset_train,
        train_sampler,
        dataset_val,
        val_sampler,
        dataset_val_official,
        val_official_sampler,
        weights_ed,
        bins_ed,
    )


def create_dataloaders(
    cfg: DictConfig,
    dataset_train,
    train_sampler,
    dataset_val,
    val_sampler,
    dataset_val_official=None,
    val_official_sampler=None,
):
    """Create PyTorch DataLoaders for training and validation.
    Args:
        cfg: Hydra configuration
        dataset_train: Training dataset
        train_sampler: Training sampler (or None)
        dataset_val: Scaffold-split validation dataset
        val_sampler: Scaffold val sampler (or None)
        dataset_val_official: Optional MSG official val dataset
        val_official_sampler: Optional MSG official val sampler
    Returns:
        (dataloader_train, dataloader_val) where dataloader_val is a single
        DataLoader when no official val is given, or a list of two DataLoaders
        [scaffold_loader, official_loader] when dataset_val_official is provided.
    """
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.hardware.num_workers,
        persistent_workers=cfg.hardware.num_workers > 0,
    )

    dataloader_val_scaffold = DataLoader(
        dataset_val,
        batch_size=cfg.training.batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=cfg.hardware.num_workers,
        persistent_workers=cfg.hardware.num_workers > 0,
        generator=torch.Generator().manual_seed(42) if val_sampler is None else None,
    )

    if dataset_val_official is not None:
        dataloader_val_official = DataLoader(
            dataset_val_official,
            batch_size=cfg.training.batch_size,
            shuffle=(val_official_sampler is None),
            sampler=val_official_sampler,
            num_workers=cfg.hardware.num_workers,
            persistent_workers=cfg.hardware.num_workers > 0,
            generator=(
                torch.Generator().manual_seed(43)
                if val_official_sampler is None
                else None
            ),
        )
        return dataloader_train, [dataloader_val_scaffold, dataloader_val_official]

    return dataloader_train, dataloader_val_scaffold


def setup_callbacks(cfg: DictConfig, val_names: list | None = None) -> tuple:
    """Setup PyTorch Lightning callbacks.
    Args:
        cfg: Hydra configuration
    Returns:
        Tuple of (checkpoint_callback, checkpoint_n_steps_callback, loss_callback)
    """
    from simba.utils.config_utils import get_model_paths

    paths = get_model_paths(cfg)

    # When multiple val loaders are used, Lightning appends /dataloader_idx_N to metric names
    val_loss_monitor = (
        "validation_loss/dataloader_idx_0"
        if val_names and len(val_names) > 1
        else "validation_loss"
    )

    # Always save best model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths["checkpoint_dir"]),
        filename=cfg.checkpoints.best_model_name.replace(".ckpt", ""),
        save_top_k=1,
        monitor=val_loss_monitor,
        mode="min",
    )

    # Optionally save periodic checkpoints
    checkpoint_n_steps_callback = None
    if cfg.checkpoints.get("save_checkpoints", True):
        checkpoint_n_steps_callback = ModelCheckpoint(
            dirpath=str(paths["checkpoint_dir"]),
            filename="checkpoint-{epoch:02d}-{step}",
            every_n_train_steps=cfg.training.val_check_interval,
            save_top_k=-1,  # Save all checkpoints
        )

    # Loss tracking callback (saves loss plot to checkpoint dir)
    loss_plot_path = paths["checkpoint_dir"] / "loss_plot.png"
    loss_callback = LossCallback(file_path=str(loss_plot_path))

    # Validation metrics callback (saves confusion matrix + MCES hexbins)
    val_metrics_callback = ValMetricsCallback(
        output_dir=str(paths["checkpoint_dir"]),
        n_classes=cfg.model.tasks.edit_distance.n_classes,
        val_names=val_names or ["val"],
    )

    # Progress logging callback (writes INFO lines to .err log file)
    progress_log_callback = ProgressLogCallback(
        log_every_n_steps=cfg.logging.get("progress_log_every_n_steps", 100)
    )

    # Optional early stopping: patience=0 means disabled.
    # When scaffold val is available, monitor its Spearman (higher=better).
    # Falls back to validation_loss (lower=better) when scaffold is absent.
    early_stopping_callback = None
    patience = cfg.training.get("early_stopping_patience", 0)
    if patience and patience > 0:
        use_scaffold = val_names and "scaffold" in val_names
        es_monitor = "val_mces_spearman/scaffold" if use_scaffold else "validation_loss"
        es_mode = "max" if use_scaffold else "min"
        early_stopping_callback = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            mode=es_mode,
            verbose=True,
        )

    return (
        checkpoint_callback,
        checkpoint_n_steps_callback,
        loss_callback,
        early_stopping_callback,
        progress_log_callback,
        val_metrics_callback,
    )


def setup_model(cfg: DictConfig, weights_mces: np.ndarray) -> SimilarityModelMultitask:
    """Setup the SIMBA model.
    Args:
        cfg: Hydra configuration
        weights_mces: MCES weights for loss calculation
    Returns:
        Initialized SimilarityModelMultitask model
    """
    model_kwargs = {
        "d_model": cfg.model.transformer.d_model,
        "n_layers": cfg.model.transformer.n_layers,
        "n_classes": cfg.model.tasks.edit_distance.n_classes,
        "use_gumbel": cfg.model.tasks.edit_distance.use_gumbel,
        "use_element_wise": cfg.model.features.use_element_wise,
        "use_cosine_distance": cfg.model.tasks.cosine_similarity.use_cosine_distance,
        "use_edit_distance_regresion": cfg.model.tasks.edit_distance.use_regression,
        "use_fingerprints": cfg.model.tasks.fingerprints.enabled,
        "USE_LEARNABLE_MULTITASK": cfg.model.multitasking.learnable,
        "use_mces20_log_loss": cfg.model.tasks.mces.use_log_loss,
        "tau_gumbel_softmax": cfg.model.tasks.edit_distance.tau_gumbel_softmax,
        "gumbel_reg_weight": cfg.model.tasks.edit_distance.gumbel_reg_weight,
        "weights": weights_mces,
        "lr": cfg.optimizer.lr,
        "use_adduct": cfg.model.features.use_adduct,
        "use_precursor_mz_for_model": cfg.model.features.use_precursor_mz,
        "use_ce": cfg.model.features.use_ce,
        "use_ion_activation": cfg.model.features.use_ion_activation,
        "use_ion_method": cfg.model.features.use_ion_method,
        "use_ion_mode": cfg.model.features.use_ion_mode,
        "use_edit_distance": cfg.model.tasks.edit_distance.enabled,
    }

    # Load pretrained weights if specified
    if cfg.model.pretrained.load_pretrained:
        from simba.utils.config_utils import get_model_paths

        paths = get_model_paths(cfg)
        pretrained_path = paths["pretrained_path"]

        if pretrained_path.exists():
            model = SimilarityModelMultitask.load_from_checkpoint(
                str(pretrained_path), **model_kwargs
            )
        else:
            model = SimilarityModelMultitask(**model_kwargs)
    else:
        model = SimilarityModelMultitask(**model_kwargs)

    return model


def train(
    model: SimilarityModelMultitask,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    cfg: DictConfig,
    checkpoint_callback: ModelCheckpoint | None,
    checkpoint_n_steps_callback: ModelCheckpoint | None,
    loss_callback: LossCallback,
    early_stopping_callback: EarlyStopping | None = None,
    progress_log_callback: ProgressLogCallback | None = None,
    val_metrics_callback: ValMetricsCallback | None = None,
) -> pl.Trainer:
    """Run the training loop.
    Args:
        model: SIMBA model to train
        dataloader_train: Training dataloader
        dataloader_val: Validation dataloader
        cfg: Hydra configuration
        checkpoint_callback: Best model checkpoint callback (optional, can be None)
        checkpoint_n_steps_callback: Periodic checkpoint callback (optional, can be None)
        loss_callback: Loss tracking callback
        early_stopping_callback: EarlyStopping callback (optional, None=disabled)
    Returns:
        The fitted PyTorch Lightning Trainer.
    """
    # Build callbacks list, excluding None values
    callbacks = [
        cb
        for cb in [
            checkpoint_callback,
            checkpoint_n_steps_callback,
            loss_callback,
            early_stopping_callback,
            progress_log_callback,
            val_metrics_callback,
        ]
        if cb is not None
    ]

    torch.set_float32_matmul_precision("high")

    from lightning.pytorch.loggers import CSVLogger

    from simba.utils.config_utils import get_model_paths

    checkpoint_dir = get_model_paths(cfg)["checkpoint_dir"]
    csv_logger = CSVLogger(save_dir=str(checkpoint_dir), name="", version="")

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        val_check_interval=cfg.training.val_check_interval,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    return trainer


def _remove_duplicates_array(arr: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from numpy array."""
    return np.unique(arr, axis=0)
