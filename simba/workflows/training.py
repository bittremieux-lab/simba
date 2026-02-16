"""Training workflow for SIMBA.

This module contains the main training logic adapted to work with Hydra configuration.
Refactored from legacy/training_scripts/final_training.py to use DictConfig.
"""

import sys
from pathlib import Path

import dill
import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import simba.core.data.molecular_pairs
import simba.core.data.molecule_pairs_opt
import simba.core.data.spectrum
from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.molecule_pairs_opt import MoleculePairsOpt
from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.data.sampling.custom_weighted_random_sampler import (
    CustomWeightedRandomSampler,
)
from simba.core.data.sampling.weight_sampling import WeightSampling
from simba.core.models.ordinal.embedder_multitask import EmbedderMultitask
from simba.core.models.ordinal.load_data_multitasking import LoadDataMultitasking
from simba.core.training.losscallback import LossCallback
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger
from simba.utils.sanity_checks import SanityChecks


# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs_opt
sys.modules["simba.molecular_pairs"] = simba.core.data.molecular_pairs
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

        mgf_path = mapping["mgf_path"]
        all_spectra = PreprocessingSimba.load_spectra(mgf_path, cfg)

        # Create spectrum lookup by MGF index
        spectra_by_idx = {s.mgf_index: s for s in all_spectra}

        # Reconstruct molecule pairs with spectra
        for split in ["train", "val", "test"]:
            df_smiles_key = f"df_smiles_{split}"
            spectrum_indexes_key = f"spectrum_indexes_{split}"

            if df_smiles_key in mapping and spectrum_indexes_key in mapping:
                df_smiles = mapping[df_smiles_key]
                spectrum_indexes = mapping[spectrum_indexes_key]

                # Load original spectra (all, including duplicates)
                original_spectra = [spectra_by_idx[idx] for idx in spectrum_indexes]

                # Build unique_spectra from df_smiles indexes
                # df_smiles['indexes'] contains lists of indexes into original_spectra
                # We take the first spectrum for each unique SMILES
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

                # Store in mapping dict
                mapping[f"molecule_pairs_{split}"] = molecule_pairs

        return (
            mapping.get("molecule_pairs_train"),
            mapping.get("molecule_pairs_val"),
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
        mapping["molecule_pairs_test"],
        mapping["uniformed_molecule_pairs_test"],
    )


def prepare_data(
    molecule_pairs_train,
    molecule_pairs_val,
    molecule_pairs_test,
    uniformed_molecule_pairs_test,
    cfg: DictConfig,
) -> tuple:
    """Prepare training data from molecule pairs.

    Args:
        molecule_pairs_train: Training molecule pairs
        molecule_pairs_val: Validation molecule pairs
        molecule_pairs_test: Test molecule pairs
        uniformed_molecule_pairs_test: Uniformed test pairs
        cfg: Hydra configuration

    Returns:
        Tuple of (dataset_train, train_sampler, dataset_val, val_sampler, weights_ed, bins_ed)
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

    # Assign edit distance to molecule pairs
    molecule_pairs_train.pair_distances = indexes_tani_multitasking_train[
        :, [0, 1, cfg.model.data_columns.edit_distance]
    ]
    molecule_pairs_val.pair_distances = indexes_tani_multitasking_val[
        :, [0, 1, cfg.model.data_columns.edit_distance]
    ]

    # Add MCES to molecule pairs
    molecule_pairs_train.extra_distances = indexes_tani_multitasking_train[
        :, cfg.model.data_columns.mces20
    ]
    molecule_pairs_val.extra_distances = indexes_tani_multitasking_val[
        :, cfg.model.data_columns.mces20
    ]

    logger.info(f"Number of pairs for train: {len(molecule_pairs_train)}")
    logger.info(f"Number of pairs for val: {len(molecule_pairs_val)}")

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

    # Calculate weights for the training set
    train_binned_list, ranges = TrainUtils.divide_data_into_bins_categories(
        molecule_pairs_train,
        cfg.model.tasks.edit_distance.n_classes - 1,
        bin_sim_1=True,
    )
    weights_ed, bins_ed = WeightSampling.compute_weights(train_binned_list)
    weights_tr = WeightSampling.compute_sample_weights_categories(
        molecule_pairs_train, weights_ed
    )
    weights_val = WeightSampling.compute_sample_weights_categories(
        molecule_pairs_val, weights_ed
    )

    # Create datasets from molecule pairs
    dataset_train = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_train,
        max_num_peaks=int(cfg.model.transformer.context_length),
        training=True,
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
    )

    dataset_val = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_val,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
    )

    # Create samplers
    train_sampler = CustomWeightedRandomSampler(
        weights=weights_tr, num_samples=len(dataset_train), replacement=True
    )
    val_sampler = CustomWeightedRandomSampler(
        weights=weights_val, num_samples=len(dataset_val), replacement=True
    )

    return (
        dataset_train,
        train_sampler,
        dataset_val,
        val_sampler,
        weights_ed,
        bins_ed,
    )


def create_dataloaders(
    cfg: DictConfig,
    dataset_train,
    train_sampler,
    dataset_val,
    val_sampler,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation.
    Args:
        cfg: Hydra configuration
        dataset_train: Training dataset
        train_sampler: Training sampler (or None)
        dataset_val: Validation dataset
        val_sampler: Validation sampler (or None)
    Returns:
        Tuple of (dataloader_train, dataloader_val)
    """
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.hardware.num_workers,
        persistent_workers=cfg.hardware.num_workers > 0,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.hardware.num_workers,
        persistent_workers=cfg.hardware.num_workers > 0,
    )

    return dataloader_train, dataloader_val


def setup_callbacks(cfg: DictConfig) -> tuple:
    """Setup PyTorch Lightning callbacks.
    Args:
        cfg: Hydra configuration
    Returns:
        Tuple of (checkpoint_callback, checkpoint_n_steps_callback, loss_callback)
    """
    from simba.utils.config_utils import get_model_paths

    paths = get_model_paths(cfg)

    # Always save best model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths["checkpoint_dir"]),
        filename=cfg.checkpoints.best_model_name.replace(".ckpt", ""),
        save_top_k=1,
        monitor="validation_loss",
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

    return checkpoint_callback, checkpoint_n_steps_callback, loss_callback


def setup_model(cfg: DictConfig, weights_mces: np.ndarray) -> EmbedderMultitask:
    """Setup the SIMBA model.
    Args:
        cfg: Hydra configuration
        weights_mces: MCES weights for loss calculation
    Returns:
        Initialized EmbedderMultitask model
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
    }

    # Load pretrained weights if specified
    if cfg.model.pretrained.load_pretrained:
        from simba.utils.config_utils import get_model_paths

        paths = get_model_paths(cfg)
        pretrained_path = paths["pretrained_path"]

        if pretrained_path.exists():
            model = EmbedderMultitask.load_from_checkpoint(
                str(pretrained_path), **model_kwargs
            )
        else:
            model = EmbedderMultitask(**model_kwargs)
    else:
        model = EmbedderMultitask(**model_kwargs)

    return model


def train(
    model: EmbedderMultitask,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    cfg: DictConfig,
    checkpoint_callback: ModelCheckpoint | None,
    checkpoint_n_steps_callback: ModelCheckpoint | None,
    loss_callback: LossCallback,
) -> None:
    """Run the training loop.
    Args:
        model: SIMBA model to train
        dataloader_train: Training dataloader
        dataloader_val: Validation dataloader
        cfg: Hydra configuration
        checkpoint_callback: Best model checkpoint callback (optional, can be None)
        checkpoint_n_steps_callback: Periodic checkpoint callback (optional, can be None)
        loss_callback: Loss tracking callback
    """
    # Build callbacks list, excluding None values
    callbacks = [
        cb
        for cb in [checkpoint_callback, checkpoint_n_steps_callback, loss_callback]
        if cb is not None
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    trainer.fit(model, dataloader_train, dataloader_val)


def _remove_duplicates_array(arr: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from numpy array."""
    return np.unique(arr, axis=0)
