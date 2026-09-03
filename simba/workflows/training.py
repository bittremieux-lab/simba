"""Training workflow for SIMBA.

This module contains the main training logic adapted to work with Hydra configuration.
Refactored from legacy/training_scripts/final_training.py to use DictConfig.
"""

import hashlib
import sys
from pathlib import Path

import dill
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.core.chemistry.chem_utils import mass_lookup_from_df_smiles
from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.data.weighted_sampling import CustomWeightedRandomSampler
from simba.core.models.similarity_models import SimilarityModelMultitask
from simba.core.training.callbacks import (
    IcebergHitRateCallback,
    LossCallback,
    ProgressLogCallback,
    ValMetricsCallback,
)
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

        mgf_path = getattr(cfg.paths, "mgf_path", None) or mapping["mgf_path"]

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
        for split in ["train", "val", "test"]:
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


def _compute_train_weights(molecule_pairs_train, cfg: DictConfig):
    """Per-pair training sample weights: an MCES-tiered, mass-tier-reweighted
    inverse-frequency scheme.

    Returns:
        weights_tr
    """
    sampling_edges = np.array(cfg.sampling.mces_sampling_bin_edges)
    bucket_multipliers = np.array(cfg.sampling.mces_sampling_bucket_multipliers)
    n_sampling_bins = len(sampling_edges) + 1

    mces_sim_tr = molecule_pairs_train.extra_distances  # similarity = 1 - MCES/40
    mces_raw_tr = (1.0 - mces_sim_tr) * 40.0
    bin_idx_tr = np.clip(
        np.searchsorted(sampling_edges, mces_raw_tr).astype(int),
        0,
        n_sampling_bins - 1,
    )
    bin_counts = np.bincount(bin_idx_tr, minlength=n_sampling_bins)
    total = bin_counts.sum()
    weights_ed = np.where(bin_counts > 0, total / bin_counts.astype(float), 0.0)
    weights_ed = weights_ed * bucket_multipliers
    weights_ed = weights_ed / weights_ed.sum()
    weights_tr = weights_ed[bin_idx_tr]

    # Within each non-self bucket, upweight low mass-difference pairs:
    # those at/below the bucket's own mass_tier_quantile get
    # mass_tier_low_share of that bucket's sampling weight, the rest
    # get the remainder. Skipped for the self bucket (mass_diff is
    # always 0 there).
    mass_tier_quantile = cfg.sampling.mass_tier_quantile
    mass_tier_low_share = cfg.sampling.mass_tier_low_share
    mol_mass = mass_lookup_from_df_smiles(molecule_pairs_train.df_smiles)
    mass_diff_tr = np.abs(
        mol_mass[molecule_pairs_train.pair_distances[:, 0].astype(int)]
        - mol_mass[molecule_pairs_train.pair_distances[:, 1].astype(int)]
    )
    mass_tier_weight = np.ones(len(bin_idx_tr))
    for b in range(1, n_sampling_bins):
        in_bucket = bin_idx_tr == b
        diffs = mass_diff_tr[in_bucket]
        n_in_bucket = diffs.size
        if n_in_bucket == 0:
            continue
        q_low = np.nanquantile(diffs, mass_tier_quantile)
        low_mask = diffs <= q_low
        n_low = int(low_mask.sum())
        n_high = int(n_in_bucket - n_low)
        low_weight = (mass_tier_low_share * n_in_bucket) / n_low if n_low else 0.0
        high_weight = (
            ((1.0 - mass_tier_low_share) * n_in_bucket) / n_high if n_high else 0.0
        )
        mass_tier_weight[in_bucket] = np.where(low_mask, low_weight, high_weight)

    weights_tr = weights_tr * mass_tier_weight
    weights_tr = weights_tr / weights_tr.sum()
    return weights_tr


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
        Tuple of (dataset_train, train_sampler, dataset_val, val_sampler)
    """
    logger.info("Loading pairs data ...")

    # Load MCES indexes for training
    indexes_tani_multitasking_train = LoadMCES.merge_numpy_arrays(
        cfg.paths.preprocessing_dir_train,
        prefix="ed_mces_indexes_tani_incremental_train",
        use_edit_distance=False,  # inert for use_multitask=True, kept for signature compatibility
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
        use_edit_distance=False,  # inert for use_multitask=True, kept for signature compatibility
        use_multitask=cfg.model.multitasking.enabled,
        add_high_similarity_pairs=cfg.sampling.add_high_similarity_pairs,
    )
    indexes_tani_multitasking_val = _remove_duplicates_array(
        indexes_tani_multitasking_val
    )

    ed_col = cfg.model.data_columns.edit_distance
    mces_col = cfg.model.data_columns.mces20

    def _remap_cache_path(prefix, remap):
        """Cache path for a split's remapped array, next to its source npy files.

        Keyed on the source files' identity (mtime+size) and the remap dict's
        content, so a changed preprocessing dir or a different dropped-molecule
        set can't silently hit a stale cache.
        """
        prepro_dir = Path(cfg.paths.preprocessing_dir_train)
        src_files = sorted(prepro_dir.glob(f"{prefix}_node*_chunk*.npy"))
        if not src_files:
            return None
        stats = [(f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in src_files]
        key = (
            f"{stats}|{sorted(remap.items())}|"
            f"{cfg.model.multitasking.enabled}|{cfg.sampling.add_high_similarity_pairs}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return prepro_dir / f"{prefix}.remap_cache.{digest}.npy"

    def _apply_remap(arr, mol_pairs, prefix):
        """Filter and remap molecule indices in pair array using mol_pairs._mol_idx_remap."""
        remap = getattr(mol_pairs, "_mol_idx_remap", None)
        if remap is None:
            return arr

        cache_path = _remap_cache_path(prefix, remap)
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading cached remap result from {cache_path}")
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load remap cache ({type(e).__name__}: {e}); "
                    "recomputing."
                )

        col0 = arr[:, 0].astype(int)
        col1 = arr[:, 1].astype(int)
        mask = np.array([c in remap and d in remap for c, d in zip(col0, col1)])
        arr = arr[mask].copy()
        arr[:, 0] = np.array([remap[int(x)] for x in arr[:, 0]])
        arr[:, 1] = np.array([remap[int(x)] for x in arr[:, 1]])
        logger.info(
            f"mol_idx_remap: kept {mask.sum()} / {len(mask)} pairs after filtering"
        )

        if cache_path is not None:
            try:
                np.save(cache_path, arr)
                logger.info(f"Cached remap result to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to write remap cache ({type(e).__name__}: {e})")

        return arr

    def _add_identity_pairs(arr, mol_pairs, name):
        """Append one molecule-paired-with-itself row per molecule in this split.
        ED/MCES columns store normalized similarity (1 - dist/max_value), so a
        raw distance of 0 encodes as 1.0."""
        if not cfg.sampling.add_identity_pairs:
            return arr
        idx = np.asarray(mol_pairs.df_smiles.index, dtype=arr.dtype)
        identity_rows = np.zeros((len(idx), arr.shape[1]), dtype=arr.dtype)
        identity_rows[:, 0] = idx
        identity_rows[:, 1] = idx
        identity_rows[:, ed_col] = 1.0
        identity_rows[:, mces_col] = 1.0
        logger.info(f"[{name}] added {len(idx)} identity pairs (MCES=0, ED=0)")
        return np.concatenate([arr, identity_rows], axis=0)

    indexes_tani_multitasking_train = _apply_remap(
        indexes_tani_multitasking_train,
        molecule_pairs_train,
        "ed_mces_indexes_tani_incremental_train",
    )
    indexes_tani_multitasking_train = _add_identity_pairs(
        indexes_tani_multitasking_train, molecule_pairs_train, "train"
    )
    indexes_tani_multitasking_val = _apply_remap(
        indexes_tani_multitasking_val,
        molecule_pairs_val,
        "ed_mces_indexes_tani_incremental_val",
    )
    indexes_tani_multitasking_val = _add_identity_pairs(
        indexes_tani_multitasking_val, molecule_pairs_val, "val"
    )

    # Assign edit distance to molecule pairs
    molecule_pairs_train.pair_distances = indexes_tani_multitasking_train[
        :, [0, 1, ed_col]
    ]
    molecule_pairs_val.pair_distances = indexes_tani_multitasking_val[:, [0, 1, ed_col]]

    # Add MCES to molecule pairs
    molecule_pairs_train.extra_distances = indexes_tani_multitasking_train[:, mces_col]
    molecule_pairs_val.extra_distances = indexes_tani_multitasking_val[:, mces_col]

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

    weights_tr = _compute_train_weights(molecule_pairs_train, cfg)

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
        precursor_mass_mode=cfg.sampling.get("precursor_mass_mode", "measured"),
        precursor_noise_mode=cfg.sampling.get("precursor_noise_mode", "legacy"),
        prob_aug=cfg.augmentation.prob_aug,
    )

    dataset_val = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_val,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
        use_ion_mode=cfg.model.features.use_ion_mode,
        precursor_mass_mode=cfg.sampling.get("precursor_mass_mode", "measured"),
    )

    # Validation always uses a full, unweighted, sequential pass regardless
    # of use_resampling -- val_sampler stays None either way.
    if cfg.sampling.use_resampling:
        train_sampler = CustomWeightedRandomSampler(
            weights=weights_tr, num_samples=len(dataset_train), replacement=True
        )
    else:
        train_sampler = None
    val_sampler = None

    return (
        dataset_train,
        train_sampler,
        dataset_val,
        val_sampler,
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
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=cfg.hardware.num_workers,
        persistent_workers=cfg.hardware.num_workers > 0,
        generator=torch.Generator().manual_seed(42) if val_sampler is None else None,
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

    save_checkpoints = cfg.checkpoints.get("save_checkpoints", True)

    checkpoint_callback = None
    checkpoint_n_steps_callback = None
    if save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(paths["checkpoint_dir"]),
            filename=cfg.checkpoints.best_model_name.replace(".ckpt", ""),
            save_top_k=1,
            monitor="validation_loss",
            mode="min",
        )
        checkpoint_n_steps_callback = ModelCheckpoint(
            dirpath=str(paths["checkpoint_dir"]),
            filename="checkpoint-{epoch:02d}-{step}",
            every_n_train_steps=cfg.training.val_check_interval,
            save_top_k=-1,  # Save all checkpoints
        )

    # Loss tracking callback (saves loss plot to checkpoint dir)
    loss_plot_path = paths["checkpoint_dir"] / "loss_plot.png"
    loss_callback = LossCallback(file_path=str(loss_plot_path))

    # Validation metrics callback (binned MAE/overlap, box plot, mces_bucket confusion)
    val_metrics_callback = ValMetricsCallback(
        output_dir=str(paths["checkpoint_dir"]),
        mae_bin_edges=cfg.model.tasks.mces.mae_bin_edges,
        mces_max_value=cfg.model.tasks.mces.max_value,
        hit_at_k_n_decoys=cfg.model.tasks.mces.hit_at_k_n_decoys,
        hit_at_k_ks=cfg.model.tasks.mces.hit_at_k_ks,
    )

    # Progress logging callback (writes INFO lines to .err log file)
    progress_log_callback = ProgressLogCallback(
        log_every_n_steps=cfg.logging.get("progress_log_every_n_steps", 100)
    )

    # Optional early stopping: patience=0 means disabled
    early_stopping_callback = None
    patience = cfg.training.get("early_stopping_patience", 0)
    if patience and patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor="validation_loss",
            patience=patience,
            mode="min",
            verbose=True,
        )

    # Optional ICEBERG retrieval Hit@k, every check_every_n_steps steps
    # (expressed as a multiple of val_check_interval, since it only ever
    # runs alongside a regular validation check)
    iceberg_hit_rate_callback = None
    if cfg.iceberg_retrieval.enabled:
        check_every_n_val_checks = max(
            1,
            cfg.iceberg_retrieval.check_every_n_steps
            // cfg.training.val_check_interval,
        )
        iceberg_hit_rate_callback = IcebergHitRateCallback(
            mgf=cfg.iceberg_retrieval.mgf,
            candidates=cfg.iceberg_retrieval.candidates,
            candidate_tsv=list(cfg.iceberg_retrieval.candidate_tsv),
            iceberg_preds=list(cfg.iceberg_retrieval.iceberg_preds),
            batch_size=cfg.iceberg_retrieval.batch_size,
            check_every_n_val_checks=check_every_n_val_checks,
        )

    return (
        checkpoint_callback,
        checkpoint_n_steps_callback,
        loss_callback,
        early_stopping_callback,
        progress_log_callback,
        val_metrics_callback,
        iceberg_hit_rate_callback,
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
        "use_element_wise": cfg.model.features.use_element_wise,
        "use_cosine_distance": cfg.model.tasks.cosine_similarity.use_cosine_distance,
        "mces_max_value": cfg.model.tasks.mces.max_value,
        "use_mces_bucket_head": cfg.model.tasks.mces_bucket.enabled,
        "mces_bucket_bin_edges": cfg.model.tasks.mces_bucket.bin_edges,
        "mces_bucket_use_mlp": cfg.model.tasks.mces_bucket.use_mlp,
        "mces_bucket_loss_weight": cfg.model.tasks.mces_bucket.loss_weight,
        "use_contrastive_loss": cfg.model.tasks.contrastive.enabled,
        "contrastive_temperature": cfg.model.tasks.contrastive.temperature,
        "contrastive_loss_weight": cfg.model.tasks.contrastive.loss_weight,
        "weights": weights_mces,
        "lr": cfg.optimizer.lr,
        "use_adduct": cfg.model.features.use_adduct,
        "use_precursor_mz_for_model": cfg.model.features.use_precursor_mz,
        "use_ce": cfg.model.features.use_ce,
        "use_ion_activation": cfg.model.features.use_ion_activation,
        "use_ion_method": cfg.model.features.use_ion_method,
        "use_ion_mode": cfg.model.features.use_ion_mode,
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


_CSV_LOGGER_RESILIENCE_PATCHED = False


def _patch_csv_logger_header_rewrite_resilience() -> None:
    """Workaround for a Lightning CSVLogger bug: when metrics.csv gets
    rewritten with a wider header partway through a run, a stray row key
    can crash csv.DictWriter and kill the whole training run. Patches the
    rewrite step to drop unknown keys instead of crashing. Idempotent;
    if Lightning's internals change, the patch just fails to apply and
    becomes a no-op.
    """
    global _CSV_LOGGER_RESILIENCE_PATCHED
    if _CSV_LOGGER_RESILIENCE_PATCHED:
        return
    try:
        import csv

        from lightning.fabric.loggers.csv_logs import _ExperimentWriter

        def _resilient_rewrite_with_new_header(self, fieldnames):
            with self._fs.open(self.metrics_file_path, "r", newline="") as file:
                metrics = list(csv.DictReader(file))
            fieldname_set = set(fieldnames)
            for m in metrics:
                for bad_key in [k for k in m if k not in fieldname_set]:
                    del m[bad_key]
            with self._fs.open(self.metrics_file_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics)

        _ExperimentWriter._rewrite_with_new_header = _resilient_rewrite_with_new_header
        _CSV_LOGGER_RESILIENCE_PATCHED = True
    except Exception:
        logger.warning(
            "Could not apply the CSVLogger header-rewrite resilience patch "
            "(lightning internals may have changed) -- continuing without it.",
            exc_info=True,
        )


def _flatten_for_hparams(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict into dot-joined scalar keys, the shape
    TensorBoard's HParams plugin expects."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_for_hparams(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple)):
            flat[key] = str(v)
        elif v is None:
            flat[key] = "null"
        else:
            flat[key] = v
    return flat


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
    iceberg_hit_rate_callback: IcebergHitRateCallback | None = None,
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
            iceberg_hit_rate_callback,
        ]
        if cb is not None
    ]

    torch.set_float32_matmul_precision("high")

    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from simba.utils.config_utils import get_model_paths

    _patch_csv_logger_header_rewrite_resilience()

    checkpoint_dir = get_model_paths(cfg)["checkpoint_dir"]
    csv_logger = CSVLogger(save_dir=str(checkpoint_dir), name="", version="")
    tb_logger = TensorBoardLogger(save_dir=str(checkpoint_dir), name="tb_logs")
    tb_logger.log_hyperparams(
        _flatten_for_hparams(OmegaConf.to_container(cfg, resolve=True))
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        precision=cfg.hardware.precision,
        val_check_interval=cfg.training.val_check_interval,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        enable_checkpointing=checkpoint_callback is not None,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    return trainer


def _remove_duplicates_array(arr: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from numpy array."""
    return np.unique(arr, axis=0)
