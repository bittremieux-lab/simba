"""Inference workflow for SIMBA."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger
from simba.utils.plotting_spectrums import plot_pair_mols_plus_spectrum_png


# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs
sys.modules["simba.molecular_pairs"] = simba.core.data.molecule_pairs
sys.modules["simba.spectrum"] = simba.core.data.spectrum
sys.modules["simba.spectrum_ext"] = simba.core.data.spectrum


from simba.workflows.inference import (  # noqa: E402
    _get_ground_truth,
    _plot_cm,
    _plot_performance,
    _which_index,
    load_inference_data,
    load_model_for_inference,
    run_inference,
)


## PARAMETERS:
USE_ONLY_ADDUCT_ANALYSIS = False


def evaluate_predictions_metadata_analysis(
    cfg: DictConfig,
    pred_ed,
    pred_mces,
    dataloader_ed,
    dataloader_mces,
    output_dir: str,
):
    """Evaluate predictions and generate visualizations.

    Args:
        cfg: Hydra configuration object
        pred_ed: Edit distance predictions
        pred_mces: MCES predictions
        dataloader_ed: Edit distance dataloader
        dataloader_mces: MCES dataloader
        output_dir: Directory to save outputs

    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating predictions...")

    # Check for empty predictions
    if not pred_ed or not pred_mces:
        raise ValueError("Empty predictions received")

    # Get ground truth
    ed_true, _ = _get_ground_truth(dataloader_ed)
    _, mces_true = _get_ground_truth(dataloader_mces)

    # Flatten MCES predictions
    # pred_mces is list of batches, each batch is (emb, emb_sim_2)
    # emb_sim_2 (index 1) has shape (batch_size,) when use_cosine_distance=True
    pred_mces_mces_flat = []
    for batch_output in pred_mces:
        batch_preds = batch_output[1]  # emb_sim_2 tensor of shape (batch_size,)
        if batch_preds.dim() == 0:  # scalar tensor
            pred_mces_mces_flat.append(batch_preds.item())
        else:  # batch of predictions
            pred_mces_mces_flat.extend(batch_preds.cpu().numpy().tolist())
    pred_mces_mces_flat = np.array(pred_mces_mces_flat)

    # Flatten ED predictions
    # pred_ed is list of batches, each batch is (emb, emb_sim_2)
    # emb (index 0) has shape (batch_size, n_classes) - classification logits
    pred_ed_ed_flat = []
    for batch_output in pred_ed:
        batch_preds = batch_output[0]  # emb tensor of shape (batch_size, n_classes)
        # Convert logits to class predictions
        for pred_logits in batch_preds:
            class_idx = _which_index(pred_logits)
            pred_ed_ed_flat.append(class_idx)
    pred_ed_ed_flat = np.array(pred_ed_ed_flat, dtype=float)

    # Clean data
    ed_true = np.array(ed_true)
    mces_true = np.array(mces_true)

    # mask = ~np.isnan(pred_ed_ed_flat)
    # ed_true_clean = ed_true[mask]
    # pred_ed_ed_clean = pred_ed_ed_flat[mask]

    ed_true_clean = ed_true
    pred_ed_ed_clean = pred_ed_ed_flat

    # Edit distance correlation
    corr_model_ed, _ = spearmanr(ed_true_clean, pred_ed_ed_clean)
    mae_model_ed = mean_absolute_error(ed_true_clean, pred_ed_ed_clean)

    logger.info(f"Edit distance correlation: {corr_model_ed:.4f}")
    logger.info(f"Edit distance mean absolute error: {mae_model_ed:.4f}")

    # Plot confusion matrix
    _plot_cm(ed_true_clean, pred_ed_ed_clean, cfg, output_dir)

    # MCES evaluation
    counts, bins = TrainUtils.count_ranges(
        mces_true, number_bins=5, bin_sim_1=False, max_value=1
    )

    logger.info(f"MCES max value: {max(mces_true):.4f}")
    logger.info(f"MCES min value: {min(mces_true):.4f}")
    logger.info(f"MCES samples per bin: {counts}")

    # Remove threshold values
<<<<<<< HEAD
    #mces_true_original = mces_true.copy()
    #mces_true = mces_true[mces_true_original != 0.5]
    #pred_mces_mces_flat = pred_mces_mces_flat[mces_true_original != 0.5]

=======
    # mces_true_original = mces_true.copy()
    # mces_true = mces_true[mces_true_original != 0.5]
    # pred_mces_mces_flat = pred_mces_mces_flat[mces_true_original != 0.5]
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

    if len(mces_true) == 0 or len(pred_mces_mces_flat) == 0:
        logger.warning("No MCES samples after filtering, skipping MCES correlation")
        corr_model_mces = float("nan")
        mae_model_mces = float("nan")

    else:
        corr_model_mces, _ = spearmanr(mces_true, pred_mces_mces_flat)
        mae_model_mces = mean_absolute_error(mces_true, pred_mces_mces_flat)
    logger.info(f"MCES/Tanimoto correlation: {corr_model_mces:.4f}")
    logger.info(
        f"MCES/Tanimoto mean absolute error: {cfg.data.mces20_max_value * mae_model_mces:.4f}"
    )

    # Denormalize if using MCES20
    if not cfg.data.use_tanimoto:
        mces_true = cfg.data.mces20_max_value * (1 - mces_true)
        pred_mces_mces_flat = cfg.data.mces20_max_value * (1 - pred_mces_mces_flat)
<<<<<<< HEAD
    

    print(f'Size of mces prediction: {pred_mces_mces_flat.shape}')
    print(f'Size of mces ground truth: {mces_true.shape}')
=======

    print(f"Size of mces prediction: {pred_mces_mces_flat.shape}")
    print(f"Size of mces ground truth: {mces_true.shape}")
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    # Plot performance
    _plot_performance(mces_true, pred_mces_mces_flat, cfg, output_dir)

    return {
        "ed_correlation": corr_model_ed,
        "mces_correlation": corr_model_mces,
        "ed_true": ed_true_clean,
        "ed_pred": pred_ed_ed_clean,
        "mces_true": mces_true,
        "mces_pred": pred_mces_mces_flat,
    }


def inference_metadata_analysis(cfg: DictConfig) -> dict:
    """Main inference workflow.

    Args:
        cfg: Hydra configuration object

    Returns:
        dict: Evaluation metrics
    """
    # Determine checkpoint path
    checkpoint_dir = cfg.paths.checkpoint_dir
    if not cfg.inference.use_last_model:
        checkpoint_path = os.path.join(checkpoint_dir, cfg.checkpoints.best_model_name)
        model_name = "best model"
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
        model_name = "last checkpoint"

    logger.info(f"Using {model_name}: {checkpoint_path}")

    # Set output directory (default to checkpoint_dir if not specified)
    output_dir = cfg.paths.get("output_dir") or checkpoint_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    molecule_pairs_ed, molecule_pairs_mces, _ = load_inference_data(cfg)
<<<<<<< HEAD
    
    print('Sample of original distances')
    print(molecule_pairs_mces.pair_distances[0:10])
    # Prepare dataloaders

    print(f'The size of the molecule pairs that comes into prepare: {len(molecule_pairs_mces.original_spectra)}')
    dataloader_ed, dataloader_mces, mols_ed, mols_mces = prepare_inference_dataloaders_return_spectrums(
        cfg, molecule_pairs_ed, molecule_pairs_mces
    )

    print('Sample of distances after uniformise')
    print(mols_mces.pair_distances[0:10])

    print(f'The size of the molecule pairs that comes out of prepare: {len(mols_mces.original_spectra)}')

    print(f'Shape of distances: {mols_mces.pair_distances.shape}')
=======

    print("Sample of original distances")
    print(molecule_pairs_mces.pair_distances[0:10])
    # Prepare dataloaders

    print(
        f"The size of the molecule pairs that comes into prepare: {len(molecule_pairs_mces.original_spectra)}"
    )
    dataloader_ed, dataloader_mces, mols_ed, mols_mces = (
        prepare_inference_dataloaders_return_spectrums(
            cfg, molecule_pairs_ed, molecule_pairs_mces
        )
    )

    print("Sample of distances after uniformise")
    print(mols_mces.pair_distances[0:10])

    print(
        f"The size of the molecule pairs that comes out of prepare: {len(mols_mces.original_spectra)}"
    )

    print(f"Shape of distances: {mols_mces.pair_distances.shape}")
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    # Load model
    model = load_model_for_inference(cfg, checkpoint_path)

    # Run inference
    pred_ed, pred_mces = run_inference(cfg, model, dataloader_ed, dataloader_mces)
<<<<<<< HEAD
    
=======

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    # Run inference
    # Evaluate
    metrics = evaluate_predictions_metadata_analysis(
        cfg, pred_ed, pred_mces, dataloader_ed, dataloader_mces, output_dir
    )
<<<<<<< HEAD
    
=======

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    logger.info(f"Results saved to: {output_dir}")

    return metrics, mols_mces


<<<<<<< HEAD



=======
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
def prepare_inference_dataloaders_return_spectrums(
    cfg: DictConfig,
    molecule_pairs_ed,
    molecule_pairs_mces,
):
    """Prepare dataloaders for inference.

    Args:
        cfg: Hydra configuration object
        molecule_pairs_ed: Molecule pairs for edit distance
        molecule_pairs_mces: Molecule pairs for MCES

    Returns:
        tuple: (dataloader_ed, dataloader_mces)
    """
    # Uniformize if needed
    if cfg.inference.uniformize_testing:
        logger.info("Uniformizing pairs across bins...")
        bins_uniformise = cfg.data.edit_distance_n_classes - 1

        molecule_pairs_ed_uniform, _ = TrainUtils.uniformise(
            molecule_pairs_ed,
            number_bins=bins_uniformise,
            return_binned_list=True,
            bin_sim_1=True,
            ordinal_classification=True,
        )
        molecule_pairs_mces_uniform, _ = TrainUtils.uniformise(
            molecule_pairs_mces,
            number_bins=bins_uniformise,
            return_binned_list=True,
            bin_sim_1=False,
        )
    else:
        molecule_pairs_ed_uniform = molecule_pairs_ed
        molecule_pairs_mces_uniform = molecule_pairs_mces

    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataset_ed = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_ed_uniform,
        training=False,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
        use_ion_mode=cfg.model.features.use_ion_mode,
    )
    dataloader_ed = DataLoader(
        dataset_ed, batch_size=cfg.inference.batch_size, shuffle=False
    )
<<<<<<< HEAD
    
    print(f'DEBUG in inf_m_ ORIGINAL SPECTRA: {len(molecule_pairs_mces_uniform.original_spectra)}')
    print(f'DEBUG in inf_m_ UNIQUE SPECTRA: {len(molecule_pairs_mces_uniform.spectra)}')
=======

    print(
        f"DEBUG in inf_m_ ORIGINAL SPECTRA: {len(molecule_pairs_mces_uniform.original_spectra)}"
    )
    print(f"DEBUG in inf_m_ UNIQUE SPECTRA: {len(molecule_pairs_mces_uniform.spectra)}")
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    dataset_mces = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_mces_uniform,
        training=False,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
        use_ion_mode=cfg.model.features.use_ion_mode,
    )
    dataloader_mces = DataLoader(
        dataset_mces, batch_size=cfg.inference.batch_size, shuffle=False
    )
    _, mces_true_temp = _get_ground_truth(dataloader_mces)
<<<<<<< HEAD
    
    print(f'DEBUG in inf_m: size of mces_true_temp: {len(mces_true_temp)}')
    return dataloader_ed, dataloader_mces, molecule_pairs_ed_uniform, molecule_pairs_mces_uniform


=======

    print(f"DEBUG in inf_m: size of mces_true_temp: {len(mces_true_temp)}")
    return (
        dataloader_ed,
        dataloader_mces,
        molecule_pairs_ed_uniform,
        molecule_pairs_mces_uniform,
    )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5


print("Inference for metadata analysis")

<<<<<<< HEAD
import hydra
from omegaconf import OmegaConf
=======
import hydra  # noqa: E402
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5


# -----------------------------------------------------------------------------
# Metadata sensitivity analysis configuration
# -----------------------------------------------------------------------------
<<<<<<< HEAD
#CODE_NAME = "train_metadata_all_adducts_seb_20260218_adduct_fixing_2"
=======
# CODE_NAME = "train_metadata_all_adducts_seb_20260218_adduct_fixing_2"
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
CODE_NAME = "ms2_reference_fixed_split_fixed_encoding"
ADDUCTS_TO_COMPARE = ["", "M+H", "M-H", "M+Na", "M+FA-H", "M+NH4"]
MAX_MCES_FOR_ADDUCT_ANALYSIS = 10
N_EXAMPLES_TO_EXPORT = 10


def set_metadata_features(
    cfg: DictConfig,
    *,
    use_adduct: int,
    use_ce: int,
    use_ion_activation: int,
    use_ion_method: int,
    use_ion_mode: int,
) -> DictConfig:
    """Toggle metadata feature flags in one place."""
    cfg.model.features.use_adduct = use_adduct
    cfg.model.features.use_ce = use_ce
    cfg.model.features.use_ion_activation = use_ion_activation
    cfg.model.features.use_ion_method = use_ion_method
    cfg.model.features.use_ion_mode = use_ion_mode
    return cfg


def configure_baseline_features(cfg: DictConfig) -> DictConfig:
    """Use all metadata features."""
    return set_metadata_features(
        cfg,
        use_adduct=1,
        use_ce=1,
        use_ion_activation=1,
        use_ion_method=1,
        use_ion_mode=1,
    )


<<<<<<< HEAD
def configure_sensitivity_features(cfg: DictConfig, use_only_adduct_analysis: bool) -> DictConfig:
=======
def configure_sensitivity_features(
    cfg: DictConfig, use_only_adduct_analysis: bool
) -> DictConfig:
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    """Remove the target metadata features for the sensitivity run."""
    if use_only_adduct_analysis:
        # Keep CE/activation/method; remove adduct-related information.
        return set_metadata_features(
            cfg,
            use_adduct=0,
            use_ion_mode=0,
            use_ce=1,
            use_ion_activation=1,
            use_ion_method=1,
        )

    # Remove all metadata features.
    return set_metadata_features(
        cfg,
        use_adduct=0,
        use_ion_mode=0,
        use_ce=0,
        use_ion_activation=0,
        use_ion_method=0,
    )


def run_metadata_variant(cfg: DictConfig, label: str):
    """Run inference for one metadata-feature configuration."""
    logger.info("Running metadata-analysis variant: %s", label)
    metrics, mols_mces = inference_metadata_analysis(cfg)
    errors = np.abs(metrics["mces_true"] - metrics["mces_pred"])
    return metrics, mols_mces, errors


<<<<<<< HEAD
def get_pair_spectra(mols_mces, pair_index: int, ):
    """Return the two original spectra for a pair index."""
    unique_index_0 = mols_mces.pair_distances[pair_index, 0]
    unique_index_1 = mols_mces.pair_distances[pair_index, 1]
    spec_0 = mols_mces.get_original_spectrum_from_unique_index(unique_index_0, pair=0, )
    spec_1 = mols_mces.get_original_spectrum_from_unique_index(unique_index_1, pair=1, )
    return spec_0, spec_1


def adduct_pair_matches(adduct_0: str, adduct_1: str, target_0: str, target_1: str) -> bool:
    """Check unordered match between observed adducts and target adduct strings."""
    return (
        (target_0 in adduct_0 and target_1 in adduct_1)
        or (target_1 in adduct_0 and target_0 in adduct_1)
    )

=======
def get_pair_spectra(
    mols_mces,
    pair_index: int,
):
    """Return the two original spectra for a pair index."""
    unique_index_0 = mols_mces.pair_distances[pair_index, 0]
    unique_index_1 = mols_mces.pair_distances[pair_index, 1]
    spec_0 = mols_mces.get_original_spectrum_from_unique_index(
        unique_index_0,
        pair=0,
    )
    spec_1 = mols_mces.get_original_spectrum_from_unique_index(
        unique_index_1,
        pair=1,
    )
    return spec_0, spec_1


def adduct_pair_matches(
    adduct_0: str, adduct_1: str, target_0: str, target_1: str
) -> bool:
    """Check unordered match between observed adducts and target adduct strings."""
    return (target_0 in adduct_0 and target_1 in adduct_1) or (
        target_1 in adduct_0 and target_0 in adduct_1
    )


>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
def find_metadata_pair_indexes(
    mols_mces,
    metrics_baseline: dict,
    target_meta_0,
    target_meta_1,
    max_mces: float = MAX_MCES_FOR_ADDUCT_ANALYSIS,
<<<<<<< HEAD
    field= None,
    filter_high_similarity_pairs=False,
) -> list[int]:
    '''
    find pairs matching a specific combination for a field
    '''
=======
    field=None,
    filter_high_similarity_pairs=False,
) -> list[int]:
    """
    find pairs matching a specific combination for a field
    """
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    matched_indexes = []
    str_target_meta_0 = [str(t) for t in target_meta_0]
    str_target_meta_1 = [str(t) for t in target_meta_1]
    for pair_index in range(mols_mces.pair_distances.shape[0]):
<<<<<<< HEAD

=======
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
        try:
            spec_0, spec_1 = get_pair_spectra(mols_mces, pair_index)
            meta_0 = spec_0.params.get(field, "")
            meta_1 = spec_1.params.get(field, "")

<<<<<<< HEAD
            
            if (str(meta_0) in str_target_meta_0) and (str(meta_1) in str_target_meta_1):
=======
            if (str(meta_0) in str_target_meta_0) and (
                str(meta_1) in str_target_meta_1
            ):
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
                if filter_high_similarity_pairs:
                    if metrics_baseline["mces_true"][pair_index] < max_mces:
                        matched_indexes.append(pair_index)
                else:
                    matched_indexes.append(pair_index)
<<<<<<< HEAD
        except:
            print(f'Problem finding the spectra for pair_index: {pair_index}')

    return matched_indexes

=======
        except Exception:
            print(f"Problem finding the spectra for pair_index: {pair_index}")

    return matched_indexes


>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
def find_adduct_pair_indexes(
    mols_mces,
    metrics_baseline: dict,
    target_adduct_0: str,
    target_adduct_1: str,
    max_mces: float = MAX_MCES_FOR_ADDUCT_ANALYSIS,
) -> list[int]:
    """Find pair indexes matching an adduct combination and an MCES threshold."""
    matched_indexes = []
    for pair_index in range(mols_mces.pair_distances.shape[0]):
        spec_0, spec_1 = get_pair_spectra(mols_mces, pair_index)
        adduct_0 = spec_0.params.get("adduct", "")
        adduct_1 = spec_1.params.get("adduct", "")

<<<<<<< HEAD
        if adduct_pair_matches(adduct_0, adduct_1, target_adduct_0, target_adduct_1):
            if metrics_baseline["mces_true"][pair_index] < max_mces:
                matched_indexes.append(pair_index)
=======
        if (
            adduct_pair_matches(adduct_0, adduct_1, target_adduct_0, target_adduct_1)
            and metrics_baseline["mces_true"][pair_index] < max_mces
        ):
            matched_indexes.append(pair_index)
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

    return matched_indexes


def plot_error_scatter_for_adduct_pair(
    baseline_errors: np.ndarray,
    sensitivity_errors: np.ndarray,
    pair_indexes: list[int],
    adduct_0: str,
    adduct_1: str,
    out_dir: Path,
) -> None:
    """Plot baseline vs sensitivity error for one adduct combination."""
    if not pair_indexes:
<<<<<<< HEAD
        logger.warning("No pairs found for adduct combination %s,%s", adduct_0, adduct_1)
=======
        logger.warning(
            "No pairs found for adduct combination %s,%s", adduct_0, adduct_1
        )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
        return

    filtered_sensitivity = sensitivity_errors[pair_indexes]
    filtered_baseline = baseline_errors[pair_indexes]
    affected_pairs = np.sum((filtered_sensitivity - filtered_baseline) > 0)
    affected_fraction = affected_pairs / len(filtered_sensitivity)

    plt.figure()
<<<<<<< HEAD
    plt.scatter(filtered_baseline, filtered_sensitivity, label=f"{adduct_0},{adduct_1}", alpha=0.10)
=======
    plt.scatter(
        filtered_baseline,
        filtered_sensitivity,
        label=f"{adduct_0},{adduct_1}",
        alpha=0.10,
    )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    plt.plot(np.arange(0, 40), np.arange(0, 40), linestyle="--", c="k")
    plt.legend()
    plt.xlim([0, 40])
    plt.ylim([0, 40])
<<<<<<< HEAD
    plt.title(f"Proportion of pairs more affected by absence of adduct: {affected_fraction:.3f}")
=======
    plt.title(
        f"Proportion of pairs more affected by absence of adduct: {affected_fraction:.3f}"
    )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    plt.xlabel("Error using all metadata")
    plt.ylabel("Error removing target variable")
    plt.savefig(out_dir / f"adduct_analysis_{adduct_0}_{adduct_1}.png")
    plt.close()


def plot_error_histogram_for_adduct_pair(
    baseline_errors: np.ndarray,
    sensitivity_errors: np.ndarray,
    pair_indexes: list[int],
    adduct_0: str,
    adduct_1: str,
    out_dir: Path,
) -> None:
    """Plot error distributions for one adduct combination."""
    if not pair_indexes:
        return

    filtered_sensitivity = sensitivity_errors[pair_indexes]
    filtered_baseline = baseline_errors[pair_indexes]
    bins = np.arange(0, 40 + 2.5, 2.5)

    plt.figure()
<<<<<<< HEAD
    plt.hist(filtered_sensitivity, density=True, bins=bins, alpha=0.3, label="All metadata - target metadata")
    plt.hist(filtered_baseline, density=True, bins=bins, alpha=0.3, label="All metadata")
=======
    plt.hist(
        filtered_sensitivity,
        density=True,
        bins=bins,
        alpha=0.3,
        label="All metadata - target metadata",
    )
    plt.hist(
        filtered_baseline, density=True, bins=bins, alpha=0.3, label="All metadata"
    )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    plt.grid(alpha=0.3)
    plt.title(f"{adduct_0},{adduct_1} pairs")
    plt.xlabel("Prediction error (MCES)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(out_dir / f"hist_together_{adduct_0}_{adduct_1}.png")
    plt.close()


def run_adduct_pair_analysis(
    mols_mces,
    metrics_baseline: dict,
    baseline_errors: np.ndarray,
    sensitivity_errors: np.ndarray,
    out_dir: Path,
    adducts: list[str] = ADDUCTS_TO_COMPARE,
) -> None:
    """Generate scatter and histogram plots for all adduct combinations."""
    for adduct_0 in adducts:
        for adduct_1 in adducts:
            pair_indexes = find_adduct_pair_indexes(
                mols_mces=mols_mces,
                metrics_baseline=metrics_baseline,
                target_adduct_0=adduct_0,
                target_adduct_1=adduct_1,
            )
            plot_error_scatter_for_adduct_pair(
<<<<<<< HEAD
                baseline_errors, sensitivity_errors, pair_indexes, adduct_0, adduct_1, out_dir
            )
            plot_error_histogram_for_adduct_pair(
                baseline_errors, sensitivity_errors, pair_indexes, adduct_0, adduct_1, out_dir
=======
                baseline_errors,
                sensitivity_errors,
                pair_indexes,
                adduct_0,
                adduct_1,
                out_dir,
            )
            plot_error_histogram_for_adduct_pair(
                baseline_errors,
                sensitivity_errors,
                pair_indexes,
                adduct_0,
                adduct_1,
                out_dir,
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
            )


def get_high_impact_indexes(
    metrics_sensitivity: dict,
    baseline_errors: np.ndarray,
    sensitivity_errors: np.ndarray,
    n_examples: int = N_EXAMPLES_TO_EXPORT,
    max_mces: float = MAX_MCES_FOR_ADDUCT_ANALYSIS,
) -> np.ndarray:
    """Return indexes where removing metadata hurt prediction error the most."""
    diff_pred = np.abs(baseline_errors - sensitivity_errors)
    diff_pred[sensitivity_errors < baseline_errors] = 0
    diff_pred[metrics_sensitivity["mces_true"] > max_mces] = 0
    return np.argsort(diff_pred)[-n_examples:]


def build_spectrum_params_df(spec_0, spec_1) -> pd.DataFrame:
    """Create a compact metadata table for a spectrum pair."""
    keys = ["pepmass", "ionmode", "adduct", "ion_activation", "ionization_method", "ce"]
    data = {}

    for key in keys:
        if key == "pepmass":
            data[key] = [
                round(spec_0.params.get(key, [None])[0], 2),
                round(spec_1.params.get(key, [None])[0], 2),
            ]
        else:
            data[key] = [spec_0.params.get(key, None), spec_1.params.get(key, None)]

    return pd.DataFrame(data, index=["spec_0", "spec_1"])


<<<<<<< HEAD
def save_spectrum_params_table(df: pd.DataFrame, example_id: int, out_dir: Path) -> None:
=======
def save_spectrum_params_table(
    df: pd.DataFrame, example_id: int, out_dir: Path
) -> None:
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    """Save spectrum-pair metadata as CSV and PDF table."""
    df.to_csv(out_dir / f"example_{example_id}.csv", index=True)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    fig.savefig(out_dir / f"example_{example_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_high_impact_examples(
    mols_mces,
    metrics_baseline: dict,
    metrics_sensitivity: dict,
    baseline_errors: np.ndarray,
    sensitivity_errors: np.ndarray,
    out_dir: Path,
    n_examples: int = N_EXAMPLES_TO_EXPORT,
) -> None:
    """Export molecule/spectrum plots and metadata tables for high-impact pairs."""
    high_impact_indexes = get_high_impact_indexes(
        metrics_sensitivity, baseline_errors, sensitivity_errors, n_examples=n_examples
    )

    for example_id, pair_index in enumerate(high_impact_indexes):
<<<<<<< HEAD
        logger.info("Exporting high-impact example %s for pair index %s", example_id, pair_index)
=======
        logger.info(
            "Exporting high-impact example %s for pair index %s", example_id, pair_index
        )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

        spec_0, spec_1 = get_pair_spectra(mols_mces, pair_index)
        plot_pair_mols_plus_spectrum_png(
            pair_index=0,
            all_spectrums_query=[spec_0],
            all_spectrums_reference=[spec_1],
            pairs_interesting=[{"indexes": [0, 0]}],
            metrics={
                "mces_gt": metrics_baseline["mces_true"][pair_index],
                "mces_pred": metrics_baseline["mces_pred"][pair_index],
                "ed_gt": 0,
                "ed_pred": 0,
                "mod_cos": 0,
            },
            mz_min=0,
            mz_max=max(max(spec_0.mz) + 10, max(spec_1.mz) + 10),
            out_dir=str(out_dir),
            out_name_tpl=str(out_dir / f"example_with_mol_{example_id}.png"),
        )

        params_df = build_spectrum_params_df(spec_0, spec_1)
        save_spectrum_params_table(params_df, example_id, out_dir)

<<<<<<< HEAD
def performance_per_msn_level(metrics, mols_mces):
    matched_indexes_per_level = {}
    mslevels =[s.params['mslevel'] for s in mols_mces.spectra]
    mslevels =np.unique(mslevels)
    mslevels= [[m] for m in mslevels]
    
=======

def performance_per_msn_level(metrics, mols_mces):
    matched_indexes_per_level = {}
    mslevels = [s.params["mslevel"] for s in mols_mces.spectra]
    mslevels = np.unique(mslevels)
    mslevels = [[m] for m in mslevels]

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    def safe_spearman(y_true, y_pred):
        if len(y_true) < 2:
            return np.nan
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return np.nan
        corr, _ = spearmanr(y_true, y_pred)
        return corr

    for msn_level in mslevels:
        matched_indexes = find_metadata_pair_indexes(
            mols_mces=mols_mces,
            metrics_baseline=metrics,
            target_meta_0=msn_level,
            target_meta_1=msn_level,
            field="mslevel",
            filter_high_similarity_pairs=False,
        )
        matched_indexes_per_level[str(msn_level)] = matched_indexes

    for msn_level in mslevels:
        idx = matched_indexes_per_level[str(msn_level)]

        mces_true = metrics["mces_true"][idx]
        mces_pred = metrics["mces_pred"][idx]

        ed_true = metrics["ed_true"][idx]
        ed_pred = metrics["ed_pred"][idx]

        mae_model_mces = mean_absolute_error(mces_true, mces_pred)
        corr_model_mces = safe_spearman(mces_true, mces_pred)

        ed_accuracy = np.mean(ed_true == ed_pred)
        corr_model_ed = safe_spearman(ed_true, ed_pred)
        mae_model_ed = mean_absolute_error(ed_true, ed_pred)

        print(f"For msn level: {msn_level}")
        print(f"N pairs: {len(idx)}")
        print(f"MCES MAE: {mae_model_mces:.4f}")
        print(f"MCES Spearman correlation: {corr_model_mces:.4f}")
        print(f"ED accuracy: {ed_accuracy:.4f}")
        print(f"ED MAE: {mae_model_ed:.4f}")
        print(f"ED Spearman correlation: {corr_model_ed:.4f}")
        print("-" * 50)


def run_metadata_sensitivity_analysis(cfg: DictConfig) -> None:
    """Run baseline and sensitivity inference, then export comparison analyses."""
<<<<<<< HEAD
    
    
    #Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
=======

    # Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

    # Determine checkpoint path
    checkpoint_dir = cfg.paths.checkpoint_dir
    if not cfg.inference.use_last_model:
        checkpoint_path = os.path.join(checkpoint_dir, cfg.checkpoints.best_model_name)
        model_name = "best model"
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
        model_name = "last checkpoint"

    logger.info(f"Using {model_name}: {checkpoint_path}")

    # Set output directory (default to checkpoint_dir if not specified)
    output_dir = cfg.paths.get("output_dir") or checkpoint_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD

    cfg = configure_baseline_features(cfg)
    metrics_baseline, mols_mces_baseline, baseline_errors = run_metadata_variant(cfg, label="baseline_all_metadata")
=======
    cfg = configure_baseline_features(cfg)
    metrics_baseline, mols_mces_baseline, baseline_errors = run_metadata_variant(
        cfg, label="baseline_all_metadata"
    )
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

    ## analysis per msn level
    performance_per_msn_level(metrics_baseline, mols_mces_baseline)

    cfg = configure_sensitivity_features(cfg, USE_ONLY_ADDUCT_ANALYSIS)
<<<<<<< HEAD
    metrics_sensitivity, mols_mces_sensitivity, sensitivity_errors = run_metadata_variant(
        cfg, label="sensitivity_without_target_metadata"
    )


=======
    metrics_sensitivity, mols_mces_sensitivity, sensitivity_errors = (
        run_metadata_variant(cfg, label="sensitivity_without_target_metadata")
    )

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
    run_adduct_pair_analysis(
        mols_mces=mols_mces_sensitivity,
        metrics_baseline=metrics_baseline,
        baseline_errors=baseline_errors,
        sensitivity_errors=sensitivity_errors,
        out_dir=Path(cfg.paths.checkpoint_dir),
    )

    export_high_impact_examples(
        mols_mces=mols_mces_sensitivity,
        metrics_baseline=metrics_baseline,
        metrics_sensitivity=metrics_sensitivity,
        baseline_errors=baseline_errors,
        sensitivity_errors=sensitivity_errors,
        out_dir=Path(cfg.paths.checkpoint_dir),
    )

    return metrics_baseline
<<<<<<< HEAD
    

@hydra.main(version_base=None, config_path="/home/spiedrahita/simba/simba/configs", config_name="config")
=======


@hydra.main(
    version_base=None,
    config_path="/home/spiedrahita/simba/simba/configs",
    config_name="config",
)
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
def main(cfg: DictConfig):
    """Hydra entry point for the metadata sensitivity analysis."""
    run_metadata_sensitivity_analysis(cfg)


if __name__ == "__main__":
    main()
