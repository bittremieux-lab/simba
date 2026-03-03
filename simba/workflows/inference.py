"""Inference workflow for SIMBA."""

import copy
import os
import sys
from pathlib import Path

import dill
import lightning.pytorch as pl
import numpy as np
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import simba.core.data.molecular_pairs
import simba.core.data.molecule_pairs_opt
import simba.core.data.spectrum
from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.molecule_pairs_opt import MoleculePairsOpt
from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.models.ordinal.embedder_multitask import EmbedderMultitask
from simba.core.models.ordinal.load_data_multitasking import LoadDataMultitasking
from simba.core.models.transformers.postprocessing import Postprocessing
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger


# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs_opt
sys.modules["simba.molecular_pairs"] = simba.core.data.molecular_pairs
sys.modules["simba.spectrum"] = simba.core.data.spectrum
sys.modules["simba.spectrum_ext"] = simba.core.data.spectrum


def load_inference_data(cfg: DictConfig):
    """Load preprocessed data for inference.

    Args:
        cfg: Hydra configuration object

    Returns:
        tuple: (molecule_pairs_ed, molecule_pairs_mces, pair_distances)
    """
    preprocessing_dir = cfg.paths.preprocessing_dir
    preprocessing_pickle = cfg.inference.preprocessing_pickle

    dataset_path = os.path.join(preprocessing_dir, preprocessing_pickle)

    logger.info(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "rb") as file:
        dataset = dill.load(file)

    # Check if lightweight format and reconstruct if needed
    if dataset.get("format_version") == "lightweight":
        logger.info("Detected lightweight format - reconstructing molecule_pairs_test")

        mgf_path = dataset["mgf_path"]
        all_spectra = PreprocessingSimba.load_spectra(mgf_path, cfg)

        # Create spectrum lookup by MGF index
        spectra_by_idx = {s.mgf_index: s for s in all_spectra}

        # Reconstruct test molecule pairs
        if "df_smiles_test" in dataset and "spectrum_indexes_test" in dataset:
            df_smiles = dataset["df_smiles_test"]
            spectrum_indexes = dataset["spectrum_indexes_test"]

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
        else:
            raise ValueError("Test split not found in lightweight format dataset")
    else:
        # Full format
        molecule_pairs = dataset["molecule_pairs_test"]

    molecule_pairs_ed = copy.deepcopy(molecule_pairs)
    molecule_pairs_mces = copy.deepcopy(molecule_pairs)

    # Load pair distances
    logger.info("Loading pair distances...")
    pair_distances = LoadMCES.merge_numpy_arrays(
        cfg.paths.preprocessing_dir_train or preprocessing_dir,
        prefix="ed_mces_indexes_tani_incremental_test",
        use_edit_distance=cfg.data.use_edit_distance,
        use_multitask=cfg.data.use_multitask,
    )
    pair_distances = _remove_duplicate_pairs(pair_distances)

    molecule_pairs_ed.pair_distances = pair_distances[:, 0:3]
    molecule_pairs_ed.extra_distances = pair_distances[:, 3]
    logger.info(f"{len(molecule_pairs_ed)} pairs remain after removing duplicates")

    molecule_pairs_mces.pair_distances = pair_distances[:, [0, 1, 3]]
    molecule_pairs_mces.extra_distances = pair_distances[:, 3]

    return molecule_pairs_ed, molecule_pairs_mces, pair_distances


def prepare_inference_dataloaders(
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
    dataset_ed = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_ed_uniform,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
    )
    dataloader_ed = DataLoader(
        dataset_ed, batch_size=cfg.inference.batch_size, shuffle=False
    )

    dataset_mces = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_mces_uniform,
        max_num_peaks=int(cfg.model.transformer.context_length),
        use_adduct=cfg.model.features.use_adduct,
        use_ce=cfg.model.features.use_ce,
        use_ion_activation=cfg.model.features.use_ion_activation,
        use_ion_method=cfg.model.features.use_ion_method,
    )
    dataloader_mces = DataLoader(
        dataset_mces, batch_size=cfg.inference.batch_size, shuffle=False
    )

    return dataloader_ed, dataloader_mces


def load_model_for_inference(cfg: DictConfig, checkpoint_path: str):
    """Load trained model for inference.

    Args:
        cfg: Hydra configuration object
        checkpoint_path: Path to model checkpoint

    Returns:
        EmbedderMultitask: Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}...")

    load_kwargs = {
        "d_model": int(cfg.model.transformer.d_model),
        "n_layers": int(cfg.model.transformer.n_layers),
        "n_classes": cfg.data.edit_distance_n_classes,
        "use_gumbel": cfg.model.tasks.edit_distance.use_gumbel,
        "use_element_wise": True,
        "use_cosine_distance": cfg.model.tasks.cosine_similarity.use_cosine_distance,
        "use_edit_distance_regresion": cfg.data.use_edit_distance_regression,
        "strict": False,
        "use_adduct": cfg.model.features.use_adduct,
        "use_ce": cfg.model.features.use_ce,
        "use_ion_activation": cfg.model.features.use_ion_activation,
        "use_ion_method": cfg.model.features.use_ion_method,
    }

    model = EmbedderMultitask.load_from_checkpoint(checkpoint_path, **load_kwargs)
    model.eval()

    return model


def run_inference(
    cfg: DictConfig,
    model: EmbedderMultitask,
    dataloader_ed,
    dataloader_mces,
):
    """Run inference on test data.

    Args:
        cfg: Hydra configuration object
        model: Trained model
        dataloader_ed: Dataloader for edit distance
        dataloader_mces: Dataloader for MCES

    Returns:
        tuple: (pred_ed, pred_mces) predictions
    """
    logger.info("Running inference...")

    trainer = pl.Trainer(
        max_epochs=2,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        devices=cfg.inference.devices,
        accelerator=cfg.inference.accelerator,
    )

    pred_ed = trainer.predict(model, dataloader_ed)
    pred_mces = trainer.predict(model, dataloader_mces)

    if pred_ed is None or pred_mces is None:
        raise ValueError(
            "Prediction failed - one or both dataloaders returned no predictions"
        )

    return pred_ed, pred_mces


def evaluate_predictions(
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
    ed_true, _ = Postprocessing.get_similarities_multitasking(dataloader_ed)
    _, mces_true = Postprocessing.get_similarities_multitasking(dataloader_mces)

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

    mask = ~np.isnan(pred_ed_ed_flat)
    ed_true_clean = ed_true[mask]
    pred_ed_ed_clean = pred_ed_ed_flat[mask]

    # Edit distance correlation
    corr_model_ed, _ = spearmanr(ed_true_clean, pred_ed_ed_clean)
    logger.info(f"Edit distance correlation: {corr_model_ed:.4f}")

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
    mces_true_original = mces_true.copy()
    mces_true = mces_true[mces_true_original != 0.5]
    pred_mces_mces_flat = pred_mces_mces_flat[mces_true_original != 0.5]

    if len(mces_true) == 0 or len(pred_mces_mces_flat) == 0:
        logger.warning("No MCES samples after filtering, skipping MCES correlation")
        corr_model_mces = float("nan")
    else:
        corr_model_mces, _ = spearmanr(mces_true, pred_mces_mces_flat)

    logger.info(f"MCES/Tanimoto correlation: {corr_model_mces:.4f}")

    # Denormalize if using MCES20
    if not cfg.data.use_tanimoto:
        mces_true = cfg.data.mces20_max_value * (1 - mces_true)
        pred_mces_mces_flat = cfg.data.mces20_max_value * (1 - pred_mces_mces_flat)

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


def inference(cfg: DictConfig) -> dict:
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

    # Prepare dataloaders
    dataloader_ed, dataloader_mces = prepare_inference_dataloaders(
        cfg, molecule_pairs_ed, molecule_pairs_mces
    )

    # Load model
    model = load_model_for_inference(cfg, checkpoint_path)

    # Run inference
    pred_ed, pred_mces = run_inference(cfg, model, dataloader_ed, dataloader_mces)

    # Evaluate
    metrics = evaluate_predictions(
        cfg, pred_ed, pred_mces, dataloader_ed, dataloader_mces, output_dir
    )

    logger.info(f"Results saved to: {output_dir}")

    return metrics


# Helper functions
def _remove_duplicate_pairs(array):
    """Remove duplicate pairs from array."""
    seen = set()
    filtered_rows = []

    for row in array:
        key = tuple(sorted(row[:2]))
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    return np.array(filtered_rows)


def _which_index(p) -> int:
    """Get the index of the maximum value."""
    return np.argmax(p)


def _plot_cm(
    true,
    preds,
    cfg: DictConfig,
    output_dir: str,
    file_name: str = "cm.png",
) -> None:
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, confusion_matrix

    cm = confusion_matrix(true, preds)
    accuracy = accuracy_score(true, preds)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    plt.figure(figsize=(10, 7))
    labels = [">5", "4", "3", "2", "1", "0"]

    im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized frequency", fontsize=15)

    threshold = cm_normalized.max() / 2.0

    # Annotate cells
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text_color = "white" if cm_normalized[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.xlabel("Substructure edit distance - Prediction", fontsize=15)
    plt.ylabel("Substructure edit distance - Ground truth", fontsize=15)
    plt.title(
        f"Confusion Matrix (Normalized), Acc: {accuracy:.2f}, Samples: {preds.shape[0]}",
        fontsize=15,
    )

    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def _plot_performance(mces_true, mces_pred, cfg: DictConfig, output_dir: str) -> None:
    """Plot performance metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    # Ensure 1D arrays
    mces_true = np.asarray(mces_true).flatten()
    mces_pred = np.asarray(mces_pred).flatten()

    if len(mces_true) == 0 or len(mces_pred) == 0:
        logger.warning("Empty arrays, skipping performance plotting")
        return

    corr_mces, _ = spearmanr(mces_true, mces_pred)
    sns.set_theme(style="ticks")

    model_code = cfg.project.extra_info

    # Hexbin plot
    plot = sns.jointplot(
        x=mces_true,
        y=mces_pred,
        kind="hex",
        color="#4CB391",
        joint_kws={"alpha": 1, "gridsize": 15},
    )
    plot.set_axis_labels("Ground truth Similarity", "Prediction", fontsize=12)
    plot.fig.suptitle(f"Spearman Correlation:{corr_mces:.4f}", fontsize=16)
    plot.ax_joint.set_xlim(0, 40)
    plot.ax_joint.set_ylim(0, 40)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hexbin_plot_{model_code}.png"))
    plt.close()

    # Scatter plot
    plt.figure()
    plt.scatter(mces_true, mces_pred, alpha=0.5)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_plot_{model_code}.png"))
    plt.close()
