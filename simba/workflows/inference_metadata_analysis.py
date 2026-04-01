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
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.models.similarity_models import SimilarityModelMultitask
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger
from simba.workflows.utils import load_spectra
import spectrum_utils.plot as sup
import matplotlib.pyplot as plt
from simba.utils.plotting_spectrums import *
import matplotlib.pyplot as plt

# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs
sys.modules["simba.molecular_pairs"] = simba.core.data.molecule_pairs
sys.modules["simba.spectrum"] = simba.core.data.spectrum
sys.modules["simba.spectrum_ext"] = simba.core.data.spectrum


from simba.workflows.inference import *


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

    # Prepare dataloaders
    dataloader_ed, dataloader_mces, mols_ed, mols_mces = prepare_inference_dataloaders_return_spectrums(
        cfg, molecule_pairs_ed, molecule_pairs_mces
    )
    

    print(f'Shape of distances: {mols_mces.pair_distances.shape}')
    # Load model
    model = load_model_for_inference(cfg, checkpoint_path)

    # Run inference
    pred_ed, pred_mces = run_inference(cfg, model, dataloader_ed, dataloader_mces)

    # Evaluate
    metrics = evaluate_predictions(
        cfg, pred_ed, pred_mces, dataloader_ed, dataloader_mces, output_dir
    )

    logger.info(f"Results saved to: {output_dir}")

    return metrics, mols_mces





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

    dataset_mces = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
        molecule_pairs_mces_uniform,
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

    return dataloader_ed, dataloader_mces, molecule_pairs_ed_uniform, molecule_pairs_mces_uniform



print('Inference for metadata analysis')


import hydra
from omegaconf import DictConfig, OmegaConf
import pickle

@hydra.main(version_base=None, config_path= "/home/spiedrahita/simba/simba/configs", config_name="config")
def main(cfg: DictConfig):
    #CODE_NAME="train_metadata_all_adducts_seb_20260318_new_metadata_encoding"
    CODE_NAME="metadata_all_adducts_seb_20260203_augmentation_refined_5"
    cfg.preprocessing="tfs_auto" 
    cfg.paths.preprocessing_dir="/data/simba_files/distance_files/ms2_ref_all_metadata/" 
    cfg.paths.preprocessing_dir_train="/data/simba_files/distance_files/ms2_ref_all_metadata/" 
    cfg.paths.checkpoint_dir=f"/data/simba_files/training_files_new_encoding/train_{CODE_NAME}/" 
    cfg.model.features.use_only_protonized_adducts=0 
    cfg.inference.preprocessing_pickle="mapping_unique_smiles.pkl"
    cfg.inference.uniformize_testing=1 
    cfg.inference.enable_progress_bar=0 
    cfg.hardware.accelerator="cpu" 
    
    print(OmegaConf.to_yaml(cfg))
    
    cfg.model.features.use_adduct=1
    cfg.model.features.use_ce=1
    cfg.model.features.use_ion_activation=1
    cfg.model.features.use_ion_method=1
    cfg.model.features.use_ion_mode=1

    metrics_baseline, mols_mces= inference_metadata_analysis(cfg)
    
    error_prediction_baseline = np.abs(metrics_baseline['mces_true']- metrics_baseline['mces_pred'])

    cfg.model.features.use_ion_mode=0
    cfg.model.features.use_adduct=0
    metrics_sensitivity, mols_mces = inference_metadata_analysis(cfg)

            
    error_prediction_sensitivity = np.abs(metrics_sensitivity['mces_true']- metrics_sensitivity['mces_pred'])
    


    ## Make a graph plotting the 2 errors remarking the cases where the first adduct is M+H and the second is M-H
    # find the pairs where the first or second item is m+h and the other is m-h
    ## only take into account the cases when the prediction got worse with deasserted

    all_indexes=[]
    remarked_pair_indexes= []
    for i in range(0,1000):
        unique_index_0 = mols_mces.pair_distances[i, 0]
        unique_index_1 = mols_mces.pair_distances[i, 1]
        spec_0 = mols_mces.get_original_spectrum_from_unique_index(unique_index_0, pair=0)
        spec_1 = mols_mces.get_original_spectrum_from_unique_index(unique_index_1, pair=1)
        adduct_0 = spec_0.params['adduct']
        adduct_1 = spec_1.params['adduct']

        print(f'adducts:{spec_0.params["adduct"]}, {spec_1.params["adduct"]}')
        
        if (('M+H' in adduct_0) and ("M-H" in adduct_1)) or (('M-H' in adduct_0) and ("M+H" in adduct_1)):
            remarked_pair_indexes.append(i)
        all_indexes.append(i)
    #plot scatter plot
    all_er_pred_sen = [error_prediction_sensitivity[i] for i in all_indexes]
    all_er_pred_baseline = [error_prediction_baseline[i] for i in all_indexes]
    filtered_er_pred_sen = [error_prediction_sensitivity[i] for i in remarked_pair_indexes]
    filtered_er_pred_baseline = [error_prediction_baseline[i] for i in remarked_pair_indexes]

    plt.scatter(all_er_pred_baseline, all_er_pred_sen, label='all pairs', alpha=0.3)
    plt.scatter(filtered_er_pred_baseline, filtered_er_pred_sen, label= "filtered", alpha=0.3)
    plt.legend()
    plt.xlabel('Error using all metadata')
    plt.ylabel('Error removing target variable')
    plt.savefig("/data/simba_files/metadata_analysis/adduct_analysis.png")
    ## plot mols
    diff_pred = np.abs(error_prediction_baseline - error_prediction_sensitivity)
    diff_pred[error_prediction_sensitivity < error_prediction_baseline]=0
    
    # remove cases where the ground truth is large
    diff_pred[metrics_sensitivity['mces_true'] > 20]=0
    ordered_indexes =np.argsort(np.abs(diff_pred))

    indexes_high_error= ordered_indexes[-10:]

    for i,highest_diff_index in enumerate(indexes_high_error):

        print("Index with the highest difference in prediction")
        print(highest_diff_index)
    
        print('Error prediction using the baseline')
        print(error_prediction_baseline[highest_diff_index])

        print('Prediction baseline')
        print(metrics_baseline['mces_pred'][highest_diff_index])
        print('Ground truth baseline')
        print(metrics_baseline['mces_true'][highest_diff_index])

        print('Error prediction using the deasserted model')
        print(error_prediction_sensitivity[highest_diff_index])
        print('Prediction deasserted model')
        print(metrics_sensitivity['mces_pred'][highest_diff_index])
        print('Ground truth deasserted model')
        print(metrics_sensitivity['mces_true'][highest_diff_index])
    
        print('Information for the refered pair based on pair distances np array')
        print(mols_mces.pair_distances[highest_diff_index])

        unique_index_0 = mols_mces.pair_distances[highest_diff_index, 0]
        unique_index_1 = mols_mces.pair_distances[highest_diff_index, 1]


        spec_0 = mols_mces.get_original_spectrum_from_unique_index(unique_index_0, pair=0)
        spec_1 = mols_mces.get_original_spectrum_from_unique_index(unique_index_1, pair=1)

        print(f'spec_0: {spec_0}')
        print(f'spec_1: {spec_1}')
    
        pairs_interesting = [{'indexes':[0,0]}]
        out = plot_pair_mols_plus_spectrum_png(
            pair_index=0,
            all_spectrums_query = [spec_0],
            all_spectrums_reference = [spec_1],
            pairs_interesting = pairs_interesting,
            metrics= {'mces_gt': metrics_baseline['mces_true'][highest_diff_index],
                      "mces_pred": metrics_baseline['mces_pred'][highest_diff_index],
                      "ed_gt": 0,
                      "ed_pred": 0,
                      "mod_cos": 0},
            out_dir = "/data/simba_files/metadata_analysis",
            out_name_tpl = f"/data/simba_files/metadata_analysis/example_with_mol_{i}.png",
            )
if __name__ == "__main__":
    main()
