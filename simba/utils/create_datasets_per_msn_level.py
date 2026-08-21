<<<<<<< HEAD
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
import pandas as pd
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
from simba.workflows.inference import load_inference_data
import shutil
from hydra import compose, initialize_config_dir
=======
import os
import shutil
import sys

import numpy as np
from omegaconf import DictConfig

import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.utils.logger_setup import logger
from simba.workflows.inference import load_inference_data

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

# Backward compatibility: Support loading old pickle files with old module paths
# These modules were refactored from simba.* to simba.core.* hierarchy
sys.modules["simba.molecule_pairs_opt"] = simba.core.data.molecule_pairs
sys.modules["simba.molecular_pairs"] = simba.core.data.molecule_pairs
sys.modules["simba.spectrum"] = simba.core.data.spectrum
sys.modules["simba.spectrum_ext"] = simba.core.data.spectrum


<<<<<<< HEAD
import hydra
from omegaconf import OmegaConf
from simba.workflows.inference import *
from simba.workflows.inference import _get_ground_truth
from simba.workflows.inference import _which_index
from simba.workflows.inference import _plot_cm, _plot_performance
import pickle
=======
from omegaconf import OmegaConf  # noqa: E402

>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

# -----------------------------------------------------------------------------
# Metadata sensitivity analysis configuration
# -----------------------------------------------------------------------------


<<<<<<< HEAD


def run_create_datasets(cfg):
    molecule_pairs_ed, molecule_pairs_mces, pair_distances=load_inference_data(cfg)
    return molecule_pairs_ed, molecule_pairs_mces, pair_distances
def get_pair_spectra(mols_mces, pair_index: int, ):
    """Return the two original spectra for a pair index."""
    unique_index_0 = mols_mces.pair_distances[pair_index, 0]
    unique_index_1 = mols_mces.pair_distances[pair_index, 1]
    spec_0 = mols_mces.get_original_spectrum_from_unique_index(unique_index_0, pair=0, )
    spec_1 = mols_mces.get_original_spectrum_from_unique_index(unique_index_1, pair=1, )
    return spec_0, spec_1

#@hydra.main(version_base=None, config_path="/home/spiedrahita/simba/simba/configs", config_name="config")
def create_msn_levels(cfg: DictConfig):

    NEW_MS_LEVEL_FOLDER_FORMAT = cfg.paths.checkpoint_dir
    """Hydra entry point for the metadata sensitivity analysis."""
    logger.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    
    molecule_pairs_ed, molecule_pairs_mces, pair_distances= run_create_datasets(cfg)


    
    print('Df smiles: {molecule_pairs_mces.df_smiles}')
    
    matched_indexes = []

    
    
    

    levels_available = [s.params['mslevel'] for s in molecule_pairs_mces.spectra]
    levels_available = np.unique(levels_available)

    for msn_level in levels_available:

        name_folder = str(os.path.basename(cfg.paths.preprocessing_dir.rstrip("/")))
        SAVE_FOLDER='/data/simba_files/' + name_folder+ '_per_level/'
        new_ms_level_folder= SAVE_FOLDER + f'_{msn_level}/'
        valid_rows=[]

        print(f'FIRST PAIR DISTANCES: {pair_distances[0:10]}')
        print(f'LAST PAIR DISTANCES: {pair_distances[-11:-1]}')
        for pair_index in range(pair_distances.shape[0]):
        #for pair_index in range(0,1000):
=======
def run_create_datasets(cfg):
    molecule_pairs_ed, molecule_pairs_mces, pair_distances = load_inference_data(cfg)
    return molecule_pairs_ed, molecule_pairs_mces, pair_distances


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


# @hydra.main(version_base=None, config_path="/home/spiedrahita/simba/simba/configs", config_name="config")
def create_msn_levels(cfg: DictConfig):
    """Hydra entry point for the metadata sensitivity analysis."""
    logger.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    molecule_pairs_ed, molecule_pairs_mces, pair_distances = run_create_datasets(cfg)

    print("Df smiles: {molecule_pairs_mces.df_smiles}")

    levels_available = [s.params["mslevel"] for s in molecule_pairs_mces.spectra]
    levels_available = np.unique(levels_available)

    for msn_level in levels_available:
        name_folder = str(os.path.basename(cfg.paths.preprocessing_dir.rstrip("/")))
        SAVE_FOLDER = "/data/simba_files/" + name_folder + "_per_level/"
        new_ms_level_folder = SAVE_FOLDER + f"_{msn_level}/"
        valid_rows = []

        print(f"FIRST PAIR DISTANCES: {pair_distances[0:10]}")
        print(f"LAST PAIR DISTANCES: {pair_distances[-11:-1]}")
        for pair_index in range(pair_distances.shape[0]):
            # for pair_index in range(0,1000):
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5

            try:
                spec_0, spec_1 = get_pair_spectra(molecule_pairs_mces, pair_index)

<<<<<<< HEAD
                    
                if (spec_0.params['mslevel']==msn_level) and (spec_1.params['mslevel']==msn_level):
                        valid_rows.append(pair_index)
            except:
                print(f'Error processing molecule pair: {pair_index}')
            #print(f'Adducts: {adduct_0},{adduct_1}')
        
        valid_pair_distances= pair_distances[valid_rows]

        if not(os.path.exists(new_ms_level_folder)):
            os.mkdir(new_ms_level_folder)

    

        a = cfg.paths.preprocessing_dir + "mapping_unique_smiles.pkl"
        b = new_ms_level_folder +"mapping_unique_smiles.pkl"

        shutil.copy2(a, b)

        np.save(new_ms_level_folder + 'ed_mces_indexes_tani_incremental_test.npy', valid_pair_distances, )
        ## create a folder with the ms level
        print('Finished successfully')



=======
                if (spec_0.params["mslevel"] == msn_level) and (
                    spec_1.params["mslevel"] == msn_level
                ):
                    valid_rows.append(pair_index)
            except Exception:
                print(f"Error processing molecule pair: {pair_index}")
            # print(f'Adducts: {adduct_0},{adduct_1}')

        valid_pair_distances = pair_distances[valid_rows]

        if not (os.path.exists(new_ms_level_folder)):
            os.mkdir(new_ms_level_folder)

        a = cfg.paths.preprocessing_dir + "mapping_unique_smiles.pkl"
        b = new_ms_level_folder + "mapping_unique_smiles.pkl"

        shutil.copy2(a, b)

        np.save(
            new_ms_level_folder + "ed_mces_indexes_tani_incremental_test.npy",
            valid_pair_distances,
        )
        ## create a folder with the ms level
        print("Finished successfully")
>>>>>>> ed28c05658a6b886f9854d4fd1a4a4395009b1f5
