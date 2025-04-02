import os

# In[268]:


import dill
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from pytorch_lightning.callbacks import ProgressBar
from simba.train_utils import TrainUtils
import matplotlib.pyplot as plt
from simba.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from simba.parser import Parser
import random
from simba.weight_sampling import WeightSampling
from simba.losscallback import LossCallback
from simba.molecular_pairs_set import MolecularPairsSet
from simba.sanity_checks import SanityChecks
from simba.transformers.postprocessing import Postprocessing
from scipy.stats import spearmanr
import seaborn as sns
from simba.ordinal_classification.load_data_ordinal import LoadDataOrdinal
from simba.ordinal_classification.embedder_ordinal import EmbedderOrdinal
from sklearn.metrics import confusion_matrix
from simba.load_mces.load_mces import LoadMCES
from simba.weight_sampling_tools.custom_weighted_random_sampler import (
    CustomWeightedRandomSampler,
)
from simba.load_mces.load_mces import LoadMCES

# parameters
config = Config()
parser = Parser()
config = parser.update_config(config)
config.USE_GUMBEL = False
config.N_CLASSES = 6
config.bins_uniformise_INFERENCE = config.N_CLASSES - 1
config.use_uniform_data_INFERENCE = True

# In[281]:
if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)


# parameters
dataset_path = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
epochs = config.epochs
bins_uniformise_inference = config.bins_uniformise_INFERENCE
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]


def merge_computed_data(directory_path, prefix):
    """
    load np arrays containing data as well as apply normalization for training
    """
    # find all np arrays
    files = LoadMCES.find_file(directory_path, prefix)

    # load np files
    print("Loading the partitioned files of the pairs")
    list_arrays = []
    for i, f in enumerate(files):
        print(f"Processing batch {i}")
        np_array = np.load(f)
        # print(f'Size without removal: {np_array.shape[0]}')
        # np_array=LoadMCES.remove_excess_low_pairs(np_array, remove_percentage=remove_percentage)

        print("preview")
        print(np_array[0:10])
        # Replace np.nan in the third column with 666
        np_array[:, 2] = np.where(np.isnan(np_array[:, 2]), 666, np_array[:, 2])

        print(f"Size with removal: {np_array.shape[0]}")
        list_arrays.append(np_array)

    # merge
    print("Merging")
    merged_array = np.concatenate(list_arrays, axis=0)

    # remove excess low pairs
    # merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

    return merged_array


# In[283]:
print("Loading pairs data ...")
indexes_tani = merge_computed_data(
    "/scratch/antwerpen/209/vsc20939/data/preprocessing_edit_distance_compute",
    prefix="indexes_tani_incremental_train",
)
# indexes_tani = merge_computed_data('/scratch/antwerpen/209/vsc20939/data/preprocessing_edit_distance_loaded_full', prefix='indexes_tani_incremental_train')


print(f"Size of the merged data:{indexes_tani.shape[0]}")

# remove duplicates
indexes_tani = np.unique(indexes_tani, axis=0)

print(f"Size of the merged data without duplicates :{indexes_tani.shape[0]}")

# np.save('/scratch/antwerpen/209/vsc20939/data/preprocessing_edit_distance_loaded_full/indexes_tani_incremental_train_merged_computed_extra.npy', indexes_tani)
