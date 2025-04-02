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
from simba.ordinal_classification.load_data_multitasking import LoadDataMultitasking
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from simba.transformers.embedder import Embedder
from sklearn.metrics import confusion_matrix
from simba.load_mces.load_mces import LoadMCES
from simba.weight_sampling_tools.custom_weighted_random_sampler import (
    CustomWeightedRandomSampler,
)
from simba.plotting import Plotting

# parameters
config = Config()
parser = Parser()
config = parser.update_config(config)
config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1
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


# In[283]:
print("Loading pairs data ...")


## Load UC Riverside data
# Initialize a set to track unique first two columns
def remove_duplicates_array(array):
    seen = set()
    filtered_rows = []

    for row in array:
        # Create a tuple of the first two columns to check uniqueness
        key = tuple(sorted(row[:2]))  # Sort to account for unordered pairs
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    # Convert the filtered rows back to a NumPy array
    result = np.array(filtered_rows)
    return result


print("Loading UC Riverside data")
indexes_tani_multitasking_train_uc = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_VAL_TEST,
    prefix="indexes_tani_incremental_train",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=0,
)


print(f"Size before removing duplicates {indexes_tani_multitasking_train_uc.shape}")
print("Remove duplicates")
indexes_tani_multitasking_train_uc = remove_duplicates_array(
    indexes_tani_multitasking_train_uc
)
print(f"Size after removing duplicates {indexes_tani_multitasking_train_uc.shape}")

## Load new generated data
print("Loading generated data")
indexes_tani_multitasking_train_generated = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_TRAIN,
    prefix="ed_mces_indexes_tani_incremental_train",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=0,
    remove_percentage=0.50,
)

print(
    f"Size before removing duplicates {indexes_tani_multitasking_train_generated.shape}"
)

print("Remove duplicates")
indexes_tani_multitasking_train_generated = remove_duplicates_array(
    indexes_tani_multitasking_train_generated
)
print(
    f"Size after removing duplicates {indexes_tani_multitasking_train_generated.shape}"
)


indexes_tani_multitasking_train_total = np.concatenate(
    (indexes_tani_multitasking_train_uc, indexes_tani_multitasking_train_generated),
    axis=0,
)
indexes_tani_multitasking_train_total = remove_duplicates_array(
    indexes_tani_multitasking_train_total
)
print("LETS COMPARE UCE DATA AND GENERATED")
# assign features
for type_data in ["uc", "generated", "total"]:
    if type_data == "uc":
        indexes_tani_multitasking_train = indexes_tani_multitasking_train_uc
    elif type_data == "generated":
        indexes_tani_multitasking_train = indexes_tani_multitasking_train_generated
    else:
        indexes_tani_multitasking_train = indexes_tani_multitasking_train_total

    molecule_pairs_train.indexes_tani = indexes_tani_multitasking_train[
        :, [0, 1, config.COLUMN_EDIT_DISTANCE]
    ]
    ## CALCULATION OF WEIGHTS
    train_binned_list, ranges = TrainUtils.divide_data_into_bins_categories(
        molecule_pairs_train,
        config.EDIT_DISTANCE_N_CLASSES - 1,
        bin_sim_1=True,
    )
    print(f"#### Data: {type_data}")
    print("SAMPLES PER RANGE:")
    for lista in train_binned_list:
        print(f"samples: {len(lista)}")
