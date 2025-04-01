

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
from sklearn.metrics import confusion_matrix
from simba.load_mces.load_mces import LoadMCES
from simba.weight_sampling_tools.custom_weighted_random_sampler import CustomWeightedRandomSampler

# parameters
config = Config()
parser = Parser()
config = parser.update_config(config)
config.USE_GUMBEL=False
config.N_CLASSES=6
config.bins_uniformise_INFERENCE=config.N_CLASSES-1
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
print('Loading pairs data ...')
indexes_tani_multitasking_train= LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_train', use_edit_distance=config.USE_EDIT_DISTANCE, use_multitask=config.USE_MULTITASK)
indexes_tani_multitasking_val  =   LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_val', use_edit_distance=config.USE_EDIT_DISTANCE, use_multitask=config.USE_MULTITASK)
indexes_tani_multitasking_test  =   LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_test', use_edit_distance=config.USE_EDIT_DISTANCE, use_multitask=config.USE_MULTITASK)

print('Which spectra is in ming data?')
indexes_tani= np.concatenate([indexes_tani_multitasking_train, indexes_tani_multitasking_val, indexes_tani_multitasking_test])


total_indexes = np.concatenate([indexes_tani[:,0],indexes_tani[:,1]])


unique_indexes= np.unique(total_indexes)
print(f'Total indexes {unique_indexes}')
print(f'length: {unique_indexes.shape}')


