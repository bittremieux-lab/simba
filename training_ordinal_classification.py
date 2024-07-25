

import os 

# In[268]:


import dill
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from pytorch_lightning.callbacks import ProgressBar
from src.train_utils import TrainUtils
import matplotlib.pyplot as plt
from src.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from src.parser import Parser
import random
from src.weight_sampling import WeightSampling
from src.losscallback import LossCallback
from src.molecular_pairs_set import MolecularPairsSet
from src.sanity_checks import SanityChecks
from src.transformers.postprocessing import Postprocessing
from scipy.stats import spearmanr
import seaborn as sns
from src.ordinal_classification.load_data_ordinal import LoadDataOrdinal
from src.ordinal_classification.embedder_ordinal import EmbedderOrdinal
from sklearn.metrics import confusion_matrix
from src.load_mces.load_mces import LoadMCES
from src.weight_sampling_tools.custom_weighted_random_sampler import CustomWeightedRandomSampler

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
molecule_pairs_train.indexes_tani = LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_train')
molecule_pairs_val.indexes_tani =   LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_val')

## Add the identitiy pairs
USE_IDENTITY_PAIRS=True
if USE_IDENTITY_PAIRS:
    # remove to avoid duplicates
    molecule_pairs_train.indexes_tani = molecule_pairs_train.indexes_tani[\
                                    molecule_pairs_train.indexes_tani[:,0]!= molecule_pairs_train.indexes_tani[:,1]]

    # create identity 
    identity_pairs = np.zeros((len(molecule_pairs_train.spectrums),3))
    identity_pairs[:,0]=np.arange(0,identity_pairs.shape[0])
    identity_pairs[:,1]=np.arange(0,identity_pairs.shape[0])
    identity_pairs[:,2]=1.0
    molecule_pairs_train.indexes_tani  = np.concatenate((molecule_pairs_train.indexes_tani, identity_pairs ))


print(f"Number of pairs for train: {len(molecule_pairs_train)}")
print(f"Number of pairs for val: {len(molecule_pairs_val)}")

## Sanity checks
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



print(f"Sanity check ids. Passed? {sanity_check_ids}")
print(f"Sanity check bms. Passed? {sanity_check_bms}")


## CALCULATION OF WEIGHTS
train_binned_list, ranges = TrainUtils.divide_data_into_bins_categories(
    molecule_pairs_train,
    config.N_CLASSES-1,
        bin_sim_1=True, 
)

# check distribution of similarities
print("SAMPLES PER RANGE:")
for lista in (train_binned_list):
    print(f"samples: {len(lista)}") 

train_binned_list[1].indexes_tani.shape


# In[286]:


plt.hist(molecule_pairs_train.indexes_tani[molecule_pairs_train.indexes_tani[:,2]>0][:,2], bins=20)


# In[287]:


[t.indexes_tani.shape for t in train_binned_list]


# In[288]:


weights, range_weights = WeightSampling.compute_weights_categories(train_binned_list)
#weights, range_weights = WeightSampling.compute_weights(train_binned_list)


# In[289]:


weights


# In[290]:


range_weights


# In[291]:


weights_tr = WeightSampling.compute_sample_weights_categories(molecule_pairs_train, weights)
weights_val = WeightSampling.compute_sample_weights_categories(molecule_pairs_val, weights)


# In[292]:


weights_val[(molecule_pairs_val.indexes_tani[:,2]<0.1)]


# In[293]:


weights_val[(molecule_pairs_val.indexes_tani[:,2]<0.21) & (molecule_pairs_val.indexes_tani[:,2]>0.19)]


# In[294]:


plt.hist(molecule_pairs_val.indexes_tani[:,2], bins=100)
plt.yscale('log')


# In[295]:


plt.hist(weights_val)
plt.yscale('log')


# In[296]:


dataset_train = LoadDataOrdinal.from_molecule_pairs_to_dataset(molecule_pairs_train, training=True)
# dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataset_val = LoadDataOrdinal.from_molecule_pairs_to_dataset(molecule_pairs_val)


# In[297]:

##  Check that the distribution is uniform
dataset_train


# In[298]:


# delete variables that are not useful for memory savings
#del molecule_pairs_val
#del molecule_pairs_test
#del uniformed_molecule_pairs_test


# In[299]:





train_sampler = CustomWeightedRandomSampler(
    weights=weights_tr, num_samples=len(dataset_train), replacement=True
)
val_sampler = CustomWeightedRandomSampler(
    weights=weights_val, num_samples=len(dataset_val), replacement=True
)


# In[300]:


weights_tr


# In[301]:


dataset['molecule_pairs_train'].indexes_tani


# In[302]:


print("Creating train data loader")
dataloader_train = DataLoader(
    dataset_train, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=10
)


# In[303]:


dataloader_train


## check that the distribution of the loader is balanced
similarities_sampled=[]
for i,batch in enumerate(dataloader_train):
    sim = batch['similarity']
    sim = np.array(sim).reshape(-1)
    similarities_sampled = similarities_sampled + list(sim)
    if i==100:
        break

counting, bins, patches =plt.hist(similarities_sampled, bins=6)

print(f'Distribution of similarity for dataset train: {counting}')
print(f'Ranges of similarity for dataset train: {bins}')
# In[304]:


def worker_init_fn(
    worker_id,
):  # ensure the dataloader for validation is the same for every epoch
    seed = 42
    torch.manual_seed(seed)
    # Set the same seed for reproducibility in NumPy and Python's random module
    np.random.seed(seed)
    random.seed(seed)


print("Creating val data loader")
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config.BATCH_SIZE,
    sampler=val_sampler,
    worker_init_fn=worker_init_fn,
    num_workers=10,
)



# Define the ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.CHECKPOINT_DIR,
    filename="best_model",
    monitor="validation_loss_epoch",
    mode="min",
    save_top_k=1,
)

checkpoint_n_steps_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.CHECKPOINT_DIR,
    filename="best_model_n_steps",
    every_n_train_steps=1000,
    save_last=True,
    save_top_k=1,
)


# checkpoint_callback = SaveBestModelCallback(file_path=config.best_model_path)
progress_bar_callback = ProgressBar()

# loss callback
losscallback = LossCallback(file_path=config.CHECKPOINT_DIR + f"loss.png")
print("define model")


# In[308]:


config.USE_GUMBEL


# In[309]:


model = EmbedderOrdinal(
    d_model=int(config.D_MODEL),
    n_layers=int(config.N_LAYERS),
    n_classes=config.N_CLASSES,
    weights=None,
    lr=config.LR,
    use_cosine_distance=config.use_cosine_distance,
    use_gumbel = config.USE_GUMBEL,
)


# In[ ]:


trainer = pl.Trainer(
    #max_steps=100000,
    val_check_interval=10000,
    max_epochs=10,
    callbacks=[checkpoint_callback, checkpoint_n_steps_callback, losscallback],
    enable_progress_bar=enable_progress_bar,
    # val_check_interval= config.validate_after_ratio,
)
# trainer = pl.Trainer(max_steps=100,  callbacks=[checkpoint_callback, losscallback], enable_progress_bar=enable_progress_bar)
trainer.fit(
    model=model,
    train_dataloaders=(dataloader_train),
    val_dataloaders=dataloader_val,
)


# ## Inference

# In[ ]:
