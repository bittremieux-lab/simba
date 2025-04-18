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


# In[283]:
print("Loading pairs data ...")

indexes_tani_multitasking_train = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_TRAIN,
    prefix="ed_mces_indexes_tani_incremental_train",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
    remove_percentage=0.75,
)
print("Loading UC Riverside data")

indexes_tani_multitasking_train_uc = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_VAL_TEST,
    prefix="indexes_tani_incremental_train",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=0,
)

indexes_tani_multitasking_train = np.concatenate(
    (indexes_tani_multitasking_train, indexes_tani_multitasking_train_uc), axis=0
)
indexes_tani_multitasking_train = remove_duplicates_array(
    indexes_tani_multitasking_train
)

indexes_tani_multitasking_val = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_TRAIN,
    prefix="ed_mces_indexes_tani_incremental_val",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
)

indexes_tani_multitasking_val = remove_duplicates_array(indexes_tani_multitasking_val)

# assign features
molecule_pairs_train.indexes_tani = indexes_tani_multitasking_train[
    :, [0, 1, config.COLUMN_EDIT_DISTANCE]
]
molecule_pairs_val.indexes_tani = indexes_tani_multitasking_val[
    :, [0, 1, config.COLUMN_EDIT_DISTANCE]
]

print(f"shape of similarity1: {molecule_pairs_train.indexes_tani.shape}")

# add tanimotos

molecule_pairs_train.tanimotos = indexes_tani_multitasking_train[
    :, config.COLUMN_MCES20
]
molecule_pairs_val.tanimotos = indexes_tani_multitasking_val[:, config.COLUMN_MCES20]


print(f"shape of similarity2: {molecule_pairs_train.tanimotos.shape}")
print(f"Number of pairs for train: {len(molecule_pairs_train)}")
print(f"Number of pairs for val: {len(molecule_pairs_val)}")
print(f"Example of data loaded for tanimotos: {molecule_pairs_train.tanimotos}")

#### RESAMPLING
if config.USE_RESAMPLING:

    high_similarity_indexes = molecule_pairs_train.indexes_tani[:, 2] > 0
    high_similarity_indexes_1 = molecule_pairs_train.indexes_tani[
        high_similarity_indexes
    ]
    high_similarity_indexes_2 = molecule_pairs_train.tanimotos[high_similarity_indexes]
    low_similarity_indexes_1 = molecule_pairs_train.indexes_tani[
        ~high_similarity_indexes
    ]
    low_similarity_indexes_2 = molecule_pairs_train.tanimotos[~high_similarity_indexes]

    # low similarity pairs
    molecule_pairs_train.indexes_tani = low_similarity_indexes_1
    molecule_pairs_train.tanimotos = low_similarity_indexes_2

    model = EmbedderMultitask(
        d_model=int(config.D_MODEL),
        n_layers=int(config.N_LAYERS),
        n_classes=config.EDIT_DISTANCE_N_CLASSES,
        weights=None,
        lr=config.LR,
        use_cosine_distance=config.use_cosine_distance,
        use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
        # weights_sim2=weights_sim2,
        use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
        use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
        use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
    )

    # Create a model:
    if config.load_pretrained:
        try:
            model = EmbedderMultitask.load_from_checkpoint(
                config.pretrained_path,
                d_model=int(config.D_MODEL),
                n_layers=int(config.N_LAYERS),
                n_classes=config.EDIT_DISTANCE_N_CLASSES,
                weights=None,
                lr=config.LR,
                use_cosine_distance=config.use_cosine_distance,
                use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
                weights_sim2=None,
                use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
                use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
                use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
            )
            print("loaded full model!!")
        except:
            model_pretrained = Embedder.load_from_checkpoint(
                config.pretrained_path,
                d_model=int(config.D_MODEL),
                n_layers=int(config.N_LAYERS),
                weights=None,
                lr=config.LR,
                use_cosine_distance=config.use_cosine_distance,
                use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
                weights_sim2=None,
                strict=False,
            )

            model.spectrum_encoder = model_pretrained.spectrum_encoder
            print("Loaded pretrained encoder model")
    else:
        print("Not loaded pretrained model")

    print("Using resampling!")

    dataset_resampling = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_train,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        training=False,
    )
    dataloader_resampling = DataLoader(
        dataset_resampling, batch_size=config.BATCH_SIZE, num_workers=10, shuffle=False
    )

    model.eval()

    # ## Postprocessing

    # In[ ]:
    trainer_inference = pl.Trainer(
        max_epochs=2, enable_progress_bar=enable_progress_bar
    )
    print("Running inference")
    pred_test = trainer_inference.predict(
        model,
        dataloader_resampling,
    )
    print("Finished inference")

    print("extracting array")

    # lets extract the mces distance
    flat_pred_test1 = [p[1] for p in pred_test]
    flat_pred_test1 = [[p.item() for p in p_list] for p_list in flat_pred_test1]
    flat_pred_test1 = [item for sublist in flat_pred_test1 for item in sublist]
    flat_pred_test1 = np.array(flat_pred_test1)

    # flat_pred_test1 = np.argmax(np.array([pred[0] for pred in pred_test]), axis=1).tolist()

    ## filtered indexes, if the prediction is higher tha
    filter_low_similarity_incorrect = flat_pred_test1 > 0.75

    random_indexes = np.random.randint(
        0, low_similarity_indexes_1.shape[0], 2 * sum(filter_low_similarity_incorrect)
    )
    filter_random = np.zeros(low_similarity_indexes_1.shape[0], dtype=bool)
    filter_random[random_indexes] = True

    new_indexes_1 = low_similarity_indexes_1[
        filter_low_similarity_incorrect | filter_random
    ]
    new_indexes_2 = low_similarity_indexes_2[
        filter_low_similarity_incorrect | filter_random
    ]

    print(
        f"The size of the data before resampling: {molecule_pairs_train.indexes_tani.shape}"
    )
    print(f"The size of the data is now: {new_indexes_1.shape}")
    print(f"Example of new array: {new_indexes_1}")

    new_indexes_1 = np.concatenate((new_indexes_1, high_similarity_indexes_1), axis=0)
    new_indexes_2 = np.concatenate((new_indexes_2, high_similarity_indexes_2), axis=0)

    # reassign for training

    # assign features
    molecule_pairs_train.indexes_tani = new_indexes_1
    molecule_pairs_train.tanimotos = new_indexes_2


else:
    print("Not using resampling!")


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
    config.EDIT_DISTANCE_N_CLASSES - 1,
    bin_sim_1=True,
)

import copy

# create a new copy of the pairs, to get the weights of the second similarity
molecule_pairs_train_similarity2 = copy.deepcopy(molecule_pairs_train)
molecule_pairs_train_similarity2.indexes_tani[:, 2] = (
    molecule_pairs_train_similarity2.tanimotos
)

# print(f'Example of tanimotos processed: {molecule_pairs_train_similarity2.indexes_tani[:,2]}')
# train_binned_list2, ranges = TrainUtils.divide_data_into_bins(
#    molecule_pairs_train_similarity2,
#    config.N_CLASSES-1,
#        bin_sim_1=False,
# )

# weights of similarity 2
# weights2, range_weights2 = WeightSampling.compute_weights(train_binned_list2)
# print(f'Train binned list of second similarity: {[len(t) for t in train_binned_list2]}')
# print(f'Weights of second similarity:{weights2}')
# print(f'Ranges of second similarity:{range_weights2}')

# check distribution of similarities
print("SAMPLES PER RANGE:")
for lista in train_binned_list:
    print(f"samples: {len(lista)}")


###### CALCULATE WEIGHTS

plt.hist(
    molecule_pairs_train.indexes_tani[molecule_pairs_train.indexes_tani[:, 2] > 0][
        :, 2
    ],
    bins=20,
)

weights, range_weights = WeightSampling.compute_weights_categories(train_binned_list)

## save info about the weights of similarity 1
Plotting.plot_weights(
    range_weights,
    weights,
    xlabel="weight bin similarity 1",
    filepath=config.CHECKPOINT_DIR + "weights_similarity_1.png",
)


# weights, range_weights = WeightSampling.compute_weights(train_binned_list)


# In[289]:


weights


# In[290]:


range_weights


# In[291]:


weights_tr = WeightSampling.compute_sample_weights_categories(
    molecule_pairs_train, weights
)
weights_val = WeightSampling.compute_sample_weights_categories(
    molecule_pairs_val, weights
)


# In[292]:


weights_val[(molecule_pairs_val.indexes_tani[:, 2] < 0.1)]


# In[293]:


weights_val[
    (molecule_pairs_val.indexes_tani[:, 2] < 0.21)
    & (molecule_pairs_val.indexes_tani[:, 2] > 0.19)
]


# In[294]:


plt.hist(molecule_pairs_val.indexes_tani[:, 2], bins=100)
plt.yscale("log")


# In[295]:


plt.hist(weights_val)
plt.yscale("log")


# In[296]:


dataset_train = LoadDataMultitasking.from_molecule_pairs_to_dataset(
    molecule_pairs_train, max_num_peaks=int(config.TRANSFORMER_CONTEXT), training=True
)
# dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataset_val = LoadDataMultitasking.from_molecule_pairs_to_dataset(
    molecule_pairs_val, max_num_peaks=int(config.TRANSFORMER_CONTEXT)
)


# In[297]:

##  Check that the distribution is uniform
dataset_train


# In[298]:


# delete variables that are not useful for memory savings
# del molecule_pairs_val
# del molecule_pairs_test
# del uniformed_molecule_pairs_test


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


dataset["molecule_pairs_train"].indexes_tani


# In[302]:


print("Creating train data loader")
dataloader_train = DataLoader(
    dataset_train, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=10
)


# In[303]:


dataloader_train


## check that the distribution of the loader is balanced
similarities_sampled = []
similarities_sampled2 = []
for i, batch in enumerate(dataloader_train):
    # sim = batch['similarity']
    # sim = np.array(sim).reshape(-1)
    similarities_sampled = similarities_sampled + list(batch["similarity"].reshape(-1))

    similarities_sampled2 = similarities_sampled2 + list(
        batch["similarity2"].reshape(-1)
    )
    if i == 100:

        # for second similarity remove the sim=1 since it is the same task as the edit distance ==0
        similarities_sampled2 = np.array(similarities_sampled2)
        # similarities_sampled2=similarities_sampled2[similarities_sampled2<1]
        break

## plot similarity distributions
# print(f'similarities 1: {similarities_sampled}')
plt.figure()
plt.xlabel("similarity 1")
plt.ylabel("freq")
plt.hist(similarities_sampled)
plt.savefig(config.CHECKPOINT_DIR + "similarity_distribution_1.png")

# print(f'similarities 2: {similarities_sampled2}')
plt.figure()
plt.hist(similarities_sampled2)
plt.xlabel("similarity 2")
plt.ylabel("freq")
plt.savefig(config.CHECKPOINT_DIR + "similarity_distribution_2.png")


counting, bins, patches = plt.hist(similarities_sampled, bins=6)

print(f"SIMILARITY 1: Distribution of similarity for dataset train: {counting}")
print(f"SIMILARITY 1: Ranges of similarity for dataset train: {bins}")
# In[304]:

# count the number of samples between
counting2, bins2 = TrainUtils.count_ranges(
    np.array(similarities_sampled2), number_bins=5, bin_sim_1=False, max_value=1
)

print(f"SIMILARITY 2: Distribution of similarity for dataset train: {counting2}")
print(f"SIMILARITY 2: Ranges of similarity for dataset train: {bins2}")

weights2 = np.array([np.sum(counting2) / c if c != 0 else 0 for c in counting2])
weights2 = weights2 / np.sum(weights2)

## save info about the weights of similarity 1
bins2_normalized = [
    b if b > 0 else 0 for b in bins2
]  # the first bin has -inf as the lower range
Plotting.plot_weights(
    bins2_normalized,
    weights2,
    xlabel="weight bin similarity 2",
    filepath=config.CHECKPOINT_DIR + "weights_similarity_2.png",
)
print(f"WEIGHTS CALCULATED FOR SECOND SIMILARITY: {weights2}")


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


# In[309]:

## use or not use weights for the second similarity loss
if config.USE_LOSS_WEIGHTS_SECOND_SIMILARITY:
    weights_sim2 = np.array(weights2)
else:
    weights_sim2 = None

model = EmbedderMultitask(
    d_model=int(config.D_MODEL),
    n_layers=int(config.N_LAYERS),
    n_classes=config.EDIT_DISTANCE_N_CLASSES,
    weights=None,
    lr=config.LR,
    use_cosine_distance=config.use_cosine_distance,
    use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
    weights_sim2=weights_sim2,
    use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
    use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
    use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
)

# Create a model:
if config.load_pretrained:
    try:
        model = EmbedderMultitask.load_from_checkpoint(
            config.pretrained_path,
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            weights=None,
            lr=config.LR,
            use_cosine_distance=config.use_cosine_distance,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            weights_sim2=weights_sim2,
            use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
            use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
            use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
        )
        print("loaded full model!!")
    except:
        model_pretrained = Embedder.load_from_checkpoint(
            config.pretrained_path,
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            weights=None,
            lr=config.LR,
            use_cosine_distance=config.use_cosine_distance,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            weights_sim2=weights_sim2,
            strict=False,
        )

        model.spectrum_encoder = model_pretrained.spectrum_encoder
        print("Loaded pretrained encoder model")
else:
    print("Not loaded pretrained model")
# In[ ]:


trainer = pl.Trainer(
    # max_steps=100000,
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
