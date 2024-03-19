import dill
import torch
from torch.utils.data import DataLoader
from src.transformers.load_data_unique import LoadDataUnique
import lightning.pytorch as pl
from src.transformers.embedder import Embedder
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

config = Config()
parser = Parser()
config = parser.update_config(config)

# parameters
dataset_path = config.dataset_path
epochs = config.epochs
use_uniform_data = config.use_uniform_data_TRAINING
bins_uniformise = config.bins_uniformise_TRAINING
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE


if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    # Get the name of the current GPU
    current_gpu = torch.cuda.get_device_name(0)  # assuming you have at least one GPU
    print(f"Current GPU: {current_gpu}")

    # Check if PyTorch is currently using GPU
    current_device = torch.cuda.current_device()
    print(f"PyTorch is using GPU: {torch.cuda.is_initialized()}")

    # Print CUDA version
    print(f"CUDA version: {torch.version.cuda}")

    # Additional information about the GPU
    print(torch.cuda.get_device_properties(current_device))

else:
    print("CUDA (GPU support) is not available.")


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]

##### ADHOC SOLUTION TO AVOID MEMORY OVERHEAD ###########################
# only get 50% of the molecules below 0.5 similarity
# num_half = int(len(molecule_pairs_train)/2)
# molecule_pairs_train = MolecularPairsSet(
#                spectrums= molecule_pairs_train.spectrums,
#                indexes_tani= np.concatenate((molecule_pairs_train.indexes_tani[ molecule_pairs_train.indexes_tani[:,2] > 0.5    ],  \
#                              molecule_pairs_train.indexes_tani[ molecule_pairs_train.indexes_tani[:,2] < 0.5    ][0:num_half]))
# )
# print(f'Adjusted size of molecular pairs: {len(molecule_pairs_train)}')


## TEST: INCREASE THE SIZE OF THE DATASET
# molecule_pairs_train =  molecule_pairs_train + molecule_pairs_train


print(f"Number of pairs for train: {len(molecule_pairs_train)}")
print(f"Number of pairs for val: {len(molecule_pairs_val)}")
print(f"Number of pairs for test: {len(molecule_pairs_test)}")
print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test)}")


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

# check distribution of similarities
samples_per_range, bins = SanityChecks.check_distribution_similarities(
    molecule_pairs_train
)
print("SAMPLES PER RANGE:")
for s, r in zip(samples_per_range, bins):
    print(f"range: {r}, samples: {s}")

print(f"Sanity check ids. Passed? {sanity_check_ids}")
print(f"Sanity check bms. Passed? {sanity_check_bms}")


## CALCULATION OF WEIGHTS
train_binned_list, _ = TrainUtils.divide_data_into_bins(
    molecule_pairs_train,
    config.bins_uniformise_TRAINING,
)
weights, range_weights = WeightSampling.compute_weights(train_binned_list)

print("Weights per range:")
print(weights)
print(range_weights)


weights_tr = WeightSampling.compute_sample_weights(molecule_pairs_train, weights)
weights_val = WeightSampling.compute_sample_weights(molecule_pairs_val, weights)

print("Similarity of the first 20 molecule pairs")
print([molecule_pairs_train[i].similarity for i in range(0, 20)])
print("Sample weights of the first 20 molecule pairs")
print(weights_tr[0:20])

print("loading datasets")
if use_uniform_data:
    print("Uniformize the data")
    uniformed_molecule_pairs_train, train_binned_list = TrainUtils.uniformise(
        molecule_pairs_train, number_bins=bins_uniformise, return_binned_list=True
    )
    uniformed_molecule_pairs_val, _ = TrainUtils.uniformise(
        molecule_pairs_val, number_bins=bins_uniformise, return_binned_list=True
    )
    # uniformed_molecule_pairs_test,_ =TrainUtils.uniformise(molecule_pairs_test, number_bins=bins_uniformise, return_binned_list=True)
    m_train = uniformed_molecule_pairs_train
    # m_test= uniformed_molecule_pairs_test
    m_val = uniformed_molecule_pairs_val
else:
    m_train = molecule_pairs_train
    # m_test= molecule_pairs_test
    m_val = molecule_pairs_val

print(f"number of train molecule pairs: {len(m_train)}")


dataset_train = LoadDataUnique.from_molecule_pairs_to_dataset(m_train, training=True)
# dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataset_val = LoadDataUnique.from_molecule_pairs_to_dataset(m_val)


# delete variables that are not useful for memory savings
del dataset
del molecule_pairs_train
del molecule_pairs_val
del molecule_pairs_test
del uniformed_molecule_pairs_test
del m_train
del m_val

# del(molecule_pairs_test)

print("Generating samplers")
# data loaders


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement,
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


train_sampler = CustomWeightedRandomSampler(
    weights=weights_tr, num_samples=len(dataset_train), replacement=True
)
val_sampler = CustomWeightedRandomSampler(
    weights=weights_val, num_samples=len(dataset_val), replacement=True
)

print("Creating train data loader")
dataloader_train = DataLoader(
    dataset_train, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=10
)
# dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False)


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
    num_workers=0,
)

# Define the ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config.CHECKPOINT_DIR,
    filename="best_model",
    monitor="validation_loss_epoch",
    mode="min",
    save_top_k=1,
)

# checkpoint_callback = SaveBestModelCallback(file_path=config.best_model_path)
progress_bar_callback = ProgressBar()

# loss callback
losscallback = LossCallback(file_path=config.CHECKPOINT_DIR + f"loss.png")

print("define model")


model = Embedder(
    d_model=int(config.D_MODEL),
    n_layers=int(config.N_LAYERS),
    weights=None,
    lr=config.LR,
    use_cosine_distance=config.use_cosine_distance,
)


if config.load_maldi_embedder:
    model.load_pretrained_maldi_embedder(config.maldi_embedder_path)

# Create a model:
if config.load_pretrained:
    model = Embedder.load_from_checkpoint(
        config.pretrained_path,
        d_model=int(config.D_MODEL),
        n_layers=int(config.N_LAYERS),
        weights=None,
        lr=config.LR,
        use_cosine_distance=config.use_cosine_distance,
    )
    print("Loaded pretrained model")
else:
    print("Not loaded pretrained model")

trainer = pl.Trainer(
    max_epochs=epochs,
    callbacks=[checkpoint_callback, losscallback],
    enable_progress_bar=enable_progress_bar,
    # val_check_interval= config.validate_after_ratio,
)
# trainer = pl.Trainer(max_steps=100,  callbacks=[checkpoint_callback, losscallback], enable_progress_bar=enable_progress_bar)
trainer.fit(
    model=model,
    train_dataloaders=(dataloader_train),
    val_dataloaders=dataloader_val,
)

# print loss
# losscallback.plot_loss(file_path = config.CHECKPOINT_DIR +  f'loss_{config.MODEL_CODE}.png')
print(losscallback.train_loss)
print(losscallback.val_loss)

print("finished successfuly.")
