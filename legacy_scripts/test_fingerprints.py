import dill
import torch
from torch.utils.data import DataLoader
from simba.transformers.load_data import LoadData
import lightning.pytorch as pl
from simba.transformers.embedder_fingerprint import EmbedderFingerprint
from pytorch_lightning.callbacks import ProgressBar
from simba.transformers.postprocessing import Postprocessing
from sklearn.metrics import r2_score
from simba.train_utils import TrainUtils
import matplotlib.pyplot as plt
from simba.deterministic_similarity import DetSimilarity
from simba.plotting import Plotting
from simba.config import Config

# parameters
dataset_path = "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207_fingerprints.pkl"
epochs = 30
bins_uniformise = 5
enable_progress_bar = True
fig_path = "./scatter_plot.png"

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


# create weights
# weights= np.array([len(b) for b in train_binned_list])
# weights = weights/np.sum(weights)


## add fingerprints
molecule_pairs_train = dataset["molecule_spairs_train"]
molecule_pairs_test = dataset["molecule_pairs_test"]
molecule_pairs_val = dataset["molecule_pairs_val"]

print("computing fingerprints")
# import numpy as np
# for molecule_pairs in [molecule_pairs_train, molecule_pairs_test,molecule_pairs_val]:
# for m in molecule_pairs:
#    m.fingerprint_0 = np.zeros(64)
#    m.fingerprint_1 =np.zeros(64)
"""
import dill
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl'
output_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl'

def generate_fingerprint(smiles, d_size=64):
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=64)
        return np.array(list(fingerprint))
    else:
        return np.zeros((64,))

def gen_fing_for_molecule_pairs(molecule_pairs):

    for m in (molecule_pairs):
        m.fingerprint_0 = generate_fingerprint(m.smiles_0)
        m.fingerprint_1 = generate_fingerprint(m.smiles_1)

    return molecule_pairs

print('generating fingerprints')
molecule_pairs_train = gen_fing_for_molecule_pairs(molecule_pairs_train)
print('finished training data')
molecule_pairs_test = gen_fing_for_molecule_pairs(molecule_pairs_test)
molecule_pairs_val = gen_fing_for_molecule_pairs(molecule_pairs_val)
"""

print("test of number of non zero fingerprint values")
import numpy as np

print(np.sum(molecule_pairs_train[0].fingerprint_0))
print("loading datasets")
dataset_train = LoadData.from_molecule_pairs_to_dataset(molecule_pairs_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(molecule_pairs_test)
dataset_val = LoadData.from_molecule_pairs_to_dataset(molecule_pairs_val)

print("Convert data to a dictionary")
dataloader_train = DataLoader(
    dataset_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=64
)
dataloader_test = DataLoader(dataset_test, batch_size=Config.BATCH_SIZE, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=Config.BATCH_SIZE, shuffle=False)

print("define checkpoint")
# Define the ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="model_checkpoints",
    filename="best_model",
    monitor="validation_loss_epoch",
    mode="min",
    save_top_k=1,
)

progress_bar_callback = ProgressBar()
print("define model")
# Create a model:
model = EmbedderFingerprint(
    d_model=Config.D_MODEL, n_layers=Config.N_LAYERS, weights=None
)

print("train model")
# loss_plot_callback = LossPlotCallback(batch_per_epoch_tr=1, batch_per_epoch_val=2)
trainer = pl.Trainer(
    max_epochs=epochs,
    callbacks=[checkpoint_callback],
    enable_progress_bar=enable_progress_bar,
)
trainer.fit(
    model=model,
    train_dataloaders=(dataloader_train),
    val_dataloaders=dataloader_val,
)
