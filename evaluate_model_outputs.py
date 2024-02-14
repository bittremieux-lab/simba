import dill
import torch
from torch.utils.data import DataLoader
from src.transformers.load_data import LoadData
import lightning.pytorch as pl
from src.transformers.embedder import Embedder
from src.transformers.embedder_fingerprint import EmbedderFingerprint
from pytorch_lightning.callbacks import ProgressBar
from src.transformers.postprocessing import Postprocessing
from sklearn.metrics import r2_score
from src.train_utils import TrainUtils
import matplotlib.pyplot as plt
from src.deterministic_similarity import DetSimilarity
from src.plotting import Plotting
from src.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.stats import spearmanr
import argparse
import sys
import os
from src.parser import Parser


# parse arguments
config = Config()
parser = Parser()
best_model_path = config.best_model_path
dataset_path = config.dataset_path
use_uniform_data = config.use_uniform_data_INFERENCE
bins_uniformise = config.bins_uniformise_INFERENCE
output_file_path = config.CHECKPOINT_DIR + "model_outputs_verification.pkl"
enable_progress_bar = config.enable_progress_bar

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


molecule_pairs_test = dataset["molecule_pairs_test"]
print(f"Number of molecule pairs: {len(molecule_pairs_test)}")
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]
print(f"Number of uniform molecule pairs: {len(molecule_pairs_test)}")

print("loading datasets")
if use_uniform_data:
    m_test = uniformed_molecule_pairs_test
else:
    m_test = molecule_pairs_test

# get info from molecular pairs set

spectrum_object_0 = [m.spectrum_object_0 for m in m_test]
precursor_mz_0 = [m.spectrum_object_0.precursor_mz for m in m_test]
precursor_charge_0 = [m.spectrum_object_0.precursor_charge for m in m_test]
intensity_0 = [m.spectrum_object_0.intensity for m in m_test]
mz_0 = [m.spectrum_object_0.mz for m in m_test]
smiles_0 = [m.smiles_0 for m in m_test]
params_0 = [m.params_0 for m in m_test]

spectrum_object_1 = [m.spectrum_object_1 for m in m_test]
precursor_mz_1 = [m.spectrum_object_1.precursor_mz for m in m_test]
precursor_charge_1 = [m.spectrum_object_1.precursor_charge for m in m_test]
intensity_1 = [m.spectrum_object_1.intensity for m in m_test]
mz_1 = [m.spectrum_object_1.mz for m in m_test]
smiles_1 = [m.smiles_1 for m in m_test]
params_1 = [m.params_1 for m in m_test]


# dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False)

# Testinbest_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2, enable_progress_bar=enable_progress_bar)
best_model = Embedder.load_from_checkpoint(
    best_model_path, d_model=int(config.D_MODEL), n_layers=int(config.N_LAYERS)
)

# plot loss:
# best_model.plot_loss()

pred_test = trainer.predict(best_model, dataloader_test)

flat_pred_test = []
for pred in pred_test:
    flat_pred_test = flat_pred_test + [float(p) for p in pred]

similarities_test = Postprocessing.get_similarities(dataloader_test)
combinations_test = [(s, p) for s, p in zip(similarities_test, flat_pred_test)]

# clip the values
x = np.array([c[0] for c in combinations_test])
y = np.array([c[1] for c in combinations_test])
y = np.clip(y, 0, 1)


with open(output_file_path, "wb") as file:
    dataset = {
        "spectrum_object_0": spectrum_object_0,
        "spectrum_object_1": spectrum_object_1,
        "ground_truth_similarity": x,
        "prediction_similarity": y,
        "precursor_mz_0": precursor_mz_0,
        "precursor_charge_0": precursor_charge_0,
        "intensity_0": intensity_0,
        "mz_0": mz_0,
        "smiles_0": smiles_0,
        "params_0": params_0,
        "precursor_mz_1": precursor_mz_1,
        "precursor_charge_1": precursor_charge_1,
        "intensity_1": intensity_1,
        "mz_1": mz_1,
        "smiles_1": smiles_1,
        "params_1": params_1,
    }
    dill.dump(dataset, file)
