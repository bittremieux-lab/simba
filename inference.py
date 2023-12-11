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
import numpy as np

# parameters
model_path = '/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints/best_model-v42.ckpt'
dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl'
epochs= 100
bins_uniformise=5
enable_progress_bar=False
fig_path = './scatter_plot.png'
fingerprint_model_path = '/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints/model_fingerprints.ckpt'

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



print('loading file')
# Load the dataset from the pickle file
with open(dataset_path, 'rb') as file:
    dataset = dill.load(file)

print('Uniformize the data')
uniformed_molecule_pairs_train,train_binned_list =TrainUtils.uniformise(dataset['molecule_spairs_train'], number_bins=bins_uniformise, return_binned_list=True)
uniformed_molecule_pairs_val,_ =TrainUtils.uniformise(dataset['molecule_pairs_val'], number_bins=bins_uniformise, return_binned_list=True)
uniformed_molecule_pairs_test,_ =TrainUtils.uniformise(dataset['molecule_pairs_test'], number_bins=bins_uniformise, return_binned_list=True)

# create weights
#weights= np.array([len(b) for b in train_binned_list])
#weights = weights/np.sum(weights)

print('loading datasets')
dataset_train = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_test)
dataset_val = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_val)

print('Convert data to a dictionary')
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=15)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

# Testing
best_model = Embedder.load_from_checkpoint(model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2,)
pred_test = trainer.predict(best_model, dataloader_test)
similarities_test = Postprocessing.get_similarities(dataloader_test)
combinations_test = [(s,float(p[0])) for s,p in zip(similarities_test, pred_test)]

new_combinations_test=[]
bins=10
for i in range(0,bins):
    delta=1/bins
    temp_list = [c for c in combinations_test if ((c[0]>=i*delta)and (c[0]<=(i+1)*(delta)))]
    new_combinations_test = new_combinations_test + temp_list[0:300]

# clip the values
x = np.array([c[0] for c in new_combinations_test])
y = np.array([c[1] for c in new_combinations_test])
y = np.clip(y, 0, 1)
r2= r2_score(x,y)

print(f'The r2 score on test set is {r2}')


# plot scatter 
plt.xlabel('tanimoto similarity')
plt.ylabel('prediction similarity')
plt.scatter([c[0] for c in new_combinations_test],[c[1] for c in new_combinations_test], label='test', alpha=1)
#plt.scatter(similarities_test,cosine_similarity_test, label='test')
plt.legend()
plt.grid()
plt.savefig(fig_path)

# comparison with 
similarities, similarities_tanimoto = DetSimilarity.compute_all_scores(uniformed_molecule_pairs_test, model_file = model_path)
Plotting.plot_similarity_graphs(similarities, similarities_tanimoto)
