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
config=Config()
parser =Parser()
config = parser.update_config(config)
dataset_path= config.dataset_path
best_model_path = config.best_model_path
use_uniform_data=config.use_uniform_data_INFERENCE
bins_uniformise=config.bins_uniformise_INFERENCE
fig_path =  config.CHECKPOINT_DIR + f'scatter_plot_{config.MODEL_CODE}.png'
roc_file_path = config.CHECKPOINT_DIR + f'roc_curve_{config.MODEL_CODE}.png'
enable_progress_bar=config.enable_progress_bar

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



print('loading file')
# Load the dataset from the pickle file
with open(dataset_path, 'rb') as file:
    dataset = dill.load(file)

molecule_pairs_test= dataset['molecule_pairs_test']
print(f'Number of molecule pairs: {len(molecule_pairs_test)}')
print('Uniformize the data')
uniformed_molecule_pairs_test,_ =TrainUtils.uniformise(molecule_pairs_test, number_bins=bins_uniformise, 
                                                       return_binned_list=True,
                                                        bin_sim_1=False) # do not treat sim==1 as another bin



print('loading datasets')
if use_uniform_data:
    m_test= uniformed_molecule_pairs_test
else:
    m_test= molecule_pairs_test

#dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False)

# Testinbest_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2,enable_progress_bar=enable_progress_bar)
best_model = Embedder.load_from_checkpoint(best_model_path, d_model=int(config.D_MODEL), n_layers=int(config.N_LAYERS))

#plot loss:
#best_model.plot_loss()

pred_test = trainer.predict(best_model, dataloader_test,  )
similarities_test = Postprocessing.get_similarities(dataloader_test)

# flat the results
flat_pred_test=[]
for pred in pred_test:
    flat_pred_test = flat_pred_test + [float(p) for p in pred]
combinations_test = [(s,p) for s,p in zip(similarities_test, flat_pred_test)]

new_combinations_test=[]
#bins=10
#for i in range(0,bins):
#    delta=1/bins
#    temp_list = [c for c in combinations_test if ((c[0]>=i*delta)and (c[0]<=(i+1)*(delta)))]
#    new_combinations_test = new_combinations_test + temp_list[0:-1]

# clip the values
x = np.array([c[0] for c in combinations_test])
y = np.array([c[1] for c in combinations_test])
y = np.clip(y, 0, 1)




# plot scatter 
plt.xlabel('tanimoto similarity')
plt.ylabel('prediction similarity')
plt.scatter(x,y, label='test', alpha=0.5)
#plt.scatter(similarities_test,cosine_similarity_test, label='test')
plt.legend()
plt.grid()
plt.savefig(fig_path)

# hexbin plot
import numpy as np
import seaborn as sns
print(f'Number of test samples: {len(y)}')
sns.set_theme(style="ticks")
plot = sns.jointplot(x=x, y=y, kind="hex", color="#4CB391", joint_kws = dict(alpha=1))
# Set x and y labels
plot.set_axis_labels("Tanimoto similarity", "Model prediction", fontsize=12)
plt.savefig(config.CHECKPOINT_DIR + f'hexbin_plot_{config.MODEL_CODE}.png')


# comparison with 
similarities, similarities_tanimoto = DetSimilarity.compute_all_scores(m_test, model_file = best_model_path, config=config)
Plotting.plot_similarity_graphs(similarities, similarities_tanimoto, config=config)
