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

# parameters
dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl'
best_model_path = '/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints_16_2/best_model.ckpt'
epochs= 1
use_uniform_data=True
bins_uniformise=5
enable_progress_bar=True
fig_path = f'./scatter_plot_{Config.MODEL_CODE}.png'
roc_file_path = f'./roc_curve_{Config.MODEL_CODE}.png'

#fingerprint_model_path = '/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints/model_fingerprints.ckpt'
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

molecule_pairs_train = dataset['molecule_spairs_train']
molecule_pairs_val = dataset['molecule_pairs_val']
molecule_pairs_test= dataset['molecule_pairs_test']

print('Uniformize the data')
uniformed_molecule_pairs_train,train_binned_list =TrainUtils.uniformise(molecule_pairs_train, number_bins=bins_uniformise, return_binned_list=True)
uniformed_molecule_pairs_val,_ =TrainUtils.uniformise(molecule_pairs_val, number_bins=bins_uniformise, return_binned_list=True)
uniformed_molecule_pairs_test,_ =TrainUtils.uniformise(molecule_pairs_test, number_bins=bins_uniformise, return_binned_list=True)

#train_sample_weights = [0.8, 0.2, 0.5, ...]  # Placeholder values, replace with your own
def compute_weights(binned_list):
    weights = np.array([len(r) for r in binned_list])
    weights = 1/weights
    weights = weights/np.sum(weights)
    range_weights = np.arange(0,len(binned_list))/len(binned_list)
    return weights, range_weights

weights, range_weights= compute_weights(train_binned_list)

print('weights per range')
print(weights)
print(range_weights)

def compute_sample_weights(molecule_pairs, weights, range_weights):
    sim = [m.similarity for m in molecule_pairs]
    import math
    index = [math.floor(s*(len(weights))) for s in sim]
    weights_sample = np.array( [weights[ind] if ind < len(weights) else weights[len(weights)-1] for ind in index])
    weights_sample = weights_sample/(sum(weights_sample))
    return weights_sample

weights_tr= compute_sample_weights(molecule_pairs_train, weights, range_weights)
weights_val= compute_sample_weights(molecule_pairs_val, weights, range_weights)

#train_sample_weights= np.ones(len(uniformed_molecule_pairs_train))/len(uniformed_molecule_pairs_train)
# Create a WeightedRandomSampler using the defined probabilities

# create weights
#weights= np.array([len(b) for b in train_binned_list])
#weights = weights/np.sum(weights)

print('loading datasets')
if use_uniform_data:
    m_train = uniformed_molecule_pairs_train
    m_test= uniformed_molecule_pairs_test
    m_val = uniformed_molecule_pairs_val
else:
    m_train = molecule_pairs_train
    m_test= molecule_pairs_test
    m_val = molecule_pairs_val

print(f'number of train molecule pairs: {len(m_train)}')
dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataset_val = LoadData.from_molecule_pairs_to_dataset(m_val)

print('Example of weights')
print(weights_tr[0:20])
print('Similarity of the molecule pairs')
print([m.similarity for m in molecule_pairs_train][0:20])

train_sampler = WeightedRandomSampler(weights=weights_tr, num_samples=len(dataset_train), replacement=True)
val_sampler = WeightedRandomSampler(weights=weights_val, num_samples=len(dataset_val), replacement=True)
print('Convert data to a dictionary')
dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=train_sampler,  num_workers=15)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=32, sampler = val_sampler)

# Define the ModelCheckpoint callback
#checkpoint_callback = pl.callbacks.ModelCheckpoint(
#    dirpath='model_checkpoints',
#    filename='best_model',
#    monitor='validation_loss_epoch',
#    mode='min',
#    save_top_k=1,
#)
#progress_bar_callback = ProgressBar()
#print('define model')
# Create a model:
#model = Embedder( d_model=Config.d_model, n_layers=Config.n_layers, weights=None)
#model_fingerprints=  EmbedderFingerprint.load_from_checkpoint(fingerprint_model_path, d_model=64, n_layers=2)
#for name_a, param_a in model.named_parameters():
#        # Check if the layer exists in model B
#        if name_a in model_fingerprints.state_dict():
#            # Load the weights from model B to model A
#            try:
#             param_a.data.copy_(model_fingerprints.state_dict()[name_a])
#            except:
#             print(f'{param_a} has a problem to be loaded.possible size mismatch')
              
#print('train model')
#loss_plot_callback = LossPlotCallback(batch_per_epoch_tr=1, batch_per_epoch_val=2)
#trainer = pl.Trainer(max_epochs=epochs,  callbacks=[checkpoint_callback], enable_progress_bar=enable_progress_bar)
#trainer.fit(model=model, train_dataloaders=(dataloader_train), val_dataloaders=dataloader_val,)


# Testinbest_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2,)
best_model = Embedder.load_from_checkpoint(best_model_path, d_model=Config.d_model, n_layers=Config.n_layers)

#plot loss:
#best_model.plot_loss()

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




# plot scatter 
plt.xlabel('tanimoto similarity')
plt.ylabel('prediction similarity')
plt.scatter([c[0] for c in new_combinations_test],[c[1] for c in new_combinations_test], label='test', alpha=0.1)
#plt.scatter(similarities_test,cosine_similarity_test, label='test')
plt.legend()
plt.grid()
plt.savefig(fig_path)

# comparison with 
similarities, similarities_tanimoto = DetSimilarity.compute_all_scores(uniformed_molecule_pairs_test, model_file = best_model_path)
Plotting.plot_similarity_graphs(similarities, similarities_tanimoto)
