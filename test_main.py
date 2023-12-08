import dill
import torch
from torch.utils.data import DataLoader
from src.transformers.load_data import LoadData
import lightning.pytorch as pl
from src.transformers.embedder import Embedder
from pytorch_lightning.callbacks import ProgressBar
from src.transformers.postprocessing import Postprocessing
from sklearn.metrics import r2_score
from src.train_utils import TrainUtils
# parameters
dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl'
epochs= 10
bins_uniformise=5
enable_progress_bar=True

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
uniformed_molecule_pairs_train =TrainUtils.uniformise(dataset['molecule_spairs_train'], number_bins=bins_uniformise)
uniformed_molecule_pairs_val =TrainUtils.uniformise(dataset['molecule_pairs_val'], number_bins=bins_uniformise)
uniformed_molecule_pairs_test =TrainUtils.uniformise(dataset['molecule_pairs_test'], number_bins=bins_uniformise)


print('loading datasets')
dataset_train = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_train)
dataset_test = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_test)
dataset_val = LoadData.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_val)

print('Convert data to a dictionary')
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=15)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

print('define checkpoint')
# Define the ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='model_checkpoints',
    filename='best_model',
    monitor='validation_loss',
    mode='min',
    save_top_k=1,
)

progress_bar_callback = ProgressBar()
print('define model')
# Create a model:
model = Embedder( d_model=64, n_layers=2)

print('train model')
#loss_plot_callback = LossPlotCallback(batch_per_epoch_tr=1, batch_per_epoch_val=2)
trainer = pl.Trainer(max_epochs=epochs,  callbacks=[checkpoint_callback], enable_progress_bar=enable_progress_bar)
trainer.fit(model=model, train_dataloaders=(dataloader_train), val_dataloaders=dataloader_val,)


# Testing
best_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
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
r2= r2_score([c[0] for c in new_combinations_test],[c[1] for c in new_combinations_test])

print(f'The r2 score on test set is {r2}')
