## remove the indexes that correspond to the same molecule
import os

os.chdir("/scratch/antwerpen/209/vsc20939/metabolomics")
import dill
from simba.config import Config
import os
from simba.parser import Parser
from simba.molecule_pairs_opt import MoleculePairsOpt


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

output_dataset_path = dataset_path.split(".pkl")[0] + "_no_sim1.pkl"


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

data_keys = [
    "molecule_pairs_train",
    "molecule_pairs_val",
    "molecule_pairs_test",
    "uniformed_molecule_pairs_test",
]
new_dataset = {}
for k in data_keys:

    target_dataset = dataset[k]
    print(f"Number of pairs original: {len(target_dataset)}")
    # new_indexes_tani = target_dataset.indexes_tani[target_dataset.indexes_tani[:,0]!= target_dataset.indexes_tani[:,1]]
    new_indexes_tani = target_dataset.indexes_tani[
        target_dataset.indexes_tani[:, 2] != 1
    ]

    target_dataset.indexes_tani = new_indexes_tani
    new_target_dataset = MoleculePairsOpt(
        spectrums_original=target_dataset.spectrums_original,
        spectrums_unique=target_dataset.spectrums,
        df_smiles=target_dataset.df_smiles,
        indexes_tani_unique=new_indexes_tani,
    )
    new_dataset[k] = new_target_dataset
    print(f"Number of pairs updated: {len(new_target_dataset)}")


# save data
with open(output_dataset_path, "wb") as file:
    dill.dump(new_dataset, file)
