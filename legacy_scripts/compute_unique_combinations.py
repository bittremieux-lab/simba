import dill
from simba.load_data import LoadData
from sklearn.model_selection import train_test_split
from simba.train_utils import TrainUtils
from simba.preprocessor import Preprocessor
import pickle
import sys
from simba.config import Config
from simba.parser import Parser
from datetime import datetime
from simba.loader_saver import LoaderSaver
from rdkit import Chem
import numpy as np
from itertools import combinations
import random
from simba.molecular_pairs_set import MolecularPairsSet
from itertools import product
from tqdm import tqdm

# Get the current date and time
print("Initiating molecular pair script ...")
print(f"Current time: {datetime.now()}")


def write_data(
    file_path,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
):
    dataset = {
        "all_spectrums_train": all_spectrums_train,
        "all_spectrums_val": all_spectrums_val,
        "all_spectrums_test": all_spectrums_test,
        "molecule_pairs_train": molecule_pairs_train,
        "molecule_pairs_val": molecule_pairs_val,
        "molecule_pairs_test": molecule_pairs_test,
    }
    with open(file_path, "wb") as file:
        dill.dump(dataset, file)


## PARAMETERS
config = Config()
parser = Parser()
config = parser.update_config(config)
gnps_path = r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
nist_path = r"/scratch/antwerpen/209/vsc20939/data/hr_msms_nist_all.MSP"

# pickle files
output_pairs_file = "../data/merged_gnps_nist_20240115_only_nist.pkl"
output_nist_file = "../data/all_spectrums_nist.pkl"
output_gnps_file = "../data/all_spectrums_gnps.pkl"
output_spectrums_file = "../data/all_spectrums_gnps_nist_2024011.pkl"

block_size_nist = 30000
use_tqdm = config.enable_progress_bar

## molecular pairs files
input_dataset_path = (
    "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240112.pkl"
)
output_dataset_path = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240112_with_identical_pairs.pkl"

# load spectrums
# use gnps

print("loading file")
# Load the dataset from the pickle file
with open(input_dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]


def get_indexes_np(molecule_pairs, pairs_per_compound=40):
    # get spectra
    all_spectrums = molecule_pairs.spectrums

    ## How many unique smiles there are?
    smiles = [spec.smiles for spec in all_spectrums]
    # smiles = [s for s in smiles if s!= '']

    # compute canon smiles
    for s in smiles:
        try:
            s = Chem.CanonSmiles(s)
        except:
            s = "NO_CANON"
    # smiles = [s for s in smiles if s!= 'NO_CANON']

    # Get unique values and their counts
    unique_values, counts = np.unique(smiles, return_counts=True)
    print(f"Unique compounds: {len(unique_values)}")
    print(f"Mean number of counts per compound: {np.mean(counts)}")
    print(f"std  of counts per compound: {np.std(counts)}")

    list_total = []
    for u in unique_values:
        if (u != "NO_CANON") and (u != ""):
            indices = np.where(np.array(smiles) == u)[0]
            index_combinations = list(product(indices, repeat=2))
            random.shuffle(index_combinations)
            list_total = list_total + index_combinations[0:pairs_per_compound]

    lenght_total = len(list_total)

    indexes_np = np.zeros((lenght_total, 3))
    print(f"number of pairs: {lenght_total}")
    for index, l in enumerate(list_total):
        indexes_np[index, 0] = l[0]
        indexes_np[index, 1] = l[1]
        indexes_np[index, 2] = 1

    return indexes_np


indexes_np_train = get_indexes_np(molecule_pairs_train)
indexes_np_val = get_indexes_np(molecule_pairs_val)

# add info to
new_molecule_pairs_train = MolecularPairsSet(
    spectrums=molecule_pairs_train.spectrums,
    indexes_tani=np.concatenate(
        (molecule_pairs_train.indexes_tani, indexes_np_train), axis=0
    ),
)
new_molecule_pairs_val = MolecularPairsSet(
    spectrums=molecule_pairs_val.spectrums,
    indexes_tani=np.concatenate(
        (molecule_pairs_val.indexes_tani, indexes_np_val), axis=0
    ),
)
write_data(
    output_dataset_path,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=new_molecule_pairs_train,
    molecule_pairs_val=new_molecule_pairs_val,
    molecule_pairs_test=molecule_pairs_test,
)

print("file written")
