import dill
from src.load_data import LoadData
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle
import sys
from src.config import Config
from src.parser import Parser
from datetime import datetime
from src.loader_saver import LoaderSaver

# Get the current date and time
print("Initiating molecular pair script ...")
print(f"Current time: {datetime.now()}")

## PARAMETERS
config = Config()
parser = Parser()
config = parser.update_config(config)
gnps_path = r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
nist_path = r"/scratch/antwerpen/209/vsc20939/data/hr_msms_nist_all.MSP"

# pickle files
output_pairs_file = "../data/merged_gnps_nist_20240118_additional_pairs.pkl"
output_nist_file = "../data/all_spectrums_nist.pkl"
output_gnps_file = "../data/all_spectrums_gnps.pkl"
output_spectrums_file = "../data/all_spectrums_gnps_nist_2024011.pkl"

# number of pairs
# max_number_spectra_gnps=70000
# max_number_spectra_nist=300000
# train_molecules=2*10**6
# val_molecules=10**5
# test_molecules=10**5

max_number_spectra_gnps = 10000000
max_number_spectra_nist = 10000000
train_molecules = 10**5


block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_gnps_spectra = True
load_train_val_test_data = (
    False  # to load previously train, test, val with proper smiles
)
write_data_flag = True
input_dataset_path = config.dataset_path
output_dataset_path = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240112_additional_pairs.pkl"


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


print("loading file")
# Load the dataset from the pickle file
with open(input_dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]

all_spectrums_train = molecule_pairs_train.spectrums


print(f"Current time: {datetime.now()}")
molecule_pairs_train = TrainUtils.compute_all_tanimoto_results_sequential(
    all_spectrums_train,
    max_combinations=train_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
)

# Dump the dictionary to a file using pickle

print(f"Total training data combinations: {len(molecule_pairs_train)}")


print(f"Current time: {datetime.now()}")
if write_data_flag:
    write_data(
        output_pairs_file,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs_train,
        molecule_pairs_val=molecule_pairs_val,
        molecule_pairs_test=molecule_pairs_test,
    )

print(f"Current time: {datetime.now()}")
