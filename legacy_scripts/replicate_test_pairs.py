import dill
from simba.load_data import LoadData
from sklearn.model_selection import train_test_split
from simba.train_utils import TrainUtils
from simba.preprocessor import Preprocessor
import pickle
import sys

## PARAMETERS
input_file = (
    "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl"
)
output_file = "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207_only_test.pkl"

test_molecules = 3 * 10**6
use_tqdm = False


print("loading file")
# Load the dataset from the pickle file
with open(input_file, "rb") as file:
    dataset = dill.load(file)

print(dataset.keys())

molecular_pairs_test_input = dataset["molecule_pairs_test"]

all_spectrums_test = [m.spectrum_object_0 for m in molecular_pairs_test_input]
all_spectrums_test = all_spectrums_test + [
    m.spectrum_object_1 for m in molecular_pairs_test_input
]


# all_spectrums_test = dataset['all_spectrums_test']

molecule_pairs_test = TrainUtils.compute_all_tanimoto_results(
    all_spectrums_test[0:10000], max_combinations=test_molecules, use_tqdm=use_tqdm
)


# Dump the dictionary to a file using pickle

dataset_augmented = {
    "molecule_pairs_test": molecule_pairs_test,
}
with open(output_file, "wb") as file:
    dill.dump(dataset_augmented, file)
