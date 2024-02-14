###DOWNSAMPLE THE SIZE OF A TRAIN DATASET
import dill
from src.sanity_checks import SanityChecks
from src.train_utils import TrainUtils

# param
NUMBER_PAIRS = 10000
CHECK_SOME_PAIRS = False
# load data
# dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231124.pkl'
dataset_path = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240207_gnps_nist_janssen_15_millions.pkl"

print("Loading data ... ")
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

# load training data
molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]

print(f"Number of pairs for train: {len(molecule_pairs_train)}")
print(f"Number of pairs for val: {len(molecule_pairs_val)}")
print(f"Number of pairs for test: {len(molecule_pairs_test)}")
print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test)}")


# get random indexes
