import dill
from simba.sanity_checks import SanityChecks
from simba.train_utils import TrainUtils

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


sanity_check_ids = SanityChecks.sanity_checks_ids(
    molecule_pairs_train,
    molecule_pairs_val,
    molecule_pairs_test,
    uniformed_molecule_pairs_test,
)
sanity_check_bms = SanityChecks.sanity_checks_bms(
    molecule_pairs_train,
    molecule_pairs_val,
    molecule_pairs_test,
    uniformed_molecule_pairs_test,
)

# check distribution of similarities
samples_per_range, bins = SanityChecks.check_distribution_similarities(
    molecule_pairs_train
)
print("SAMPLES PER RANGE:")
for s, r in zip(samples_per_range, bins):
    print(f"range: {r}, samples: {s}")

print(f"Sanity check ids. Passed? {sanity_check_ids}")
print(f"Sanity check bms. Passed? {sanity_check_bms}")
if CHECK_SOME_PAIRS:
    for i in range(NUMBER_PAIRS):
        mol = molecule_pairs_train[i]
        if mol.similarity > 0.95:
            print(" ")

            print("*** Pair 0")
            print(mol.params_0)
            print(mol.smiles_0)
            print(mol.global_feats_0)
            print(mol.spectrum_object_0.mz)
            print(mol.spectrum_object_0.intensity)

            print("*** Pair 1")
            print(mol.params_1)
            print(mol.smiles_1)
            print(mol.global_feats_1)
            print(mol.spectrum_object_1.mz)
            print(mol.spectrum_object_1.intensity)

            print(f"Similarity: {mol.similarity}")
