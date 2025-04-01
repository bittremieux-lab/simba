import dill
from simba.sanity_checks import SanityChecks
from simba.train_utils import TrainUtils
from simba.config import Config

config=Config()
def write_data(
    file_path,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    uniformed_molecule_pairs_test=None,
):
    dataset = {
        "all_spectrums_train": all_spectrums_train,
        "all_spectrums_val": all_spectrums_val,
        "all_spectrums_test": all_spectrums_test,
        "molecule_pairs_train": molecule_pairs_train,
        "molecule_pairs_val": molecule_pairs_val,
        "molecule_pairs_test": molecule_pairs_test,
        "uniformed_molecule_pairs_test": uniformed_molecule_pairs_test,
    }
    with open(file_path, "wb") as file:
        dill.dump(dataset, file)


# load data
dataset_path_1 = "../data/merged_gnps_nist_20240516_exhaustive.pkl"
dataset_path_out ="../data/merged_gnps_nist_20240516_exhaustive_cleaned.pkl"

print("Loading data 1 ... ")
with open(dataset_path_1, "rb") as file:
    dataset_1 = dill.load(file)



# load training data
molecule_pairs_train = dataset_1["molecule_pairs_train"]
molecule_pairs_val = dataset_1["molecule_pairs_val"]
molecule_pairs_test = dataset_1["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset_1["uniformed_molecule_pairs_test"]

print('examples of similarities loaded')
print(molecule_pairs_train.indexes_tani[1000:1100])

print('loaded')
print(f"Number of pairs  for dataset: {len(molecule_pairs_train)}")
print(f"Number of pairs  for dataset: {len(molecule_pairs_val)}")
print(f"Number of pairs  for dataset: {len(molecule_pairs_test)}")

# check distribution of similarities
samples_per_range, bins = SanityChecks.check_distribution_similarities(
    molecule_pairs_train
)
print("SAMPLES PER RANGE:")
for s, r in zip(samples_per_range, bins):
    print(f"range: {r}, samples: {s}")


## CALCULATION OF WEIGHTS
train_binned_list, _ = TrainUtils.divide_data_into_bins(
    molecule_pairs_train,
    config.bins_uniformise_TRAINING,
)
weights, range_weights = WeightSampling.compute_weights(train_binned_list)

print("Weights per range:")
print(weights)
print(range_weights)


# remove unvalid similarities
print( molecule_pairs_train.indexes_tani [molecule_pairs_train.indexes_tani[:,2]<=1].shape)
molecule_pairs_train.indexes_tani = molecule_pairs_train.indexes_tani [molecule_pairs_train.indexes_tani[:,2]<=1]
molecule_pairs_val.indexes_tani = molecule_pairs_val.indexes_tani [molecule_pairs_val.indexes_tani[:,2]<=1]
molecule_pairs_test.indexes_tani = molecule_pairs_test.indexes_tani [molecule_pairs_test.indexes_tani[:,2]<=1]
uniformed_molecule_pairs_test.indexes_tani = uniformed_molecule_pairs_test.indexes_tani[uniformed_molecule_pairs_test.indexes_tani[:,2]<=1]

write_data(
        dataset_path_out,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs_train,
        molecule_pairs_val=molecule_pairs_val,
        molecule_pairs_test=molecule_pairs_test,
        uniformed_molecule_pairs_test=uniformed_molecule_pairs_test,
    )
print('saved')
print(f"Number of pairs saved for dataset: {len(molecule_pairs_train)}")
print(f"Number of pairs saved for dataset: {len(molecule_pairs_val)}")
print(f"Number of pairs saved for dataset: {len(molecule_pairs_test)}")
print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test)}")

