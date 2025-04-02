import dill
from simba.sanity_checks import SanityChecks


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
dataset_path_1 = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_100_million_v2_no_identity.pkl"
dataset_path_2 = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240516_unique_smiles_extra_dummy_low_range.pkl"
dataset_path_out = (
    "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240516_150_millions.pkl"
)

print("Loading data 1 ... ")
with open(dataset_path_1, "rb") as file:
    dataset_1 = dill.load(file)


print("Loading data 2 ... ")
with open(dataset_path_2, "rb") as file:
    dataset_2 = dill.load(file)

# load training data
molecule_pairs_train_1 = dataset_1["molecule_pairs_train"]
molecule_pairs_val_1 = dataset_1["molecule_pairs_val"]
molecule_pairs_test_1 = dataset_1["molecule_pairs_test"]
uniformed_molecule_pairs_test_1 = dataset_1["uniformed_molecule_pairs_test"]

molecule_pairs_train_2 = dataset_2["molecule_pairs_train"]
molecule_pairs_val_2 = dataset_2["molecule_pairs_val"]
molecule_pairs_test_2 = dataset_2["molecule_pairs_test"]
uniformed_molecule_pairs_test_2 = dataset_2["uniformed_molecule_pairs_test"]

print(f"Number of pairs for dataset 1: {len(molecule_pairs_train_1)}")
print(f"Number of pairs for dataset 2: {len(molecule_pairs_train_2)}")
print(f"Number of pairs for uniform test 1: {len(uniformed_molecule_pairs_test_1)}")
print(f"Number of pairs for uniform test 2: {len(uniformed_molecule_pairs_test_2)}")

# merge
molecules_pairs_train = molecule_pairs_train_1 + molecule_pairs_train_2
# molecules_pairs_val = molecule_pairs_val_1 + molecule_pairs_val_2
# molecules_pairs_test = molecule_pairs_test_1 + molecule_pairs_test_2
# uniformed_molecule_pairs_test = (
#    uniformed_molecule_pairs_test_1 + uniformed_molecule_pairs_test_2
# )
molecules_pairs_val = molecule_pairs_val_1
molecules_pairs_test = molecule_pairs_test_1
uniformed_molecule_pairs_test = uniformed_molecule_pairs_test_1

# remove duplicates
molecules_pairs_train = molecules_pairs_train.remove_duplicates()
molecules_pairs_val = molecules_pairs_val.remove_duplicates()
molecules_pairs_test = molecules_pairs_test.remove_duplicates()
uniformed_molecule_pairs_test = uniformed_molecule_pairs_test.remove_duplicates()

# check that the spectrums of train are not in test or val


sanity_check_ids = SanityChecks.sanity_checks_ids(
    molecules_pairs_train,
    molecules_pairs_val,
    molecules_pairs_test,
    uniformed_molecule_pairs_test,
)
sanity_check_bms = SanityChecks.sanity_checks_bms(
    molecules_pairs_train,
    molecules_pairs_val,
    molecules_pairs_test,
    uniformed_molecule_pairs_test,
)

if sanity_check_ids and sanity_check_bms:
    print("Sanity check passed")
    write_data(
        dataset_path_out,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecules_pairs_train,
        molecule_pairs_val=molecules_pairs_val,
        molecule_pairs_test=molecules_pairs_test,
        uniformed_molecule_pairs_test=uniformed_molecule_pairs_test,
    )
    print(f"Number of pairs saved for dataset: {len(molecules_pairs_train)}")
    print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test)}")
else:
    print("There are train ids in val or test :/")
