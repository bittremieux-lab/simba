import dill
from src.sanity_checks import SanityChecks
from src.molecular_pairs_set import MolecularPairsSet


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
dataset_path_1 = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240227_gnps_nist_janssen_20_millions_OUTSIDE_MAX_DIFF.pkl"
dataset_path_out = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240227_gnps_nist_janssen_20_millions_OUTSIDE_MAX_DIFF_NO_JANSSEN.pkl"

print("Loading data 1 ... ")
with open(dataset_path_1, "rb") as file:
    dataset_1 = dill.load(file)


# load training data
molecule_pairs_train_1 = dataset_1["molecule_pairs_train"]
molecule_pairs_val_1 = dataset_1["molecule_pairs_val"]
molecule_pairs_test_1 = dataset_1["molecule_pairs_test"]
uniformed_molecule_pairs_test_1 = dataset_1["uniformed_molecule_pairs_test"]


print(
    f"Number of pairs for train dataset before Janssen removal: {len(molecule_pairs_train_1)}"
)
print(
    f"Number of pairs for val dataset before Janssen removal: {len(molecule_pairs_val_1)}"
)
print(
    f"Number of pairs for test dataset before Janssen removal: {len(molecule_pairs_test_1)}"
)
print(
    f"Number of pairs for unif test dataset before Janssen removal: {len(uniformed_molecule_pairs_test_1)}"
)


molecule_pairs_train_1 = molecule_pairs_train_1.remove_library_pairs(library="janssen")
molecule_pairs_val_1 = molecule_pairs_val_1.remove_library_pairs(library="janssen")

# sanity check removal

libraries_train = [
    (
        molecule_pairs_train_1.spectrums[int(row[0])].library
        and molecule_pairs_train_1.spectrums[int(row[1])].library
    )
    for row in molecule_pairs_train_1.indexes_tani
]
libraries_val = [
    (
        molecule_pairs_val_1.spectrums[int(row[0])].library
        and molecule_pairs_val_1.spectrums[int(row[1])].library
    )
    for row in molecule_pairs_val_1.indexes_tani
]
sanity_check_janssen = not ("janssen" in libraries_train + libraries_val)

print("")
print(
    f"Number of pairs for train dataset after Janssen removal: {len(molecule_pairs_train_1)}"
)
print(
    f"Number of pairs for val dataset after Janssen removal: {len(molecule_pairs_val_1)}"
)
print(
    f"Number of pairs for test dataset after Janssen removal: {len(molecule_pairs_test_1)}"
)
print(
    f"Number of pairs for unif test dataset after Janssen removal: {len(uniformed_molecule_pairs_test_1)}"
)
# check that the spectrums of train are not in test or val


sanity_check_ids = SanityChecks.sanity_checks_ids(
    molecule_pairs_train_1,
    molecule_pairs_val_1,
    molecule_pairs_test_1,
    uniformed_molecule_pairs_test_1,
)
sanity_check_bms = SanityChecks.sanity_checks_bms(
    molecule_pairs_train_1,
    molecule_pairs_val_1,
    molecule_pairs_test_1,
    uniformed_molecule_pairs_test_1,
)

if sanity_check_ids and sanity_check_bms and sanity_check_janssen:
    print("Sanity check passed")
    write_data(
        dataset_path_out,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs_train_1,
        molecule_pairs_val=molecule_pairs_val_1,
        molecule_pairs_test=molecule_pairs_test_1,
        uniformed_molecule_pairs_test=uniformed_molecule_pairs_test_1,
    )
    print(f"Number of pairs saved for dataset: {len(molecule_pairs_train_1)}")
    print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test_1)}")
else:
    print("There are problems with the data leakage :/")
