import dill

# from simba.load_data import LoadData
from sklearn.model_selection import train_test_split
from simba.train_utils import TrainUtils
from simba.preprocessor import Preprocessor
import pickle
import sys
from simba.config import Config
from simba.parser import Parser
from datetime import datetime
from simba.loader_saver import LoaderSaver
import pickle
import numpy as np

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
output_pairs_file = "../data/merged_gnps_nist_20240516_exhaustive.pkl"
output_np_indexes_train = "../data/indexes_tani_train.npy"
output_np_indexes_val = "../data/indexes_tani_val.npy"
output_np_indexes_test = "../data/indexes_tani_test.npy"

output_nist_file = "../data/all_spectrums_nist.pkl"
output_gnps_file = "../data/all_spectrums_gnps.pkl"
output_spectrums_file = "../data/all_spectrums_gnps_nist_20240311.pkl"
USE_ONLY_LOW_RANGE = True
high_tanimoto_range = (
    0 if USE_ONLY_LOW_RANGE else 0.5
)  # to get more high similarity pairs

print(f"output_file:{output_pairs_file}")
# params
max_number_spectra_gnps = 1000000000
max_number_spectra_nist = 10000000000
# train_molecules = 100 * (10**6)
train_molecules = 50 * (10**6)
val_molecules = 10**6
test_molecules = 10**6

block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_gnps_spectra = True
load_train_val_test_data = (
    True  # to load previously train, test, val with proper smiles
)
write_data_flag = True


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
        pickle.dump(dataset, file)


if load_train_val_test_data:

    with open(output_spectrums_file, "rb") as file:
        dataset_augmented = dill.load(file)
    all_spectrums_train = dataset_augmented["all_spectrums_train"]
    all_spectrums_val = dataset_augmented["all_spectrums_val"]
    all_spectrums_test = dataset_augmented["all_spectrums_test"]
else:
    # load spectrums
    # use gnps

    loader_saver = LoaderSaver(
        block_size=block_size_nist,
        pickle_nist_path=output_nist_file,
        pickle_gnps_path=output_gnps_file,
        pickle_janssen_path=None,
    )

    print(f"Current time: {datetime.now()}")

    # load gnps_spectra
    if load_gnps_spectra:
        with open(output_gnps_file, "rb") as file:
            all_spectrums_gnps = dill.load(file)["spectrums"]
    else:
        all_spectrums_gnps = loader_saver.get_all_spectrums(
            gnps_path,
            max_number_spectra_gnps,
            use_tqdm=use_tqdm,
            use_nist=False,
            config=config,
        )

    print(f"Total of GNPS spectra: {len(all_spectrums_gnps)}")
    # use nist
    print(f"Current time: {datetime.now()}")
    if load_nist_spectra:
        with open(output_nist_file, "rb") as file:
            all_spectrums_nist = dill.load(file)["spectrums"]
    else:
        all_spectrums_nist = loader_saver.get_all_spectrums(
            nist_path,
            max_number_spectra_nist,
            use_tqdm=use_tqdm,
            use_nist=True,
            config=config,
        )

    print(f"Total of NIST spectra: {len(all_spectrums_nist)}")
    print(f"Current time: {datetime.now()}")
    # merge spectrums
    all_spectrums = all_spectrums_gnps + all_spectrums_nist

    print(f"Total of all spectra: {len(all_spectrums)}")
    # divide data
    print("Dividing between training, validation and test")
    all_spectrums_train, all_spectrums_val, all_spectrums_test = (
        TrainUtils.train_val_test_split_bms(all_spectrums)
    )
    print(f"Current time: {datetime.now()}")
    print("Writing data ...")
    # write data
    if write_data_flag:
        write_data(
            output_spectrums_file,
            all_spectrums_train=all_spectrums_train,
            all_spectrums_val=all_spectrums_val,
            all_spectrums_test=all_spectrums_test,
            molecule_pairs_train=None,
            molecule_pairs_val=None,
            molecule_pairs_test=None,
        )


print(f"Current time: {datetime.now()}")
molecule_pairs_train = TrainUtils.compute_all_tanimoto_results_unique(
    all_spectrums_train,
    max_combinations=train_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    use_exhaustive=True,
)
print(f"Current time: {datetime.now()}")
molecule_pairs_val = TrainUtils.compute_all_tanimoto_results_unique(
    all_spectrums_val,
    max_combinations=val_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    use_exhaustive=True,
)
print(f"Current time: {datetime.now()}")
molecule_pairs_test = TrainUtils.compute_all_tanimoto_results_unique(
    all_spectrums_test,
    max_combinations=test_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    use_exhaustive=True,
)

## add molecules with similarity=1
# molecule_pairs_train = TrainUtils.compute_unique_combinations(molecule_pairs_train)
# molecule_pairs_val = TrainUtils.compute_unique_combinations(molecule_pairs_val)

# Dump the dictionary to a file using pickle

print(f"Total training data combinations: {len(molecule_pairs_train)}")
print(f"Total val data combinations: {len(molecule_pairs_val)}")
print(f"Total test data combinations: {len(molecule_pairs_test)}")
print(f"Current time: {datetime.now()}")


# save np files
np.save(arr=molecule_pairs_train.indexes_tani, file=output_np_indexes_train)
np.save(arr=molecule_pairs_val.indexes_tani, file=output_np_indexes_val)
np.save(arr=molecule_pairs_test.indexes_tani, file=output_np_indexes_test)

# create uniform test data
uniformed_molecule_pairs_test, _ = TrainUtils.uniformise(
    molecule_pairs_test,
    number_bins=config.bins_uniformise_INFERENCE,
    return_binned_list=True,
    bin_sim_1=False,
)  # do not treat sim==1 as another bin
if write_data_flag:
    write_data(
        output_pairs_file,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs_train,
        molecule_pairs_val=molecule_pairs_val,
        molecule_pairs_test=molecule_pairs_test,
        uniformed_molecule_pairs_test=uniformed_molecule_pairs_test,
    )

print(f"Current time: {datetime.now()}")
