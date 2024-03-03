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
janssen_path = r"/scratch/antwerpen/209/vsc20939/data/drug_plus.mgf"

# pickle files
output_pairs_file = (
    "../data/merged_gnps_nist_20240301_gnps_nist_100_millions.pkl"
)
output_nist_file = "../data/all_spectrums_nist.pkl"
output_gnps_file = "../data/all_spectrums_gnps.pkl"
output_janssen_file = "../data/all_spectrums_janssen.pkl"
output_spectrums_file = "../data/all_spectrums_gnps_nist_20240201_gnps_nist_janssen.pkl"

# params
max_number_spectra_gnps = 1000000000
max_number_spectra_janssen = 1000000000
max_number_spectra_nist = 10000000000
train_molecules = 100 * (10**6)
val_molecules = 10**6
test_molecules = 10**6

block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_gnps_spectra = True
load_janssen_spectra = True
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
        dill.dump(dataset, file)


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
        pickle_janssen_path=output_janssen_file,
    )

    print(f"Current time: {datetime.now()}")

    # load janssen_spectra
    if load_janssen_spectra:
        with open(output_janssen_file, "rb") as file:
            all_spectrums_janssen = dill.load(file)["spectrums"]
    else:
        all_spectrums_janssen = loader_saver.get_all_spectrums(
            janssen_path,
            max_number_spectra_janssen,
            use_tqdm=use_tqdm,
            use_nist=False,
            config=config,
            use_janssen=True,
        )

    print(f"Total of Janssen spectra: {len(all_spectrums_janssen)}")
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
    all_spectrums = all_spectrums_gnps + all_spectrums_nist + all_spectrums_janssen

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
molecule_pairs_train = TrainUtils.compute_all_tanimoto_results(
    all_spectrums_train,
    max_combinations=train_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
)
print(f"Current time: {datetime.now()}")
molecule_pairs_val = TrainUtils.compute_all_tanimoto_results(
    all_spectrums_val,
    max_combinations=val_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
)
print(f"Current time: {datetime.now()}")
molecule_pairs_test = TrainUtils.compute_all_tanimoto_results(
    all_spectrums_test,
    max_combinations=test_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
)

## add molecules with similarity=1
molecule_pairs_train = TrainUtils.compute_unique_combinations(molecule_pairs_train)
molecule_pairs_val = TrainUtils.compute_unique_combinations(molecule_pairs_val)

# Dump the dictionary to a file using pickle

print(f"Total training data combinations: {len(molecule_pairs_train)}")
print(f"Total val data combinations: {len(molecule_pairs_val)}")
print(f"Total test data combinations: {len(molecule_pairs_test)}")
print(f"Current time: {datetime.now()}")

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
