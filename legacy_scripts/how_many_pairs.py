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
output_pairs_file = "../data/merged_gnps_nist_20240115_only_nist.pkl"
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
train_molecules = 10 * 10**6
val_molecules = 10**6
test_molecules = 10**6

block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_gnps_spectra = True
load_train_val_test_data = (
    False  # to load previously train, test, val with proper smiles
)
write_data_flag = False


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
    )

    print(f"Current time: {datetime.now()}")

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


all_spectrums = all_spectrums[0:40000]

print(f"Current time: {datetime.now()}")
number_of_pairs = TrainUtils.compute_number_of_pairs(
    all_spectrums,
    max_combinations=train_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
)
print(f"Current time: {datetime.now()}")
print(f"Number of pairs:{number_of_pairs}")
