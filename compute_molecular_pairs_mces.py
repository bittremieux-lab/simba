import dill
#from src.load_data import LoadData
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle
import sys
from src.config import Config
from src.parser import Parser
from datetime import datetime
from src.loader_saver import LoaderSaver
import pickle
import numpy as np
from src.mces.mces_computation import MCES
import random 
import os 



# Get the current date and time
print("Initiating molecular pair script ...")
print(f"Current time: {datetime.now()}")

## PARAMETERS
config = Config()
parser = Parser()
config = parser.update_config(config)
neurips_path = r"/scratch/antwerpen/209/vsc20939/data/MassSpecGym.mgf"
nist_path = r"/scratch/antwerpen/209/vsc20939/data/hr_msms_nist_all.MSP"

if not os.path.exists(config.PREPROCESSING_DIR):
    os.makedirs(config.PREPROCESSING_DIR)

if config.RANDOM_MCES_SAMPLING:
    subfix=''
else:
    subfix='_exhaustive'

output_pairs_file = config.PREPROCESSING_DIR   +     config.PREPROCESSING_PICKLE_FILE
output_np_indexes_train = config.PREPROCESSING_DIR+f"indexes_tani_mces_train{subfix}.npy"
output_np_indexes_val = config.PREPROCESSING_DIR+f"indexes_tani_mces_val{subfix}.npy"
output_np_indexes_test = config.PREPROCESSING_DIR+f"indexes_tani_mces_test{subfix}.npy"
output_nist_file = config.PREPROCESSING_DIR+f"all_spectrums_nist.pkl"
output_neurips_file = config.PREPROCESSING_DIR+f"all_spectrums_neurips.pkl"
output_spectrums_file = config.PREPROCESSING_DIR+f"all_spectrums_neurips_nist_20240814.pkl"

USE_ONLY_LOW_RANGE=True

high_tanimoto_range = 0 if USE_ONLY_LOW_RANGE else 0.5 # to get more high similarity pairs

print(f"output_file:{output_pairs_file}")
# params
max_number_spectra_neurips = 1000000000
max_number_spectra_nist = 10000000000
#train_molecules = 1 * (10**6)
#val_molecules = 10**5
#test_molecules = 10**5
train_molecules =  1*(10**4)
val_molecules = 1*(10**4)
test_molecules = 1*(10**4)

block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_neurips_spectra = True
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
    loaded_molecule_pairs_train= dataset_augmented["molecule_pairs_train"]
    loaded_molecule_pairs_val= dataset_augmented["molecule_pairs_val"]
    loaded_molecule_pairs_test= dataset_augmented["molecule_pairs_test"]
else:
    # load spectrums
    # use neurips

    loader_saver = LoaderSaver(
        block_size=block_size_nist,
        pickle_nist_path=output_nist_file,
        pickle_gnps_path=None,
        pickle_janssen_path=output_neurips_file,
    )

    print(f"Current time: {datetime.now()}")

    # load neurips_spectra
    if load_neurips_spectra:
        with open(output_neurips_file, "rb") as file:
            all_spectrums_neurips = dill.load(file)["spectrums"]
    else:
        all_spectrums_neurips = loader_saver.get_all_spectrums(
            neurips_path,
            max_number_spectra_neurips,
            use_tqdm=use_tqdm,
            use_nist=False,
            use_janssen=True,
            config=config,
        )

    print(f"Total of NEURIPS spectra: {len(all_spectrums_neurips)}")
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
    all_spectrums = all_spectrums_neurips + all_spectrums_nist
    
    # apply filtering by number of peaks
    pp= Preprocessor()
    all_spectrums = pp.return_valid_spectra_n_peaks(all_spectrums, min_peaks=config.MIN_N_PEAKS )

    # get random spectrums
    if config.SUBSAMPLE_PREPROCESSING:
        sample_size=int(len(all_spectrums)/10)
        random_indexes = random.sample(range(len(all_spectrums)), sample_size)
        all_spectrums = [all_spectrums[i] for i in random_indexes]

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

    # not loaded data
    loaded_molecule_pairs_train=None
    loaded_molecule_pairs_val= None
    loaded_molecule_pairs_test= None


print('Training pairs ...')
start_time=datetime.now()
print(f"Current time: {datetime.now()}")
molecule_pairs_train = MCES.compute_all_mces_results_unique(
    all_spectrums_train,
    max_combinations=train_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    num_workers=config.PREPROCESSING_NUM_WORKERS,
    use_exhaustive=True,
    random_sampling=config.RANDOM_MCES_SAMPLING,
    config=config,
    identifier='_train',
    use_edit_distance=config.USE_EDIT_DISTANCE,
    loaded_molecule_pairs= loaded_molecule_pairs_train,
)
end_time=datetime.now()

print(f"Current time: {datetime.now()}")
# Convert timedelta to minutes
# Calculate the difference
time_difference = end_time - start_time
minutes_difference = time_difference.total_seconds() / 60
print(f"Time difference in minutes for training pairs: {minutes_difference:.2f} minutes")




print('Validation pairs ...')
molecule_pairs_val = MCES.compute_all_mces_results_unique(
    all_spectrums_val,
    max_combinations=val_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    num_workers=config.PREPROCESSING_NUM_WORKERS,
    use_exhaustive=True,
    random_sampling=config.RANDOM_MCES_SAMPLING,
    config=config,
    identifier='_val',
    use_edit_distance=config.USE_EDIT_DISTANCE,
    loaded_molecule_pairs= loaded_molecule_pairs_val,
)
print(f"Current time: {datetime.now()}")

print('Test pairs ...')
molecule_pairs_test = MCES.compute_all_mces_results_unique(
    all_spectrums_test,
    max_combinations=test_molecules,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
    high_tanimoto_range=high_tanimoto_range,
    num_workers=config.PREPROCESSING_NUM_WORKERS,
    use_exhaustive=True,
    random_sampling=config.RANDOM_MCES_SAMPLING,
    config=config,
    identifier='_test',
    use_edit_distance=config.USE_EDIT_DISTANCE,
    loaded_molecule_pairs= loaded_molecule_pairs_test,
)





## add molecules with similarity=1
#if (config.RANDOM_MCES_SAMPLING):
#    molecule_pairs_train = TrainUtils.compute_unique_combinations(molecule_pairs_train)
#    molecule_pairs_val = TrainUtils.compute_unique_combinations(molecule_pairs_val)
#    molecule_pairs_test = TrainUtils.compute_unique_combinations(molecule_pairs_test)

# Dump the dictionary to a file using pickle

print(f"Total training data combinations: {len(molecule_pairs_train)}")
print(f"Total val data combinations: {len(molecule_pairs_val)}")
print(f"Total test data combinations: {len(molecule_pairs_test)}")
print(f"Current time: {datetime.now()}")


# save np files
#np.save(arr=molecule_pairs_train.indexes_tani, file= output_np_indexes_train)
#np.save(arr=molecule_pairs_val.indexes_tani, file= output_np_indexes_val)
#np.save(arr=molecule_pairs_test.indexes_tani, file= output_np_indexes_test)

# create uniform test data
#uniformed_molecule_pairs_test, _ = TrainUtils.uniformise(
#    molecule_pairs_test,
#    number_bins=config.bins_uniformise_INFERENCE,
#    return_binned_list=True,
#    bin_sim_1=False,
#)  # do not treat sim==1 as another bin

if write_data_flag:
    write_data(
        output_pairs_file,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs_train,
        molecule_pairs_val=molecule_pairs_val,
        molecule_pairs_test=molecule_pairs_test,
        uniformed_molecule_pairs_test=None,
        
    )


print(f"Current time: {datetime.now()}")

# Calculate the difference
time_difference = end_time - start_time

# Convert timedelta to minutes
minutes_difference = time_difference.total_seconds() / 60

print(f"Time difference in minutes for training pairs: {minutes_difference:.2f} minutes")
