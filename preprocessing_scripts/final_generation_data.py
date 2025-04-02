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
from simba.mces.mces_computation import MCES
import random
import os

import argparse

from simba.simba.preprocessing_simba import PreprocessingSimba


def preprocess_data(array):
    # if any of edit distance or mces is undefined, then we have to put mces and edit distance to a high value (e.g 666)
    ed = array[:, config.COLUMN_EDIT_DISTANCE]
    mces = array[:, config.COLUMN_MCES20]

    indexes_unvalid = np.isnan(ed) | np.isnan(mces)
    # array[indexes_unvalid, config.COLUMN_EDIT_DISTANCE]=config.EDIT_DISTANCE_MAX_VALUE
    # array[indexes_unvalid, config.COLUMN_MCES20]=config.THRESHOLD_MCES

    # make sure that any value of the columns does not exceed the permitted value
    ed_exceeded = array[:, config.COLUMN_EDIT_DISTANCE] == 666
    # array[ed_exceeded, config.COLUMN_EDIT_DISTANCE]= config.EDIT_DISTANCE_MAX_VALUE

    mces_exceeded = array[:, config.COLUMN_MCES20] == 666
    # array[mces_exceeded, config.COLUMN_MCES20]= config.THRESHOLD_MCES

    # remove invalid values
    array = array[(~indexes_unvalid) & (~ed_exceeded) & (~mces_exceeded)]
    return array


parser = argparse.ArgumentParser(description="script.")
parser.add_argument(f"--spectra_path", type=str, default=None)
parser.add_argument(f"--workspace", type=str, default=None)
parser.add_argument(f"--MAX_SPECTRA_TRAIN", type=int, default=100)
parser.add_argument(f"--MAX_SPECTRA_VAL", type=int, default=100)
parser.add_argument(f"--MAX_SPECTRA_TEST", type=int, default=100)
parser.add_argument(f"--mapping_file_name", type=str, default=None)
parser.add_argument(f"--PREPROCESSING_NUM_WORKERS", type=int, default=None)
args = parser.parse_args()

## Parsing arguments
config = Config()
config.PREPROCESSING_PICKLE_FILE = args.mapping_file_name

## PARAMETERS

# spectra_path = r"/Users/sebas/projects/data/processed_massformer/spec_df.pkl"
# spectra_path = r"/Users/sebas/projects/data/MassSpecGym.mgf"
spectra_path = args.spectra_path


config.PREPROCESSING_DIR = args.workspace
config.PREPROCESSING_NUM_WORKERS = args.PREPROCESSING_NUM_WORKERS
MAX_SPECTRA_TRAIN = args.MAX_SPECTRA_TRAIN
MAX_SPECTRA_VAL = args.MAX_SPECTRA_VAL
MAX_SPECTRA_TEST = args.MAX_SPECTRA_TEST

if config.RANDOM_MCES_SAMPLING:
    subfix = ""
else:
    subfix = "_exhaustive"


# In[7]:


output_pairs_file = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
output_np_indexes_train = (
    config.PREPROCESSING_DIR + f"indexes_tani_mces_train{subfix}.npy"
)
output_np_indexes_val = config.PREPROCESSING_DIR + f"indexes_tani_mces_val{subfix}.npy"
output_np_indexes_test = (
    config.PREPROCESSING_DIR + f"indexes_tani_mces_test{subfix}.npy"
)
output_nist_file = config.PREPROCESSING_DIR + f"all_spectrums_nist.pkl"
output_neurips_file = config.PREPROCESSING_DIR + f"all_spectrums_neurips.pkl"
output_spectrums_file = (
    config.PREPROCESSING_DIR + f"all_spectrums_neurips_nist_20240814.pkl"
)

USE_ONLY_LOW_RANGE = True
high_tanimoto_range = (
    0 if USE_ONLY_LOW_RANGE else 0.5
)  # to get more high similarity pairs


print(f"output_file:{output_pairs_file}")
# params
max_number_spectra_neurips = 1000000000
max_number_spectra_nist = 10000000000
train_molecules = 1000
val_molecules = 1000
test_molecules = -1000

block_size_nist = 30000
use_tqdm = config.enable_progress_bar
load_nist_spectra = True
load_neurips_spectra = True
load_train_val_test_data = (
    True  # to load previously train, test, val with proper smiles
)
write_data_flag = True


# In[9]:


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


if __name__ == "__main__":
    # In[10]:

    all_spectrums = PreprocessingSimba.load_spectra(
        spectra_path, config, use_gnps_format=False
    )

    # ## Division, training, validation, test

    # In[11]:

    print("Dividing between training, validation and test")
    all_spectrums_train, all_spectrums_val, all_spectrums_test = (
        TrainUtils.train_val_test_split_bms(all_spectrums)
    )

    # In[12]:

    all_spectrums_train = all_spectrums_train[0:MAX_SPECTRA_TRAIN]
    all_spectrums_val = all_spectrums_val[0:MAX_SPECTRA_VAL]
    all_spectrums_test = all_spectrums_test[0:MAX_SPECTRA_TEST]

    # In[13]:
    molecule_pairs = {}
    print("Validation pairs ...")
    for type_data in ["_train", "_val", "_test"]:

        if type_data == "_train":
            spectra = all_spectrums_train
        elif type_data == "_val":
            spectra = all_spectrums_val
        elif type_data == "_test":
            spectra = all_spectrums_test

        for use_edit_distance in [False, True]:
            molecule_pairs[type_data] = MCES.compute_all_mces_results_unique(
                spectra,
                max_combinations=10000000000000,
                use_tqdm=use_tqdm,
                max_mass_diff=config.MAX_MASS_DIFF,
                min_mass_diff=config.MIN_MASS_DIFF,
                high_tanimoto_range=high_tanimoto_range,
                num_workers=config.PREPROCESSING_NUM_WORKERS,
                use_exhaustive=True,
                random_sampling=config.RANDOM_MCES_SAMPLING,
                config=config,
                identifier=type_data,
                use_edit_distance=use_edit_distance,
                loaded_molecule_pairs=None,
            )

    ## Write the data file
    write_data(
        output_pairs_file,
        all_spectrums_train=None,
        all_spectrums_val=None,
        all_spectrums_test=None,
        molecule_pairs_train=molecule_pairs["_train"],
        molecule_pairs_val=molecule_pairs["_val"],
        molecule_pairs_test=molecule_pairs["_test"],
        uniformed_molecule_pairs_test=None,
    )

    # List all files in the directory
    all_files = os.listdir(config.PREPROCESSING_DIR)

    ed_data_path = config.PREPROCESSING_DIR
    mces_data_path = config.PREPROCESSING_DIR
    output_data_path = config.PREPROCESSING_DIR

    for partition in ["train", "val", "test"]:
        # Filter only the files ending with '.npy'
        npy_files = [f for f in all_files if f.endswith(".npy")]
        npy_files = [f for f in npy_files if partition in f]
        for file_loaded in npy_files:
            # get the index
            index = file_loaded.split("_")[-1].split(".npy")[0]
            index = int(index)

            print(f"Loading index: {index}")

            sufix = "indexes_tani_incremental_" + partition + "_"
            prefix_ed = "edit_distance_"
            prefix_mces = "mces_"
            prefix_output = "ed_mces_"

            file_ed = prefix_ed + sufix + str(index) + ".npy"
            file_mces = prefix_mces + sufix + str(index) + ".npy"
            file_output = prefix_output + sufix + str(index) + ".npy"

            # add directory
            file_ed = ed_data_path + file_ed
            file_mces = mces_data_path + file_mces
            file_output = output_data_path + file_output

            # load data
            ed_data = np.load(file_ed)
            print(f"ed_data: {ed_data}")

            try:
                mces_data = np.load(file_mces)
                print(f"mces_data {mces_data}")
            except:
                print("The MCES partition is not present.")
                continue

            ## check that the indexes are the same:
            if np.all(ed_data[:, 0] == mces_data[:, 0]) and np.all(
                ed_data[:, 1] == mces_data[:, 1]
            ):
                print("The data loaded correspond to the same pairs")

                total_data = np.column_stack((ed_data, mces_data[:, 2]))
                print(f"The data loaded has the original shape: {total_data.shape}")

                print("Preprocessing:")
                total_data = preprocess_data(total_data)

                print(total_data)
                np.save(
                    file_output,
                    total_data,
                )
            else:
                print("ERROR: The data loaded does not correspond to the same pairs")
