### Check the distribution of edit distance and mces with threshold 20

from simba.load_mces.load_mces import LoadMCES
from simba.mces.mces_computation import MCES
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from simba.config import Config
from simba.mces.mces_computation import MCES

def remove_duplicates_array(array):
    seen = set()
    filtered_rows = []

    for row in array:
        # Create a tuple of the first two columns to check uniqueness
        key = tuple(sorted(row[:2]))  # Sort to account for unordered pairs
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    # Convert the filtered rows back to a NumPy array
    result = np.array(filtered_rows)
    return result
config=Config()

## CODE TO MERGE DATA FROM EDIT DISTANCE CALCULATIONS AND MCES 20

data_folder= '/scratch/antwerpen/209/vsc20939/data/'
INPUT_FOLDER =  data_folder + 'preprocessing_ed_mces_20250123'
SPLITS= ['_val', '_test','_train']
MCES_INDEX=config.COLUMN_MCES20 #column containing mces data
ED_INDEX=config.COLUMN_EDIT_DISTANCE #col containing edit dist. data
BINS_HIST=30
# the edit distance data is saved in files starting with INPUT_SPECIFIC_PAIRS
# the mces 20 data is saved in files  starting with indexes_tani...

data_train=  LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR_TRAIN, 
                            prefix='ed_mces_indexes_tani_incremental_train', 
                            use_edit_distance=config.USE_EDIT_DISTANCE, 
                            use_multitask=config.USE_MULTITASK,
                            add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
                            remove_percentage=0.0,)

print('Loading UC Riverside data')
indexes_tani_multitasking_train_uc  =   LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR_VAL_TEST, 
                            prefix='indexes_tani_incremental_train', 
                            use_edit_distance=config.USE_EDIT_DISTANCE, 
                            use_multitask=config.USE_MULTITASK,
                            add_high_similarity_pairs=0,)

data_train = np.concatenate((data_train, indexes_tani_multitasking_train_uc), axis=0)
data_train= remove_duplicates_array(data_train)


data_val=  LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR_TRAIN, 
                            prefix='ed_mces_indexes_tani_incremental_val', 
                            use_edit_distance=config.USE_EDIT_DISTANCE, 
                            use_multitask=config.USE_MULTITASK,
                            add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
                            remove_percentage=0.0,)

data_val = remove_duplicates_array(data_val)

data_test=  LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR_TRAIN, 
                            prefix='ed_mces_indexes_tani_incremental_test', 
                            use_edit_distance=config.USE_EDIT_DISTANCE, 
                            use_multitask=config.USE_MULTITASK,
                            add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
                            remove_percentage=0.0,)

data_test = remove_duplicates_array(data_test)




print(f'TRAIN Number of pairs: {data_train.shape} ')
print(f'VAL Number of pairs: {data_val.shape} ')

print(f'TEST Number of pairs: {data_test.shape} ')

# plot the Edit distance distribution
'''
plt.figure()
_ = plt.hist(data[:,ED_INDEX], bins=BINS_HIST)
plt.xlabel('edit distance')
plt.ylabel('freq')
plt.grid()
plt.savefig('temp/ed_hist.png')




plt.figure()
_ = plt.hist(mces_normalized, bins=BINS_HIST)
plt.xlabel('mces normalized')
plt.ylabel('freq')
plt.grid()
plt.savefig('temp/mces_hist.png')





mces_raw=MCES.return_mces_raw(mces_normalized, scale=config.NORMALIZATION_MCES_20)

print(f'There are values lower than 0.2? {mces_raw[mces_raw<0.2]}')
plt.figure()
_ = plt.hist(mces_raw, bins=BINS_HIST)
plt.xlabel('mces raw')
plt.ylabel('freq')
plt.grid()
plt.savefig('temp/mces_raw_hist.png')

'''