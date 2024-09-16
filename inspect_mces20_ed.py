### Check the distribution of edit distance and mces with threshold 20

from src.load_mces.load_mces import LoadMCES
from src.mces.mces_computation import MCES
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.config import Config
from src.mces.mces_computation import MCES
config=Config()

## CODE TO MERGE DATA FROM EDIT DISTANCE CALCULATIONS AND MCES 20

data_folder= '/scratch/antwerpen/209/vsc20939/data/'
INPUT_FOLDER =  data_folder + 'preprocessing_mces20_edit_distance_merged_20240912'
SPLITS= ['_val', '_test','_train']
MCES_INDEX=config.COLUMN_MCES20 #column containing mces data
ED_INDEX=config.COLUMN_EDIT_DISTANCE #col containing edit dist. data
BINS_HIST=30
# the edit distance data is saved in files starting with INPUT_SPECIFIC_PAIRS
# the mces 20 data is saved in files  starting with indexes_tani...

data = LoadMCES.load_raw_data(directory_path=INPUT_FOLDER,
                                prefix="indexes_tani_incremental_val")
mces_normalized= data[:,MCES_INDEX]



print(f'Number of pairs: {data.shape} ')
# plot the Edit distance distribution

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
