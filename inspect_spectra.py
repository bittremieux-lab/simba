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
from rdkit import Chem
import numpy as np

# Get the current date and time
print('Initiating molecular pair script ...')
print(f'Current time: {datetime.now()}')

## PARAMETERS
config=Config()
parser = Parser()
config = parser.update_config(config)
gnps_path= r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
nist_path = r'/scratch/antwerpen/209/vsc20939/data/hr_msms_nist_all.MSP'

#pickle files
output_pairs_file= '../data/merged_gnps_nist_20240115_only_nist.pkl'
output_nist_file= '../data/all_spectrums_nist.pkl'
output_gnps_file= '../data/all_spectrums_gnps.pkl'
output_spectrums_file = '../data/all_spectrums_gnps_nist_2024011.pkl'

block_size_nist=30000
use_tqdm=config.enable_progress_bar


# load spectrums
# use gnps


print(f'Current time: {datetime.now()}')
with open(output_gnps_file, 'rb') as file:
        all_spectrums_gnps = dill.load(file)['spectrums']


print(f'Total of GNPS spectra: {len(all_spectrums_gnps)}')
# use nist
print(f'Current time: {datetime.now()}')
with open(output_nist_file, 'rb') as file:
        all_spectrums_nist = dill.load(file)['spectrums']

## Evaluate the number of unique smiles
all_spectrums = all_spectrums_nist + all_spectrums_gnps


## How many unique smiles there are?
smiles = [spec.smiles for spec in all_spectrums]
smiles = [s for s in smiles if s!= '']

# compute canon smiles
for s in smiles:
    try:
        s = Chem.CanonSmiles(s)
    except:
        s = 'NO_CANON'
smiles = [s for s in smiles if s!= 'NO_CANON']


# Get unique values and their counts
unique_values, counts = np.unique(smiles, return_counts=True)

# Get the indices that would sort the counts array in descending order
sorted_indices = np.argsort(counts)[::-1]

# Sort unique_values and counts based on the sorted indices
sorted_unique_values = unique_values[sorted_indices]
sorted_counts = counts[sorted_indices]


print(f'Unique compounds: {len(unique_values)}')
print(f'Mean number of counts per compound: {np.mean(counts)}')
print(f'std  of counts per compound: {np.std(counts)}')

n =30
for s,c in zip(sorted_unique_values[0:n], sorted_counts[0:n]):
    print(f'{s} + {c}')
