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

# Get the current date and time
print(f'Current time: {datetime.now()}')

## PARAMETERS
config=Config()
parser = Parser()
config = parser.update_config(config)
gnps_path= r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
nist_path = r'/scratch/antwerpen/209/vsc20939/data/hr_msms_nist_all.MSP'

#pickle files
input_pickle_file = '../data/all_spectrums_gnps_nist_20231222.pkl' #if you want to load directly spectra
output_pairs_file= '../data/merged_gnps_nist_20240105.pkl'
output_spectrums_file = '../data/all_spectrums_gnps_nist_20240105.pkl'

# number of pairs
max_number_spectra_gnps=100000
max_number_spectra_nist=100000

train_molecules=2*10**6
val_molecules=10**5
test_molecules=10**5

use_tqdm=config.enable_progress_bar
load_spectra_pickle=False # to load previously train, test, val with proper smiles 

def write_data(file_path,
                    all_spectrums_train=None, 
                    all_spectrums_val=None, 
                    all_spectrums_test=None, 
                    molecule_pairs_train=None, 
                    molecule_pairs_val=None, 
                    molecule_pairs_test=None):
    dataset ={
           'all_spectrums_train':all_spectrums_train,
           'all_spectrums_val':all_spectrums_val,
           'all_spectrums_test':all_spectrums_test,
          'molecule_pairs_train':molecule_pairs_train,
          'molecule_pairs_val':molecule_pairs_val,
          'molecule_pairs_test': molecule_pairs_test,
         }
    with open(file_path, 'wb') as file:
        dill.dump(dataset, file)


if load_spectra_pickle:

    with open(input_pickle_file, 'rb') as file:
        dataset_augmented = dill.load(file)
        all_spectrums_train= dataset_augmented['all_spectrums_train'] 
        all_spectrums_val= dataset_augmented['all_spectrums_val'] 
        all_spectrums_test= dataset_augmented['all_spectrums_test'] 
else:
    # load spectrums
    # use gnps
    print(f'Current time: {datetime.now()}')
    all_spectrums_gnps = LoadData.get_all_spectrums(gnps_path,max_number_spectra_gnps, use_tqdm=use_tqdm, use_nist=False, config=config)
    print(f'Total of GNPS spectra: {len(all_spectrums_gnps)}')
    # use nist
    print(f'Current time: {datetime.now()}')
    all_spectrums_nist = LoadData.get_all_spectrums(nist_path,max_number_spectra_nist, use_tqdm=use_tqdm, use_nist=True, config=config)
    print(f'Total of NIST spectra: {len(all_spectrums_nist)}')
    print(f'Current time: {datetime.now()}')
    # merge spectrums
    all_spectrums = all_spectrums_gnps + all_spectrums_nist

    #preprocessor
    
    print('Preprocessing spectra')
    pp= Preprocessor()

    ### preprocess 
    all_spectrums = pp.preprocess_all_spectrums(all_spectrums)
    print(f'Current time: {datetime.now()}')
    # divide data
    print('Dividing between training, validation and test')
    all_spectrums_train, all_spectrums_val, all_spectrums_test = TrainUtils.train_val_test_split_bms(all_spectrums)
    print(f'Current time: {datetime.now()}')
    print('Writing data ...')
    # write data
    write_data(output_spectrums_file, 
                    all_spectrums_train=all_spectrums_train, 
                    all_spectrums_val=all_spectrums_val, 
                    all_spectrums_test=all_spectrums_test, 
                    molecule_pairs_train=None, 
                    molecule_pairs_val=None, 
                    molecule_pairs_test=None)

    
print(f'Current time: {datetime.now()}')
molecule_pairs_train= TrainUtils.compute_all_tanimoto_results(all_spectrums_train, max_combinations=train_molecules, use_tqdm=use_tqdm, max_mass_diff=config.MAX_MASS_DIFF)
print(f'Current time: {datetime.now()}')
molecule_pairs_val = TrainUtils.compute_all_tanimoto_results(all_spectrums_val, max_combinations=val_molecules, use_tqdm=use_tqdm, max_mass_diff=config.MAX_MASS_DIFF)
print(f'Current time: {datetime.now()}')
molecule_pairs_test = TrainUtils.compute_all_tanimoto_results(all_spectrums_test, max_combinations=test_molecules, use_tqdm=use_tqdm, max_mass_diff=config.MAX_MASS_DIFF)


# Dump the dictionary to a file using pickle

print(f'Total training data combinations: {len(molecule_pairs_train)}')
print(f'Total val data combinations: {len(molecule_pairs_val)}')
print(f'Total test data combinations: {len(molecule_pairs_test)}')

print(f'Current time: {datetime.now()}')
write_data(output_pairs_file, 
                    all_spectrums_train=None, 
                    all_spectrums_val=None, 
                    all_spectrums_test=None, 
                    molecule_pairs_train=molecule_pairs_train, 
                    molecule_pairs_val=molecule_pairs_val, 
                    molecule_pairs_test=molecule_pairs_test)

print(f'Current time: {datetime.now()}')
