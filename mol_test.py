import dill
from src.load_data import LoadData
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle
import sys
## PARAMETERS
mgf_path = r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
output_file= './dataset_processed_augmented_20231219.pkl'

max_number_spectra=100000
train_molecules=10**8
val_molecules=10**6
test_molecules=10**6
use_tqdm=True

all_spectrums_original = LoadData.get_all_spectrums(mgf_path,max_number_spectra, use_tqdm=use_tqdm)
#sys.exit()
# Dump the dictionary to a file using pickle
#with open(all_spectrums_path, 'rb') as file:
#    all_spectrums_dict = dill.load(file)
#with open(dataset_path, 'rb') as file:
#    dataset = dill.load(file)
#all_spectrums_train= all_spectrums_dict['all_spectrums_train']


#preprocessor
pp= Preprocessor()

### preprocess 
all_spectrums = pp.preprocess_all_spectrums(all_spectrums_original)

# divide data
all_spectrums_train, all_spectrums_val, all_spectrums_test = TrainUtils.train_val_test_split_bms(all_spectrums)

molecule_pairs_train= TrainUtils.compute_all_tanimoto_results(all_spectrums_train, max_combinations=train_molecules, use_tqdm=use_tqdm)
molecule_pairs_val = TrainUtils.compute_all_tanimoto_results(all_spectrums_val, max_combinations=val_molecules, use_tqdm=use_tqdm)
molecule_pairs_test = TrainUtils.compute_all_tanimoto_results(all_spectrums_test, max_combinations=test_molecules, use_tqdm=use_tqdm)


# Dump the dictionary to a file using pickle

dataset_augmented ={
           'all_spectrums_train':all_spectrums_train,
           'all_spectrums_val':all_spectrums_val,
           'all_spectrums_test':all_spectrums_test,
          'molecule_spairs_train':molecule_pairs_train,
          'molecule_pairs_val':molecule_pairs_val,
          'molecule_pairs_test': molecule_pairs_test,
         }
with open(output_file, 'wb') as file:
    dill.dump(dataset_augmented, file)
