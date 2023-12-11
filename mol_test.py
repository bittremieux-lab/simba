import dill
from src.load_data import LoadData
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle

## PARAMETERS
mgf_path = r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
all_spectrums_path = "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231124.pkl"
dataset_path = "/scratch/antwerpen/209/vsc20939/data/dataset.pkl"
output_file= './dataset_processed_augmented_20231210_bigone.pkl'
#max_number_spectra=70000
#train_molecules=10**7
#val_molecules=10**5
#test_molecules=10**5
#use_tqdm=False

max_number_spectra=70000
train_molecules=3*10**7
val_molecules=10**6
test_molecules=10**6
use_tqdm=False

all_spectrums_original = LoadData.get_all_spectrums(mgf_path,max_number_spectra, use_tqdm=use_tqdm)

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

# get a uniform distribution of taminoto scores
uniformed_molecule_pairs_train =TrainUtils.uniformise(molecule_pairs_train, number_bins=2)
uniformed_molecule_pairs_val =TrainUtils.uniformise(molecule_pairs_val, number_bins=2)
uniformed_molecule_pairs_test =TrainUtils.uniformise(molecule_pairs_test, number_bins=2)

# Dump the dictionary to a file using pickle

dataset_augmented ={'molecule_spairs_train':molecule_pairs_train,
          'molecule_pairs_val':molecule_pairs_val,
          'molecule_pairs_test': molecule_pairs_test,
          'uniformed_molecule_pairs_train':uniformed_molecule_pairs_train,
          'uniformed_molecule_pairs_val':uniformed_molecule_pairs_val,
          'uniformed_molecule_pairs_test': uniformed_molecule_pairs_test,
         }
with open(output_file, 'wb') as file:
    dill.dump(dataset_augmented, file)
