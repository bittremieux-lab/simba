import dill
from simba.load_data import LoadData
from sklearn.model_selection import train_test_split
from simba.train_utils import TrainUtils
from simba.preprocessor import Preprocessor
import pickle
import sys
import numpy as np

## PARAMETERS
mgf_path = r"/scratch/antwerpen/209/vsc20939/data/ALL_GNPS_NO_PROPOGATED_wb.mgf"
all_spectrums_path = (
    "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231124.pkl"
)
dataset_path = "/scratch/antwerpen/209/vsc20939/data/dataset.pkl"
output_file = "./dataset_processed_augmented_20231410_experiments.pkl"
# max_number_spectra=70000
# train_molecules=10**7
# val_molecules=10**5
# test_molecules=10**5
# use_tqdm=False

max_number_spectra = 10000000
train_molecules = 3 * 10**7
val_molecules = 10**6
test_molecules = 10**6
use_tqdm = True

all_spectrums_original = LoadData.get_all_spectrums(
    mgf_path, max_number_spectra, use_tqdm=use_tqdm
)
# sys.exit()
# Dump the dictionary to a file using pickle
# with open(all_spectrums_path, 'rb') as file:
#    all_spectrums_dict = dill.load(file)
# with open(dataset_path, 'rb') as file:
#    dataset = dill.load(file)
# all_spectrums_train= all_spectrums_dict['all_spectrums_train']


# preprocessor
# pp= Preprocessor()

### preprocess
# all_spectrums = pp.preprocess_all_spectrums(all_spectrums_original)


# get raw adducts
# raw_adducts = [s.params['name'].split(' ')[-1] for s in all_spectrums]
raw_adducts = [s.split(" ")[-1] for s in all_spectrums_original]
# clean adducts
clean_adducts = [LoadData._clean_adduct(r) for r in raw_adducts]

# count adducts
adduct_repetitions, number_repetitions = np.unique(clean_adducts, return_counts=True)


# Create a list of tuples with (element, count)
element_count_tuples = list(zip(adduct_repetitions, number_repetitions))

# Sort the list in descending order based on counts
sorted_tuples = sorted(element_count_tuples, key=lambda x: x[1], reverse=True)

# Print the results
for element, count in sorted_tuples:
    print(f"Adduct: {element}, repetitions: {count}")
