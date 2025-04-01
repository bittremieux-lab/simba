## remove the indexes that correspond to the same molecule
import os
os.chdir('/scratch/antwerpen/209/vsc20939/metabolomics')
import dill
from simba.config import Config
import os
from simba.parser import Parser
from simba.molecule_pairs_opt import  MoleculePairsOpt
import numpy as np
#### 
### THIS SCRIPT INTENDS TO GET LOW RANGE TRAINING DATA THAT IS SIMILAR TO HIGH SIMILARITY PAIRS


config = Config()
parser = Parser()
config = parser.update_config(config)

# parameters
dataset_path = config.dataset_path
epochs = config.epochs
use_uniform_data = config.use_uniform_data_TRAINING
bins_uniformise = config.bins_uniformise_TRAINING
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE

output_dataset_path= dataset_path.split('.pkl')[0]+ '_rebalanced.pkl'

## add 10, million pairs
PROPORTION_ADDITIONAL_PAIRS=0.10

print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

data_keys= ["molecule_pairs_train",'molecule_pairs_val','molecule_pairs_test', 'uniformed_molecule_pairs_test' ]
new_dataset={}

# initialize
for k in dataset.keys():
    new_dataset[k]= dataset[k]

for k in ["molecule_pairs_train",'molecule_pairs_val']:
    print(f'Data: {k}')
    target_dataset = dataset[k]

    # get the number of additonal pairs:
    number_additional_pairs = int(PROPORTION_ADDITIONAL_PAIRS*target_dataset.indexes_tani.shape[0])
    ## Check proportion of high similarity pairs > 0.90
    high_train_data = target_dataset.indexes_tani[target_dataset.indexes_tani[:,2]==1]
    low_train_data = target_dataset.indexes_tani[target_dataset.indexes_tani[:,2]<0.5]

    ## Now get the same spectrums but in the low range
    high_range_indexes = np.concatenate((high_train_data[:,0],high_train_data[:,1]), axis=0)
    
    #matched_data_indexes = [True if ((row[0] in high_range_indexes)or(row[1] in high_range_indexes)) else False for row in low_train_data ]
    matched_data_indexes_1= (np.isin(low_train_data[:,0], list(high_range_indexes)),)
    matched_data_indexes_2= (np.isin(low_train_data[:,1], list(high_range_indexes)),)
    matched_data_1 = low_train_data[matched_data_indexes_1]
    matched_data_2 = low_train_data[matched_data_indexes_2]
    matched_data = np.concatenate((matched_data_1,matched_data_2), axis=0)

    print(f'Size of matched data: {matched_data.shape[0]}')

    new_indexes_tani = np.concatenate((target_dataset.indexes_tani, matched_data), axis=0)

    ### New size of new indexes tani
    print(f'Size before:{target_dataset.indexes_tani.shape[0]}')
    print(f'Size of new pairs: {new_indexes_tani.shape[0]}')

    ## new object:
    new_target_dataset=MoleculePairsOpt(
        spectrums_original=target_dataset.spectrums_original, 
        spectrums_unique = target_dataset.spectrums, 
        df_smiles=target_dataset.df_smiles, 
        indexes_tani_unique=new_indexes_tani,
    )

    new_dataset[k]=new_target_dataset


# save data
with open(output_dataset_path, "wb") as file:
        dill.dump(new_dataset, file)