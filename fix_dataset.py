import dill

dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl' 
# Load the dataset from the pickle file
with open(dataset_path, 'rb') as file:
    dataset = dill.load(file)

print(dataset.keys())


# Dump the dictionary to a file using pickle

dataset_augmented ={
          'molecule_pairs_train':dataset['molecule_spairs_train'],
          'molecule_pairs_val':dataset['molecule_pairs_val'],
          'molecule_pairs_test': dataset['molecule_pairs_test'],
         }
with open(dataset_path, 'wb') as file:
    dill.dump(dataset_augmented, file)
