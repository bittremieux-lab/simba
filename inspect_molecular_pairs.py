import dill


# param
NUMBER_PAIRS=10000

# load data
#dataset_path= '/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231124.pkl'
dataset_path= '/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240112.pkl'

print('Loading data ... ')
with open(dataset_path, 'rb') as file:
    dataset = dill.load(file)

# load training data
molecule_pairs_train = dataset['molecule_pairs_train']
molecule_pairs_val = dataset['molecule_pairs_val']
molecule_pairs_test= dataset['molecule_pairs_test']



for i in range(NUMBER_PAIRS):
    mol =molecule_pairs_train[i]
    if mol.similarity > 0.95:
        print(' ')
        

        
        
        print("*** Pair 0")
        print(mol.params_0)
        print(mol.smiles_0)
        print(mol.global_feats_0)
        print(mol.spectrum_object_0.mz)
        print(mol.spectrum_object_0.intensity)

        print("*** Pair 1")
        print(mol.params_1)
        print(mol.smiles_1)
        print(mol.global_feats_1)
        print(mol.spectrum_object_1.mz)
        print(mol.spectrum_object_1.intensity)

        print(f'Similarity: {mol.similarity}')
        