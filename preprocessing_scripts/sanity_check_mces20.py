from simba.load_mces.load_mces import LoadMCES
import numpy as np
import dill

data_folder='/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_1/'
np_path = data_folder + 'indexes_tani_incremental_val_1.npy'
mol_file =data_folder + f"edit_distance_neurips_nist_exhaustive.pkl"

array= np.load(np_path)

index=31


pair_index_0=array[index,0]
pair_index_1=array[index,1]


print('indexes:')
print(pair_index_0)
print(pair_index_1)

print('mces loaded')
print(array[index,2])

# load mol file
with open(mol_file, 'rb') as f:
    dataset= dill.load(f)

mol_val= dataset['molecule_pairs_val']

smiles_0 = mol_val.df_smiles.loc[pair_index_0]['canon_smiles']
smiles_1 = mol_val.df_smiles.loc[pair_index_1]['canon_smiles']

print(smiles_0)
print(smiles_1)
