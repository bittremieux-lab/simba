import pickle
import sys
import simba
import numpy as np
import pandas as pd
from simba.config import Config
from simba.load_mces.load_mces import LoadMCES
from tqdm import tqdm

class FilteringMassSpecGym:
    # Initialize a set to track unique first two columns
    def remove_duplicates_array(array):
        seen = set()
        filtered_rows = []

        for row in array:
            # Create a tuple of the first two columns to check uniqueness
            key = tuple(sorted(row[:2]))  # Sort to account for unordered pairs
            if key not in seen:
                seen.add(key)
                filtered_rows.append(row)

        # Convert the filtered rows back to a NumPy array
        result = np.array(filtered_rows)
        return result

    def filter_massspecgym(molecule_pairs_test, path_pairs, config, prefix):

        indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(
            path_pairs,
            prefix=prefix,
            use_edit_distance=config.USE_EDIT_DISTANCE,
            use_multitask=config.USE_MULTITASK,
            add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
        )

        indexes_tani_multitasking_test = FilteringMassSpecGym.remove_duplicates_array(indexes_tani_multitasking_test)

        # let's mark the spectrums as nist or no nist
        is_massspecgym =   [ 1       if 'organism' not in s.params else 0 for s in molecule_pairs_test.spectrums_original ]
        index_massspecgym= [ index    for index, s in enumerate(molecule_pairs_test.spectrums_original ) if 'organism' not in s.params]


        spectrums_massspecgym = [molecule_pairs_test.spectrums_original[ind] for ind in index_massspecgym]


        df_translation_indexes= pd.DataFrame() 
        df_translation_indexes['new_index']=np.arange(0, len(index_massspecgym))
        df_translation_indexes['old_index']=index_massspecgym 

        good_indexes=[]
        for index in tqdm(range(0,indexes_tani_multitasking_test.shape[0])):
            pair=indexes_tani_multitasking_test[index]
            spectrums_0= [molecule_pairs_test.spectrums_original[index] for index in molecule_pairs_test.df_smiles.loc[pair[0],'indexes']]
            spectrums_1= [molecule_pairs_test.spectrums_original[index] for index in molecule_pairs_test.df_smiles.loc[pair[1],'indexes']]
            spectrums_0_massspecgym = [s for s in spectrums_0 if 'organism' not in s.params]
            spectrums_1_massspecgym = [s for s in spectrums_1 if 'organism' not in s.params]
            if (len(spectrums_0_massspecgym)>0) & (len(spectrums_1_massspecgym)>0):
                good_indexes.append(index)

        indexes_tani_multitasking_test_filtered =indexes_tani_multitasking_test[good_indexes]

        new_df_smiles = molecule_pairs_test.df_smiles.copy()

        def filter_massspecgym_indexes(indexes, df_translation_indexes):
            new_indexes=[]
            indexes = np.reshape(indexes, -1)
            for index in indexes:
                    rows_filtered =df_translation_indexes[df_translation_indexes['old_index']==index]
                    if rows_filtered.shape[0]>0:
                        new_indexes.append(rows_filtered['new_index'].values[0])
            return new_indexes

        new_df_smiles['indexes'] = new_df_smiles['indexes'].apply(lambda x:filter_massspecgym_indexes(x, df_translation_indexes))

        delete_rows = new_df_smiles.apply(lambda x:len(x['indexes'])==0, axis=1)
        new_df_smiles['number_indexes']= new_df_smiles['indexes'].apply(lambda x: len(x))

        new_df_smiles['library']='massspecgym'
        for column in new_df_smiles.keys():
            new_df_smiles.loc[delete_rows, column]=np.nan
        
        new_spectrums=[]
        for index, row in new_df_smiles.iterrows():
            new_item = molecule_pairs_test.spectrums[index]
            if np.any(np.isnan((row['indexes']))):
                new_item.params['smiles']=''
            new_spectrums.append(new_item)

        from simba.molecule_pairs_opt import MoleculePairsOpt
        new_mols = MoleculePairsOpt(
                    spectrums_original= spectrums_massspecgym,
                    spectrums_unique=new_spectrums,
                    df_smiles=new_df_smiles,
                    indexes_tani_unique=None,
                )

        return new_mols, indexes_tani_multitasking_test_filtered