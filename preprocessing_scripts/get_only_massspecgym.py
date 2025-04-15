molecular_file= '/Users/sebas/projects/data/edit_distance_neurips_nist_exhaustive.pkl'
output_inchi_file= '/Users/sebas/projects/data/inchi_keys_simba.csv'
preprocessing_folder =  '/Users/sebas/projects/data/preprocessing_ed_mces_20250123/'
new_preprocessing_folder =  '/Users/sebas/projects/data/preprocessing_ed_mces_massspecgym/'
molecular_file_output =    '/Users/sebas/projects/data/preprocessing_ed_mces_massspecgym/mapping.pkl'


# In[283]:


import pickle
import sys
import simba
import numpy as np
import pandas as pd
sys.modules['src']=simba


# In[284]:


from simba.config import Config
from simba.load_mces.load_mces import LoadMCES


# In[285]:


config=Config()




# In[287]:


with open(molecular_file, 'rb') as f:
    dataset= pickle.load(f)


# In[288]:


molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]


# In[289]:


## Load molecular file
## Based on the pairs numpy array, filter only those pairs where both molecules are from massspecgym 
## Change the indexes to the new mapping 


# In[290]:


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


# In[291]:


indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_TRAIN,
    prefix="ed_mces_indexes_tani_incremental_test",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
    add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
)

indexes_tani_multitasking_test = remove_duplicates_array(indexes_tani_multitasking_test)


# In[292]:


indexes_tani_multitasking_test


# In[ ]:





# ## Refactor

# In[293]:


from simba.filtering_massspecgym.filtering_massspecgym import FilteringMassSpecGym


# In[304]:


molecule_pairs_test_filtered, indexes_tani_multitasking_test_filtered=FilteringMassSpecGym.filter_massspecgym(
                                molecule_pairs_test, 
                                path_pairs=preprocessing_folder, 
                                config=config,
                                prefix="ed_mces_indexes_tani_incremental_test")


# In[ ]:


molecule_pairs_val_filtered, indexes_tani_multitasking_val_filtered=FilteringMassSpecGym.filter_massspecgym(
                                molecule_pairs_test, 
                                path_pairs=preprocessing_folder, 
                                config=config,
                                prefix="ed_mces_indexes_tani_incremental_val")


# In[ ]:


molecule_pairs_train_filtered, indexes_tani_multitasking_train_filtered=FilteringMassSpecGym.filter_massspecgym(
                                molecule_pairs_test, 
                                path_pairs=preprocessing_folder, 
                                config=config,
                                prefix="ed_mces_indexes_tani_incremental_train")


# In[295]:


import os
if not(os.path.exists(new_preprocessing_folder)):
    os.mkdir(new_preprocessing_folder)


# In[296]:


def write_data(
    file_path,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    uniformed_molecule_pairs_test=None,
):
    dataset = {
        "all_spectrums_train": None,
        "all_spectrums_val": None,
        "all_spectrums_test": None,
        "molecule_pairs_train": molecule_pairs_train,
        "molecule_pairs_val": molecule_pairs_val,
        "molecule_pairs_test": molecule_pairs_test,
        "uniformed_molecule_pairs_test": None,
    }
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)


# ## Write data

# In[297]:


np.save(new_preprocessing_folder + 'ed_mces_indexes_tani_incremental_massspecgym_test.npy', indexes_tani_multitasking_test_filtered)
np.save(new_preprocessing_folder + 'ed_mces_indexes_tani_incremental_massspecgym_val.npy', indexes_tani_multitasking_val_filtered)
np.save(new_preprocessing_folder + 'ed_mces_indexes_tani_incremental_massspecgym_train.npy', indexes_tani_multitasking_train_filtered)


# In[301]:


write_data(molecular_file_output,  all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=molecule_pairs_train_filtered,
    molecule_pairs_val=molecule_pairs_val_filtered,
    molecule_pairs_test=molecule_pairs_test_filtered,
    uniformed_molecule_pairs_test=None,)


# In[ ]:




