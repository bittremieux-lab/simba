from src.load_mces.load_mces import LoadMCES
from src.mces.mces_computation import MCES
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from src.config import Config

config=Config()


## CODE TO MERGE DATA FROM EDIT DISTANCE CALCULATIONS AND MCES 20

data_folder= '/scratch/antwerpen/209/vsc20939/data/'
INPUT_FOLDER_ED_format= 'preprocessing_mces_threshold20_matching_ed_'
NUMBER_MCES_FOLDERS=10
OUTPUT_FOLDER= data_folder + 'preprocessing_mces20_edit_distance_merged_20240912/'
N_OUTPUT_SPLITS= 30
SPLITS= ['_val', '_test','_train']


# the edit distance data is saved in files starting with INPUT_SPECIFIC_PAIRS
# the mces 20 data is saved in files  starting with indexes_tani...
for split in SPLITS:
    print(split)
    ## Load data EDIT DISTANCE
    edit_distance_data = LoadMCES.load_mces_20_data(directory_path=data_folder+ INPUT_FOLDER_ED_format,
                                                prefix="INPUT_SPECIFIC_PAIRS_indexes_tani_incremental"+ split, 
                                                number_folders=NUMBER_MCES_FOLDERS)
                                                
    # select only edit distance data which is the 4th one
    edit_distance_data = edit_distance_data[:, [0,1,2]]
    print(f'edit distance: {edit_distance_data}')


    for index in range(0, NUMBER_MCES_FOLDERS):
        print(f'mcs folder: {index}')
        mces_data= LoadMCES.load_raw_data(directory_path=data_folder+INPUT_FOLDER_ED_format + str(index),
                                                prefix="indexes_tani_incremental"+ split) 
        




    
        print(f'mces distance: {mces_data}')

        ## Trasform MCES 20 to normalized MCES 20 (0-1) f(x)=1/(1 + log(x))   
        mces_data[:,2] = MCES.exp_normalize_mces20(mces_data[:,2], scale=config.NORMALIZATION_MCES_20)


        # transform to dfs:
        df_edit_distance_data= pd.DataFrame()
        df_edit_distance_data['index_0']=edit_distance_data[:,0]
        df_edit_distance_data['index_1']=edit_distance_data[:,1]
        df_edit_distance_data['ed']=edit_distance_data[:,2]
        df_edit_distance_data['new_key']= df_edit_distance_data.apply(lambda x:(str(x['index_0'])+'_' + str(x['index_1'])), axis=1).astype(str)

        df_mces_data= pd.DataFrame()
        df_mces_data['index_0']=mces_data[:,0]
        df_mces_data['index_1']=mces_data[:,1]
        df_mces_data['mces']=mces_data[:,2]
        df_mces_data['new_key']= df_mces_data.apply(lambda x:(str(x['index_0'])+'_' + str(x['index_1'])), axis=1).astype(str)

        print('Setting key as index:')
        df_edit_distance_data=df_edit_distance_data.set_index('new_key')   
        df_mces_data=df_mces_data.set_index('new_key')   


        print('Edit distance:')
        print(df_edit_distance_data)
        print('MCES:')
        print(df_mces_data)

        # Merge the DataFrames, keeping the columns from the left DataFrame
        print('Mergin...')
        df_joint = pd.merge(df_mces_data, df_edit_distance_data, on='new_key', suffixes=('', '_drop'))

        print('Merged :)')
        # Drop the duplicate columns from the right DataFrame that are not needed
        df_joint = df_joint.drop([col for col in df_joint.columns if col.endswith('_drop')], axis=1)

        print('joint:')
        print(df_joint)

        print(f'shape of edit_distance_data {edit_distance_data.shape[0]}')
        print(f'shape of mces_data {mces_data.shape[0]}')
        

        ## Split the array and save
        if not(os.path.exists(OUTPUT_FOLDER)):
            os.mkdir(OUTPUT_FOLDER)
        
        #extract the np array
        np_array = df_joint[['index_0', 'index_1','ed', 'mces', ]].to_numpy()

        print('example of array to be saved')
        print(np_array)

        #convert to a list, overwrite to avoid memory issues
        np_array= np.array_split(np_array, N_OUTPUT_SPLITS )

        
        print('Saving numpy arrays ...')
        for sub_index, array in enumerate(np_array):
            path_file = OUTPUT_FOLDER + 'indexes_tani_incremental' + split + '_'+ str(index) + '_' + str(sub_index)+ '.npy'
            np.save( path_file,array)

    print('Saved numpy arrays')