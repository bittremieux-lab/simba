import numpy as np
import pickle
from src.config import Config
import os

config=Config()
## merge edit distance and mces data
## make sure the indexes are alligned
## make sure to define not defined values to the desired values

# edit distance data
ed_data_path= '/scratch/antwerpen/209/vsc20939/data/preprocessing_edit_distance_20250117/'
mces_data_path= '/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_20250118/'
output_path= '/scratch/antwerpen/209/vsc20939/data/preprocessing_ed_mces_20250123/'

# create folder if it does not exist
if not(os.path.exists(output_path)):
    os.mkdir(output_path)


#file_path= data_path + 'mces_indexes_tani_incremental_train_1.npy'
#molecular_file= data_path + 'edit_distance_neurips_nist_exhaustive.pkl'

# List all files in the directory
all_files = os.listdir(ed_data_path)


def preprocess_data(array):
    # if any of edit distance or mces is undefined, then we have to put mces and edit distance to a high value (e.g 666)
    ed = array[:,config.COLUMN_EDIT_DISTANCE ]
    mces=  array[:,config.COLUMN_MCES20 ]

    indexes_unvalid = np.isnan(ed) | np.isnan(mces)
    #array[indexes_unvalid, config.COLUMN_EDIT_DISTANCE]=config.EDIT_DISTANCE_MAX_VALUE
    #array[indexes_unvalid, config.COLUMN_MCES20]=config.THRESHOLD_MCES

    
    # make sure that any value of the columns does not exceed the permitted value
    ed_exceeded= array[:,config.COLUMN_EDIT_DISTANCE ] == 666
    #array[ed_exceeded, config.COLUMN_EDIT_DISTANCE]= config.EDIT_DISTANCE_MAX_VALUE

    mces_exceeded= array[:,config.COLUMN_MCES20 ]==666
    #array[mces_exceeded, config.COLUMN_MCES20]= config.THRESHOLD_MCES

    # remove invalid values
    array = array[(~indexes_unvalid) & (~ed_exceeded) &  (~mces_exceeded)]
    return array

for partition in ['train', 'val', 'test']:
    # Filter only the files ending with '.npy'
    npy_files = [f for f in all_files if f.endswith('.npy')]
    npy_files= [f for f in npy_files if partition in f]
    for file_loaded in npy_files: 
            # get the index
            index = file_loaded.split('_')[-1].split('.npy')[0]
            index= int(index)

            print(f'Loading index: {index}')

            sufix = "indexes_tani_incremental_"  + partition +'_'
            prefix_ed= 'edit_distance_'
            prefix_mces= 'mces_'
            prefix_output= 'ed_mces_'

            file_ed = prefix_ed + sufix + str(index) + '.npy'
            file_mces = prefix_mces + sufix + str(index)+ '.npy'
            file_output= prefix_output + sufix  + str(index) + '.npy'

            # add directory
            file_ed= ed_data_path + file_ed
            file_mces= mces_data_path + file_mces
            file_output= output_path + file_output

            # load data
            ed_data= np.load(file_ed)
            print(f'ed_data: {ed_data}')

            try:
                mces_data= np.load(file_mces)
                print(f'mces_data {mces_data}')
            except:
                print('The MCES partition is not present.')
                continue


        

            ## check that the indexes are the same:
            if np.all(ed_data[:,0]==mces_data[:,0]) and np.all(ed_data[:,1]==mces_data[:,1]):
                print('The data loaded correspond to the same pairs')
                
                total_data= np.column_stack((ed_data,mces_data[:,2]))
                print(f'The data loaded has the original shape: {total_data.shape}')

                print('Preprocessing:')
                total_data = preprocess_data(total_data)

                print(total_data)
                np.save(file_output, total_data, )
            else:
                print('ERROR: The data loaded does not correspond to the same pairs')