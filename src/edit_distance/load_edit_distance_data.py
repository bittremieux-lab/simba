import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from rdkit import Chem
import pandas as pd
import os
from tqdm import tqdm 
import gzip 
import csv

class LoadEditDistanceData:
    '''
    class to load data from ming processing
    '''

    def load_file(path_file, names= ['smiles_0', 'smiles_1', 'edit_distance', 'tanimoto', 'substructure','inchi_0', 'inchi_1']
        ):
        #df_rs = pd.read_csv(path_file, header=None, delimiter=',')
        
        # use default but in case of error, try engine python because of problem with files

        try:
            df_rs = pd.read_csv(path_file,  names=names,
                    on_bad_lines='warn')
        except:
            print('Error using default engine, trying python (slower but more effective)')
            #load all the lines
            df_rs = pd.read_csv(path_file,  
                    names=['raw_data'],
                    delimiter="\t",    
                    engine='python',
                    on_bad_lines='warn')
            
            # remove the firs tand last row
            df_rs=df_rs.loc[1:df_rs.shape[0]-2,:].reset_index(drop=True)

            # remove the last 2 columns
            df_rs['raw_data'] = df_rs['raw_data'].apply(lambda x:x.split('"')[0] + 'False')
            df_rs.to_csv(path_file, header=None, index=False)

            df_rs = pd.read_csv(path_file,  names=names, 
                           delimiter=',',
                           quotechar="'",
                    on_bad_lines='warn')
            
            # eliminate possible quote symbols
            df_rs['smiles_0']=df_rs['smiles_0'].apply(lambda x:x.split('"')[-1])
        #df_rs  = df_rs.drop(index=0)
        # the first row is defective
        #df_rs
        #df_rs = df_rs.rename(columns=rename_dict)
        return df_rs 
    
    def get_dataset(file_path):
        with open(file_path, 'rb') as f:
            dataset= pickle.load(f)
        return dataset
    
    
    # get a dictionary with the canonical smiles based on no canonical smiles
    def get_canon(smiles):
            canon_smiles= {}
            for s in smiles:
                try:
                    temp_smiles=Chem.CanonSmiles(s)
                except:
                    print('Error parsing smiles')
                    temp_smiles=s
                canon_smiles[s] = temp_smiles
            return canon_smiles

    def get_matched_rows(df_rs, specs):
        # a list wiht all the smiles in the spectrum domain
        our_smiles = [s.params['smiles'] for s in specs]
        # dictionary that maps the no canon smiles to canon smiles in the spec domain
        our_canon_smiles_dict = {s.params['smiles']:Chem.CanonSmiles(s.params['smiles']) for s in specs}
        our_canon_smiles_list = [our_canon_smiles_dict[k] for k in our_canon_smiles_dict]

        # going from canon to non canon in the spec domain
        from_canon_to_no_canon = {our_canon_smiles_dict[k]:k for k in our_canon_smiles_dict}

        #cast to string
        df_rs['smiles_0']= df_rs['smiles_0'].astype(str)
        df_rs['smiles_1']= df_rs['smiles_1'].astype(str)
        
        # get all the unique smiles of pair 0
        unique_ming_smiles_0 = np.unique(df_rs['smiles_0'])
        # get all the unique smiles of pair 1
        unique_ming_smiles_1 = np.unique(df_rs['smiles_1'])

        # get the canonical smiles of unique
        canon_unique_ming_smiles_0= LoadEditDistanceData.get_canon(unique_ming_smiles_0)
        canon_unique_ming_smiles_1= LoadEditDistanceData.get_canon(unique_ming_smiles_1)

        # get the uniquelist of smiles
        list_unique_canon_smiles_0 = [canon_unique_ming_smiles_0[k] for k in canon_unique_ming_smiles_0]
        list_unique_canon_smiles_1 = [canon_unique_ming_smiles_1[k] for k in canon_unique_ming_smiles_1]

        ## Find the unique matches in the unique list
        matched_smiles_0 = [s for s in our_canon_smiles_list if s in list_unique_canon_smiles_0]
        ## Find the matches in ming_smiles_1
        matched_smiles_1 = [s for s in our_canon_smiles_list if s in list_unique_canon_smiles_1]

        # dictionary indexed by canon smiles to indexes in the spec domain
        unique_indexes_tani_0 = {s:our_smiles.index(from_canon_to_no_canon[s]) for s in  matched_smiles_0}
        unique_indexes_tani_1 = {s:our_smiles.index(from_canon_to_no_canon[s]) for s in  matched_smiles_1}

        # modify df_rs
        df_rs['canon_smiles_0']=df_rs['smiles_0'].apply(lambda x:canon_unique_ming_smiles_0[x])
        df_rs['canon_smiles_1']=df_rs['smiles_1'].apply(lambda x:canon_unique_ming_smiles_1[x])
        matched_dict_0 = {s:(s in matched_smiles_0) for s in canon_unique_ming_smiles_0}
        matched_dict_1 = {s:(s in matched_smiles_1) for s in canon_unique_ming_smiles_1}

        #find matches
        df_rs['matched_0'] = df_rs['canon_smiles_0'].apply(lambda x:matched_dict_0[x])
        df_rs['matched_1'] = df_rs['canon_smiles_1'].apply(lambda x:matched_dict_1[x])
        df_rs['matched'] = df_rs['matched_0'] *df_rs['matched_1'] 

        # filteering
        df_rs_filtered = df_rs[df_rs['matched']]
        df_rs_filtered.loc[:,'indexes_tani_0'] = df_rs_filtered['canon_smiles_0'].apply(lambda x:unique_indexes_tani_0[x])
        df_rs_filtered.loc[:,'indexes_tani_1'] = df_rs_filtered['canon_smiles_1'].apply(lambda x:unique_indexes_tani_1[x])

        return df_rs_filtered
    

    def generate_np(df_rs_filtered):
        print('Writing edit distance and tanimoto')
        indexes_tani_np = np.zeros((df_rs_filtered.shape[0],3 ))
        indexes_tani_np[:,0]=df_rs_filtered['indexes_tani_0']
        indexes_tani_np[:,1]=df_rs_filtered['indexes_tani_1']
        indexes_tani_np[:,2]=df_rs_filtered['edit_distance']
        indexes_tani_np[:,3]=df_rs_filtered['tanimoto']
        return indexes_tani_np
    

    def get_files(directory_path, extension='.gz'):
        # List all files in the directory
        all_files = os.listdir(directory_path)

        # Filter out the CSV files
        return [os.path.join(directory_path, file) for file in all_files if file.endswith('.gz')]
    
    def foward(dataset_file, input_folder, output_folder, 
        dict_save= {'molecule_pairs_train': 'train',
                    'molecule_pairs_val': 'val',
                    'molecule_pairs_test': 'test',}
        ):
        '''
        load list of csv files containing edit distance info and save the matched pairs in np arrays
        '''
        # get input files
        list_files = LoadEditDistanceData.get_files(input_folder)

        # load dataset with target spectrums
        dataset= LoadEditDistanceData.get_dataset(dataset_file)


        for index,l in tqdm(enumerate(list_files)):
                print(f'Processing file: {l}')
                # decompress the gz file
                csv_file = l.split('.gz')[-2]+'.csv'
                print('Decompressing ...')
                LoadEditDistanceData.decompress_gz(input_gz_file=l, 
                                                   output_file= csv_file)
                
                print('Loading file')
                df= LoadEditDistanceData.load_file(csv_file)
                
                # loop through train, val and test
                for k in dict_save:
                    print(f'Filtering: {k}')
                    specs =dataset[k].spectrums
                
                    
                    
                    df_filtered = LoadEditDistanceData.get_matched_rows(df,specs=specs)
                    np_array= LoadEditDistanceData.generate_np(df_filtered)

                    #file
                    file_name= output_folder+ f'indexes_tani_incremental_{dict_save[k]}_{str(index)}.npy'
                    np.save(file_name, np_array)

                # delete decompred file
                LoadEditDistanceData.delete_file(csv_file)

    def delete_file(file_path):
        os.system(f'rm {file_path}')

    def decompress_gz(input_gz_file, output_file):
        # Open the .gz file in read mode and the output file in write mode
        with gzip.open(input_gz_file, 'rb') as gz_file:
            with open(output_file, 'wb') as out_file:
                # Read from the compressed file and write to the output file
                out_file.write(gz_file.read())