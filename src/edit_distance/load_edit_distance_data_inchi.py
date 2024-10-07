from functools import lru_cache
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from tqdm import tqdm

class LoadEdistanceDataInchi:   
    '''
    code to load edit distance data generated with inchis on september 2024
    '''
    
    @lru_cache(maxsize=10000)
    def get_smiles_from_inchi_rdkit(inchi_string):
        mol = Chem.MolFromInchi(inchi_string)
        if mol:
            return Chem.MolToSmiles(mol)
        else:
            return ''

    def generate_smiles_csv(df, columns, ):
        df['smiles_0']=df['inchi1'].apply(lambda x:LoadEdistanceDataInchi.get_smiles_from_inchi_rdkit(x))
        df['smiles_1']=df['inchi2'].apply(lambda x:LoadEdistanceDataInchi.get_smiles_from_inchi_rdkit(x))
        df['inchi_0']=''
        df['inchi_1']=''
        df['edit_distance']=df['distance']
        df['substructure']=df['is_sub']
        df = df[(df['smiles_0']!='')& (df['smiles_1']!='') ]
        return df[columns]

    def generate_smiles_and_save(df, index, output_folder, csv_new_names):
        '''
        from a list of files save the results in the output folder
        '''
        output_file_name= 'ed_data_' + str(index) + '.csv'
        df_output=LoadEdistanceDataInchi.generate_smiles_csv(df, columns=csv_new_names,)
        df_output.to_csv(output_folder + output_file_name)

    def forward(path_file, output_folder, 
                            chunk_size=10, number_chunks=2, 
                csv_new_names= ['smiles_0', 'smiles_1', 'edit_distance', 'tanimoto', 'substructure','inchi_0', 'inchi_1']):
        '''
        it takes the input csv file and save all the results to the output folder
        '''
    
        
        # Initialize an empty list to collect processed data (if needed)
        processed_data = []
        
        # Iterate over the CSV file in chunks
        generator=pd.read_csv(path_file, chunksize=chunk_size)
        for chunk_index, df_chunk in tqdm(enumerate(generator)):
            
            # Process the chunk (e.g., filtering, aggregation, etc.)
            # For example, you can print the size of each chunk:            
            # If processing data, append the result to a list
            LoadEdistanceDataInchi.generate_smiles_and_save(df_chunk, chunk_index, output_folder, csv_new_names)
    
            if (chunk_index+1)==number_chunks:
                break