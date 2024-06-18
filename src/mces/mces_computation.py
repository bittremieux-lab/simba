
from src.train_utils import TrainUtils
from src.molecule_pairs_opt import MoleculePairsOpt
from src.molecular_pairs_set import MolecularPairsSet
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os

class MCES:
    @staticmethod
    def compute_all_mces_results_unique(
        spectrums_original,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
        use_exhaustive=True,
    ):
        """
        compute tanimoto results using unique spectrums
        """

        print("Computing tanimoto results based on unique smiles")

        function_tanimoto=MCES.compute_all_mces_results_exhaustive 

        spectrums_unique, df_smiles = TrainUtils.get_unique_spectra(spectrums_original)

        molecule_pairs_unique, df_results_mces = function_tanimoto(
            spectrums_unique,
            max_combinations=max_combinations,
            limit_low_tanimoto=limit_low_tanimoto,
            max_low_pairs=max_low_pairs,
            use_tqdm=use_tqdm,
            max_mass_diff=max_mass_diff,  # maximum number of elements in which we stop adding new items
            min_mass_diff=min_mass_diff,
            num_workers=num_workers,
            MIN_SIM=MIN_SIM,
            MAX_SIM=MAX_SIM,
            high_tanimoto_range=high_tanimoto_range,
        )
        return MoleculePairsOpt(
            spectrums_original=spectrums_original,
            spectrums_unique=spectrums_unique,
            df_smiles=df_smiles,
            indexes_tani_unique=molecule_pairs_unique.indexes_tani,
        ), df_results_mces

    @staticmethod
    def create_input_df(all_spectrums):
        indexes = [i for i in range(0,len(all_spectrums))]
        list_indexes_0=[]
        list_indexes_1=[]
        list_smiles_0=[]
        list_smiles_1=[]
        for it in itertools.combinations(indexes, 2):
            list_indexes_0.append(it[0])
            list_indexes_1.append(it[1])
            list_smiles_0.append(all_spectrums[it[0]].params['smiles'])
            list_smiles_1.append(all_spectrums[it[1]].params['smiles'])
        
        input_df=pd.DataFrame()
        input_df['indexes_0']=list_indexes_0
        input_df['indexes_1']=list_indexes_1
        input_df['smiles_0']=list_smiles_0
        input_df['smiles_1']=list_smiles_1

        return input_df
    
    @staticmethod
    def compute_all_mces_results_exhaustive(
        all_spectrums,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
    ):

        print("Starting computation of molecule pairs")
        print(datetime.now())
        # asume the spectra is already ordered previously
        #all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)

        # indexes=[]
        first_row=0
        max_row=int(len(all_spectrums))
        M= max_row-first_row
        N= len(all_spectrums)

        exhaustive_combinations = M*N
       

        print(f"Number of workers: {num_workers}")

        ## CREATE temp df
        df = MCES.create_input_df(all_spectrums)

        print(f'Number of combinations: {df.shape[0]}')
        print(df)
        df[['smiles_0', 'smiles_1']].to_csv('./input.csv', header=False)

        # compute mces
        command = 'myopic_mces  ./input.csv ./output.csv'
        os.system(command)

        # read results
        results= pd.read_csv('./output.csv', header=None)
        os.system('rm ./input.csv')
        os.system('rm ./output.csv')
        df['mces'] = results[2] # the column 2 is the mces result

        # normalize mces. the higher the mces the lower the similarity
        df['mces_normalized'] = df['mces'].apply(lambda x:x if x<=6 else 6)
        df['mces_normalized'] = df['mces_normalized'].apply(lambda x:(1-x/6))

        # get indexes_np array
        indexes_np = np.zeros(((df.shape[0]), 3),  dtype=np.float16)
        indexes_np[:,0]= df['indexes_0']
        indexes_np[:,1]= df['indexes_1']
        indexes_np[:,2]= df['mces_normalized']

        #print('Remove similarities not computed')
        #indexes_np = indexes_np[indexes_np[:,2]<=1]
        # 
        # avoid duplicates:
        #print(f"Number of effective pairs originally computed: {indexes_np.shape[0]} ")
        #indexes_np = np.unique(indexes_np, axis=0)

        #remove reordered 

        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectrums=all_spectrums, indexes_tani=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set, df