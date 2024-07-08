
from src.train_utils import TrainUtils
from src.molecule_pairs_opt import MoleculePairsOpt
from src.molecular_pairs_set import MolecularPairsSet
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
import subprocess
import random 
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

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

        print("Computing MCES results based on unique smiles")

        function_tanimoto=MCES.compute_all_mces_results_exhaustive 

        spectrums_unique, df_smiles = TrainUtils.get_unique_spectra(spectrums_original)

        molecule_pairs_unique = function_tanimoto(
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
        )


                
    @staticmethod
    def create_combinations(all_spectrums):

        print(f'Number of unique spectra:{len(all_spectrums)}')
        indexes = [i for i in range(0,len(all_spectrums))]
        combinations= list(itertools.combinations(indexes, 2))
        return combinations
    
    @staticmethod
    def create_input_df(smiles, indexes_0, indexes_1):
        df=pd.DataFrame()
        print(f'Length of spectrums: {len(smiles)}')

        df['smiles_0']= [smiles[int(r)]  for r in indexes_0]
        df['smiles_1']= [smiles[int(r)]  for r in indexes_1]

        return df
    
    @staticmethod
    def normalize_mces(mces, max_mces=5):
        # normalize mces. the higher the mces the lower the similarity
        mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        return mces_normalized.apply(lambda x:(1-(x/max_mces)))
    
    def compute_mces_myopic(smiles, sampled_index, size_batch, id):

        indexes_np = np.zeros((int(size_batch), 3),)
        indexes_np[:,0] = np.random.randint(0,len(smiles), int(size_batch))
        indexes_np[:,1] = np.random.randint(0,len(smiles), int(size_batch))
        
        #indexes_np[:,0]=sampled_index*np.ones((size_batch,))
        #indexes_np[:,1]= np.arange(0,len(all_spectrums))

        print('Creating df for computation')
        #print('ground truth')
        #print(random_first_index[index:index+size_batch])
        df = MCES.create_input_df(smiles, indexes_np[:,0], 
                                                indexes_np[:,1])

        print('Saving df ...')
        df[['smiles_0', 'smiles_1']].to_csv(f'./input_{str(id)}.csv', header=False)

        # compute mces
        #command = 'myopic_mces  ./input.csv ./output.csv'
        print('Running myopic ...')
        command = ['myopic_mces']
        #command = ['myopic_mces', f'./input_{str(id)}.csv', f'./output_{str(id)}.csv']

        # Add the argument --num_jobs 15
        #command.extend(['--num_jobs', '32'])
        command.extend([ f'./input_{str(id)}.csv'])
        command.extend([f'./output_{str(id)}.csv'])
        command.extend(['--num_jobs', '1'])

        # Define threshold
        command.extend(['--threshold', '5'])

        command.extend(['--solver_onethreaded'])
        command.extend(['--solver_no_msg'])

        #x = subprocess.run(command,capture_output=True)
        subprocess.run(command)
        print('Finished myopic')

        # read results
        print('reading csv')
        results= pd.read_csv(f'./output_{str(id)}.csv', header=None)
        os.system(f'rm ./input_{str(id)}.csv')
        os.system(f'rm ./output_{str(id)}.csv')
        df['mces'] = results[2] # the column 2 is the mces result

        # normalize mces. the higher the mces the lower the similarity
        df['mces_normalized'] = MCES.normalize_mces(df['mces'])

        #print('saving intermediate results')
        indexes_np[:,2]= df['mces_normalized'].values
        return indexes_np 

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


        print(f"Number of workers: {num_workers}")
        size_batch = 10000
        number_sampled_spectrums = np.floor(max_combinations/size_batch)
        random_samples = np.random.randint(0,len(all_spectrums), int(number_sampled_spectrums))

        # Use ProcessPoolExecutor for parallel processing
        smiles=[s.params['smiles'] for s in all_spectrums]

        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_workers)
        # Generate arrays in parallel
        results = [pool.apply_async(MCES.compute_mces_myopic, args=(smiles, 
                                sampled_index, 
                                size_batch, 
                                identifier)) for identifier, sampled_index in enumerate(random_samples)]
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
    
        # Get results from async objects
        indexes_np = np.concatenate([result.get() for result in results], axis=0)


        #with ProcessPoolExecutor(max_workers=num_workers) as executor:
        #    # Submit tasks to the executor
        #    manager = multiprocessing.Manager()
        #    smiles = manager.list(smiles)  # Create a managed list
        #    futures = [executor.submit(MCES.compute_mces_myopic, 
        #                        smiles, 
        #                        sampled_index, 
        #                        size_batch, 
        #                        identifier) for identifier, sampled_index in enumerate(random_samples)]

        #    # Collect results as they complete
        #    results = []
        #    for future in as_completed(futures):
        #        results.append(future.result())

        # Concatenate the results along the first axis
        #indexes_np = np.concatenate(results, axis=0)
        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")


        print('Example of pairs retreived:')
        print(indexes_np[0:1000])
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectrums=all_spectrums, indexes_tani=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set


    @staticmethod
    def compute_all_mces_results_exhaustive_single_thread(
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


        print(f"Number of workers: {num_workers}")

        print('Initializing big array')
        size_batch = len(all_spectrums)
        number_sampled_spectrums = np.floor(max_combinations/size_batch)
        indexes_np = np.zeros((int(number_sampled_spectrums*size_batch), 3),)
        random_samples = np.random.randint(0,len(all_spectrums), int(number_sampled_spectrums))

        #for index in tqdm(range(0,max_combinations, size_batch)):
        for i, sampled_index in tqdm(enumerate(random_samples)):
        #for index in tqdm(range(0, size_combinations,size_batch)):
            # compute the consecutive pairs
            indexes_np[i:i+size_batch,0]=sampled_index*np.ones((size_batch,))
            indexes_np[i:i+size_batch,1]= np.arange(0,len(all_spectrums))

            print(f'{sampled_index}')
            print('Creating df for computation')

            #print('ground truth')
            #print(random_first_index[index:index+size_batch])
            df = MCES.create_input_df(all_spectrums, indexes_np[:,0][i:i+size_batch], 
                                                    indexes_np[:,1][i:i+size_batch])

            print('Saving df ...')
            df[['smiles_0', 'smiles_1']].to_csv('./input.csv', header=False)

            # compute mces
            #command = 'myopic_mces  ./input.csv ./output.csv'
            print('Running myopic ...')
            command = ['myopic_mces', './input.csv', './output.csv']

            # Add the argument --num_jobs 15
            #command.extend(['--num_jobs', '32'])
            command.extend(['--num_jobs', '32'])

            # Define threshold
            command.extend(['--threshold', '5'])

            #command.extend(['--solver_onethreaded'])
            command.extend(['--solver_no_msg'])

            #x = subprocess.run(command,capture_output=True)
            subprocess.run(command)
            print('Finished myopic')

            # read results
            print('reading csv')
            results= pd.read_csv('./output.csv', header=None)
            #os.system('rm ./input.csv')
            #os.system('rm ./output.csv')
            df['mces'] = results[2] # the column 2 is the mces result

            # normalize mces. the higher the mces the lower the similarity
            df['mces_normalized'] = MCES.normalize_mces(df['mces'])
 
            #print('saving intermediate results')
            indexes_np[i:i+size_batch,2]= df['mces_normalized'].values


        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")


        print('Example of pairs retreived:')
        print(indexes_np[0:1000])
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectrums=all_spectrums, indexes_tani=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set, df