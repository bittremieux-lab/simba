
from simba.train_utils import TrainUtils
from simba.molecule_pairs_opt import MoleculePairsOpt
from simba.molecular_pairs_set import MolecularPairsSet
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
from simba.edit_distance.edit_distance import EditDistance
from rdkit.Chem import AllChem
from rdkit import Chem, Geometry
from simba.load_mces.load_mces import LoadMCES
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
        random_sampling=True,
        config=None,
        identifier="",
        use_edit_distance=False,
        loaded_molecule_pairs=None
    ):
        """
        compute tanimoto results using unique spectrums
        """

        print("Computing MCES results based on unique smiles")

        if loaded_molecule_pairs is None:
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
                random_sampling=random_sampling,
                config=config,
                identifier=identifier,
                use_edit_distance=use_edit_distance,
            )
        else:
            molecule_pairs_unique= loaded_molecule_pairs
            df_smiles= molecule_pairs_unique.df_smiles
            spectrums_unique=molecule_pairs_unique.spectrums    
            spectrums_original=molecule_pairs_unique.spectrums_original,

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
        print(f'Length of smiles array: {len(smiles)}')
        print(f'Max value of indexes_0:{max(indexes_0)}')
        print(f'Max value of indexes_0:{max(indexes_1)}')
        df['smiles_0']= [smiles[int(r)]  for r in indexes_0]
        df['smiles_1']= [smiles[int(r)]  for r in indexes_1]

        return df
    
    @staticmethod
    def normalize_mces(mces, max_mces=5):
        # asuming series
        # normalize mces. the higher the mces the lower the similarity
        #mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        #return mces_normalized.apply(lambda x:(1-(x/max_mces)))

        ## asuming numpy
        print(f'Example of input mces: {mces}')
        mces_normalized = mces.copy()
        mces_normalized[mces_normalized >= max_mces] = max_mces 
        mces_normalized = 1 - (mces_normalized/max_mces)
        print(f'Example of normalized mces: {mces_normalized}')
        return mces_normalized

    def compute_mces_myopic(smiles, sampled_index, size_batch, id, random_sampling, 
                config,
                split_group='train',# if it is train, val or test
                to_compute_indexes_np=None,  # the indexes to be computed
                ):

        # where to save results
        
        # initialize randomness
        
        
        if config.COMPUTE_SPECIFIC_PAIRS:
            size_batch_effective = to_compute_indexes_np.shape[0]
            indexes_np = np.zeros((int(size_batch_effective), 3),)
            indexes_np[:,0:2]=to_compute_indexes_np[:,0:2]
        else:
            indexes_np = np.zeros((int(size_batch), 3),)
            if random_sampling:
                np.random.seed(id)
                indexes_np[:,0] = np.random.randint(0,len(smiles), int(size_batch))
                indexes_np[:,1] = np.random.randint(0,len(smiles), int(size_batch))
            else:
                indexes_np[:,0] = sampled_index
                indexes_np[:,1]= np.arange(0, size_batch)

        #indexes_np[:,0]=sampled_index*np.ones((size_batch,))
        #indexes_np[:,1]= np.arange(0,len(all_spectrums))

        print('Creating df for computation')
        #print('ground truth')
        #print(random_first_index[index:index+size_batch])
        df = MCES.create_input_df(smiles, indexes_np[:,0], 
                                                indexes_np[:,1])

        print('Saving df ...')
        df[['smiles_0', 'smiles_1']].to_csv(f'{config.PREPROCESSING_DIR}input_{str(id)}.csv', header=False)

        # compute mces
        #command = 'myopic_mces  ./input.csv ./output.csv'
        print('Running myopic ...')
        command = ['myopic_mces']
        #command = ['myopic_mces', f'./input_{str(id)}.csv', f'./output_{str(id)}.csv']

        # Add the argument --num_jobs 15
        #command.extend(['--num_jobs', '32'])
        command.extend([ f'{config.PREPROCESSING_DIR}input_{str(id)}.csv'])
        command.extend([f'{config.PREPROCESSING_DIR}output_{str(id)}.csv'])
        command.extend(['--num_jobs', '1'])
        #command.extend(['--solver','CPLEX_CMD'])
        # Define threshold
        command.extend(['--threshold', str(int(config.THRESHOLD_MCES))])

        command.extend(['--solver_onethreaded'])
        command.extend(['--solver_no_msg'])
        command.extend(['--choose_bound_dynamically'])

        #x = subprocess.run(command,capture_output=True)
        subprocess.run(command)
        print('Finished myopic')

        # read results
        print('reading csv')
        results= pd.read_csv(f'{config.PREPROCESSING_DIR}output_{str(id)}.csv', header=None)
        os.system(f'rm {config.PREPROCESSING_DIR}input_{str(id)}.csv')
        os.system(f'rm {config.PREPROCESSING_DIR}output_{str(id)}.csv')
        df['mces'] = results[2] # the column 2 is the mces result

        # normalize mces. the higher the mces the lower the similarity
        #df['mces_normalized'] = MCES.normalize_mces(df['mces'])
        df['mces_normalized'] = df['mces']
        
        #print('saving intermediate results')
        indexes_np[:,2]= df['mces_normalized'].values
        return indexes_np 

    def get_samples(all_spectrums,random_sampling,max_combinations,size_batch_no_random=100):
        '''
        get sample indexes if we do random or deterministic sampling
        '''
        if random_sampling:
            print('Random sampling')
            size_batch = size_batch_no_random
            number_sampled_spectrums = np.floor(max_combinations/size_batch)
            samples = np.random.randint(0,len(all_spectrums), int(number_sampled_spectrums))
        else:
            print('No random sampling')
            size_batch = len(all_spectrums)
            samples = np.arange(0, len(all_spectrums))
        return samples, size_batch

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
        random_sampling=True,
        config=None,
        identifier = "",
        use_edit_distance=False,
    ):

        print("Starting computation of molecule pairs")
        print(datetime.now())


        print(f"Number of workers: {num_workers}")

        samples,size_batch = MCES.get_samples(all_spectrums,random_sampling,max_combinations)

        # Use ProcessPoolExecutor for parallel processing
        smiles=[s.params['smiles'] for s in all_spectrums]

       
        N= num_workers*10
        indexes_np= np.array([])

        if config.COMPUTE_SPECIFIC_PAIRS: ## specifically for mces
                directory_path=config.PREPROCESSING_DIR 
                prefix= config.FORMAT_FILE_SPECIFIC_PAIRS  +  identifier
                indexes_np_loaded = LoadMCES.load_raw_data(directory_path, prefix)
                #indexes_np_loaded = LoadMCES.remove_excess_low_pairs(indexes_np_loaded, remove_percentage=0.99, max_mces=config.THRESHOLD_MCES)

                print(f'Size of the pairs loaded for computing specific pairs: {indexes_np_loaded.shape[0]}')

                size_chunks = config.PREPROCESSING_BATCH_SIZE*num_workers
                N= int(np.ceil(indexes_np_loaded.shape[0]/size_chunks))
                split_arrays = np.array_split(indexes_np_loaded, N)


                
        else:
                 # Calculate the number of chunks needed
                num_chunks = int(np.ceil(len(samples) / N))
                # Split the array into chunks of size N using np.array_split
                split_arrays = np.array_split(samples, num_chunks)
                
        print(f'Number of split arrays: {len(split_arrays)}')
        print(f'Size of each array: {split_arrays[0].shape[0]}')
        print(f'Size of each sub-array: {np.array_split(split_arrays[0], num_workers)[0].shape[0]}')

        # using the loading of csv 
        #comp_function = EditDistance.compute_edit_distance if use_edit_distance else MCES.compute_mces_myopic

        # using the edit distance repository
        #comp_function = EditDistance.compute_edit_distance if use_edit_distance else EditDistance.compute_mces_myopic
        comp_function = EditDistance.compute_ed_or_mces

        ## SINCE I WANT TO EXECUTE THE PREPROCESSING ACROSS SEVERAL NDOES, I WANT TO EXECUTE ONLY SOME indexes

        for index_array, array in enumerate(split_arrays):
                # Create a multiprocessing pool
                #pool = multiprocessing.Pool(processes=num_workers)      ## USE MULTIPLE PROCESSES - PRIVATE MEMORY


                #if (index_array>38) and ((index_array%config.PREPROCESSING_NUM_NODES)==config.PREPROCESSING_CURRENT_NODE):
                if ((index_array%config.PREPROCESSING_NUM_NODES)==config.PREPROCESSING_CURRENT_NODE):
                    prefix_file= 'edit_distance_' if use_edit_distance else 'mces_'
                    name_file=f'{config.PREPROCESSING_DIR}'+ prefix_file + f'indexes_tani_incremental{identifier}_{str(index_array)}.npy'
                    
                    if not(os.path.exists(name_file)): ## by default not overwriting
                        print(f'Processing index_array: {index_array}')
                        #pool = multiprocessing.dummy.Pool(processes=num_workers)      ## USE MULTIPLE PROCESSES - PRIVATE MEMOR 
                        pool = multiprocessing.Pool(processes=num_workers)   

                        print(f'Value of COMPUTE_SPECIFIC_PAIRS: {config.COMPUTE_SPECIFIC_PAIRS}')
                        if config.COMPUTE_SPECIFIC_PAIRS: 
                                print(f'Size of each array {array.shape[0]}')
                                sub_arrays = np.array_split(array, num_workers)
                                print(f'Size of each sub-array {sub_arrays[0].shape[0]}')
                        
                                results = [pool.apply_async(comp_function, args=(smiles, 
                                                    None, 
                                                    size_batch, 
                                                    (index_array*split_arrays[0].shape[0])+sub_index,
                                                    None, 
                                                    config,
                                                    identifier,
                                                    sampled_array
                                                    )) for sub_index, sampled_array in enumerate(sub_arrays)]
                        else:
                                    #if use_edit_distance:

                                    print('Compute the mols')
                                    
                                    mols = [Chem.MolFromSmiles(s) for s in smiles]
                                    print('Computing fingerprints')
                                    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3,fpSize=512)
                                    fps = [fpgen.GetFingerprint(m) for m in mols]

                                    print('Finished fongerprints')
                                    results = [pool.apply_async(comp_function, args=(smiles,
                                                            sampled_index, 
                                                            size_batch, 
                                                            (index_array*len(split_arrays[0]))+sub_index,
                                                            random_sampling, config, fps, mols,
                                                            use_edit_distance,
                                                            )) for sub_index, sampled_index in enumerate(array)]
                                    #else:


                                    #results = [pool.apply_async(comp_function, args=(smiles, 
                                    #                        sampled_index, 
                                    #                        size_batch, 
                                    #                        sampled_index,
                                    #                        #(index_array*len(split_arrays[0]))+sub_index,
                                    #                        random_sampling, config,
                                    #                        )) for sub_index, sampled_index in enumerate(array)]
                            
                            
                        # Close the pool and wait for all processes to finish
                        pool.close()
                        pool.join()
                    
                        # Get results from async objects
                        indexes_np_temp = np.concatenate([result.get() for result in results], axis=0)

                        np.save(arr=indexes_np_temp, file= name_file)

        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")


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
            command.extend(['--num_jobs', '1'])
            #command.extend(['--num_jobs', '32'])
            #command.extend(['--num_jobs', '32'])

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

    def exp_normalize_mces20(x, scale=20, low_threshold=0.20):
        '''
        normalize the input np array 

        values of 0 are set to 1
        values of infinite are set to 0

        if we set scale=2, the mces distance of 20 corresponds to ~0.1
        '''
        mces_normalized= 1/(1+(x/scale))

        mces_normalized[mces_normalized<=low_threshold] = 0

        return mces_normalized

    def inverse_exp_normalize_mces20(mces_normalized, scale, epsilon=0.000000000001):
    
        #add epsilon to avoid divide by 0
        mces_normalized_epsilon= mces_normalized +epsilon

        return scale *((1/mces_normalized_epsilon)-1)



    def compute_mces_list_smiles(smiles_0, smiles_1, threshold_mces=20):

        if not(os.path.exists(f'temp')):
            os.mkdir(f'temp')

        input_csv_file=f'temp/smiles_myopic_input.csv'
        output_csv_file=f'temp/smiles_myopic_output.csv'

        df= pd.DataFrame()

        df['smiles_0']=smiles_0
        df['smiles_1']=smiles_1

        df[['smiles_0', 'smiles_1']].to_csv(input_csv_file, header=False)

        # compute mces
        #command = 'myopic_mces  ./input.csv ./output.csv'
        print('Running myopic ...')
        command = ['myopic_mces']
        #command = ['myopic_mces', f'./input_{str(id)}.csv', f'./output_{str(id)}.csv']

        # Add the argument --num_jobs 15
        #command.extend(['--num_jobs', '32'])
        command.extend([input_csv_file])
        command.extend([output_csv_file])
        command.extend(['--num_jobs', '1'])

        # Define threshold
        command.extend(['--threshold', str(int(threshold_mces))])

        command.extend(['--solver_onethreaded'])
        command.extend(['--solver_no_msg'])

        #x = subprocess.run(command,capture_output=True)
        subprocess.run(command)
        print('Finished myopic')

        # read results
        print('reading csv')
        results= pd.read_csv(output_csv_file, header=None)
        os.system(f'rm {input_csv_file}')
        os.system(f'rm {output_csv_file}')
        df['mces'] = results[2] # the column 2 is the mces result


        os.system(f'rm -r temp')
        return df