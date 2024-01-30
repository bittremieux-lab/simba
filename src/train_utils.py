from itertools import combinations
from tqdm import tqdm
from src.tanimoto import Tanimoto
import numpy as np
import random
from src.molecule_pair import MoleculePair
from  src.preprocessor import Preprocessor
from src.preprocessing_utils import PreprocessingUtils
from src.config import Config
import functools
import random 
from src.molecular_pairs_set import MolecularPairsSet
import concurrent.futures
from datetime import datetime
from rdkit import Chem
from itertools import product

class TrainUtils:

    @staticmethod
    def compute_unique_combinations(molecule_pairs,pairs_per_compound=40):
        '''
        get pairs with sim=1
        '''
        # get spectra
        all_spectrums = molecule_pairs.spectrums

        ## How many unique smiles there are?
        smiles = [spec.smiles for spec in all_spectrums]
        #smiles = [s for s in smiles if s!= '']

        # compute canon smiles
        for s in smiles:
            try:
                s = Chem.CanonSmiles(s)
            except:
                s = 'NO_CANON'
        #smiles = [s for s in smiles if s!= 'NO_CANON']

        # Get unique values and their counts
        unique_values, counts = np.unique(smiles, return_counts=True)
        print(f'Unique compounds: {len(unique_values)}')
        print(f'Mean number of counts per compound: {np.mean(counts)}')
        print(f'std  of counts per compound: {np.std(counts)}')

        list_total=[]
        for u in (unique_values):
            if (u!='NO_CANON') and (u != ''):
                indices = np.where(np.array(smiles) == u)[0]
                index_combinations = list(product(indices, repeat=2))
                random.shuffle(index_combinations)
                list_total = list_total + index_combinations[0:pairs_per_compound]

        lenght_total = len(list_total)

        indexes_np =np.zeros((lenght_total,3))
        print(f'number of pairs: {lenght_total}')
        for index,l in enumerate(list_total):
            indexes_np[index,0] = l[0]
            indexes_np[index,1] = l[1]
            indexes_np[index,2] = 1

        # add info to 
        new_molecule_pairs= MolecularPairsSet(spectrums=molecule_pairs.spectrums,
                                                    indexes_tani= np.concatenate((molecule_pairs.indexes_tani, indexes_np), axis=0))
        
        return new_molecule_pairs
    
    @staticmethod
    def train_val_test_split_bms(spectrums, val_split=0.1, test_split=0.1):

        # get the percentage of training data
        train_split= 1- val_split-test_split
        # get the murcko scaffold
        bms = [s.murcko_scaffold for s in spectrums]
        
        # count the unique elements
        unique_values, counts = np.unique(bms, return_counts=True)
        
        # remove the appearence of not identified bms
        unique_values = unique_values[unique_values != '']
  

        # randomize
        random.shuffle(unique_values)
        
        # get indexes
        train_index= int((train_split)*(len(unique_values)))
        val_index= train_index + int(val_split*(len(unique_values)))
        
        # get elements
        train_bms= unique_values[0:train_index]
        val_bms = unique_values[train_index:val_index]
        test_bms=unique_values[val_index:]
        
        # get data
        spectrums_train = [s for s in spectrums if s.murcko_scaffold in train_bms]
        spectrums_val = [s for s in spectrums if s.murcko_scaffold in val_bms]
        spectrums_test = [s for s in spectrums if s.murcko_scaffold in test_bms]
        return spectrums_train, spectrums_val, spectrums_test


    @staticmethod
    def  get_combination_indexes(num_samples, combination_length=2):
        # Define the number of elements in each combination (e.g., 2 for pairs of indexes)
        return list(combinations(range(num_samples), combination_length))


    def generate_random_combinations(num_samples, num_combinations):
        all_indices = list(range(num_samples))
        
        for _ in range(num_combinations):
            random_indices = random.sample(all_indices, 2)  # Generate random combination of 2 indices
            yield random_indices

    
    @staticmethod
    def compute_all_fingerprints(all_spectrums):
        fingerprints = []

        #mols = [Chem.MolFromSmiles(s.params['smiles']) if (s.params['smiles'] != '' and s.params['smiles'] != 'N/A') else None
        #        for s in all_spectrums ]
        #fingerprints = [Chem.RDKFingerprint(m) if m is not None else None for m in mols ]

        for i in (range(0, len(all_spectrums))):
            fp = Tanimoto.compute_fingerprint(all_spectrums[i].params['smiles'])
            fingerprints.append(fp)
        return fingerprints

    @staticmethod
    def compute_number_of_pairs(all_spectrums, max_combinations=1000000, limit_low_tanimoto=True, 
                                     max_low_pairs=0.3, use_tqdm=True, max_mass_diff=None, #maximum number of elements in which we stop adding new items
                                     num_workers = 15):
        
        print('Starting computation of number of pairs')
        print(datetime.now())
        # order the spectrums by mass
        all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)
        
        # get mz
        total_mz = np.array([s.precursor_mz for s in all_spectrums])

        #indexes=[]
        indexes_np = np.zeros((max_combinations, 3 ))
        counter_indexes=0
        # Iterate through the list to form pairsi


        # Compute all the fingerprints:
        #print('Compute all the fingerprints')
        #fingerprints = TrainUtils.compute_all_fingerprints(all_spectrums)

        # get random indexes for the first part of the pair
        number_of_pairs=0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            num_workers = executor._max_workers  # Accessing the internal attribute
            print(f"Number of workers: {num_workers}")
            
            for i in tqdm(range(0,len(all_spectrums))):

                diff_total = (total_mz - (all_spectrums[i].precursor_mz+ max_mass_diff))
                max_mz_index = np.where(diff_total > 0)[0] # get list
                max_mz_index = max_mz_index[0] if len(max_mz_index) > 0 else len(all_spectrums)-1
                
                # get the other index

                #for j in range(i, max_mz_index):
                    #number_of_pairs = number_of_pairs+1
                number_of_pairs = number_of_pairs + (max_mz_index - i)
        return number_of_pairs
    
    @staticmethod
    def compute_all_tanimoto_results(all_spectrums, max_combinations=1000000, limit_low_tanimoto=True, 
                                     max_low_pairs=0.5, use_tqdm=True, 
                                     max_mass_diff=None, #maximum number of elements in which we stop adding new items
                                     min_mass_diff=0,
                                     num_workers = 15,
                                     MIN_SIM=0.8,
                                     MAX_SIM=1):

        print('Starting computation of molecule pairs')
        print(datetime.now())
        # order the spectrums by mass
        all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)
        
        # get mz
        total_mz = np.array([s.precursor_mz for s in all_spectrums])

        #indexes=[]
        indexes_np = np.zeros((max_combinations, 3 ))
        counter_indexes=0
        # Iterate through the list to form pairsi

        print('Computing all the tanimoto results')
        if use_tqdm:
            # Initialize tqdm with the total number of iterations
            progress_bar = tqdm(total=max_combinations, desc="Processing")

        # Compute all the fingerprints:
        print('Compute all the fingerprints')
        fingerprints = TrainUtils.compute_all_fingerprints(all_spectrums)

        # get random indexes for the first part of the pair
        #random_i_np = np.random.randint(0, len(all_spectrums)-2, max_combinations)

        print(f"Number of workers: {num_workers}")
        counter_indexes = 0
        while ( counter_indexes <(max_combinations)):

            i = np.random.randint(0, len(all_spectrums)-2) 
            diff_total_max = (total_mz - (all_spectrums[i].precursor_mz+ max_mass_diff))
            diff_total_min = (total_mz - (all_spectrums[i].precursor_mz+ min_mass_diff))
            min_mz_index=  np.where((diff_total_min > 0))[0] 
            max_mz_index = np.where((diff_total_max > 0))[0] # get list

            min_mz_index = min_mz_index[0] if len(min_mz_index) > 0 else 0
            max_mz_index = max_mz_index[0] if len(max_mz_index) > 0 else len(all_spectrums)-1
            
            # get the other index
            j = random.randint(min_mz_index, max_mz_index)

            # Submit the task to the executor
            tani = Tanimoto.compute_tanimoto(
                                            fingerprints[i],
                                            fingerprints[j],)
 
            if tani is not None:
                #if tani>MIN_SIM and tani<MAX_SIM:
                if (counter_indexes < max_low_pairs*max_combinations) or (tani>0.5):   
                    
                    indexes_np[counter_indexes,0]=i
                    indexes_np[counter_indexes,1]=j
                    indexes_np[counter_indexes,2] = tani
                    counter_indexes= counter_indexes+1
                    if use_tqdm:
                            progress_bar.update(1)

        
        print(f'Number of effective pairs retrieved: {counter_indexes} ')
        #molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes_np)
    
        print(datetime.now())
        return molecular_pair_set

    @staticmethod
    def compute_all_tanimoto_results_no_sequential(all_spectrums, max_combinations=1000000, limit_low_tanimoto=True, 
                                     max_low_pairs=0.3, use_tqdm=True, 
                                     max_mass_diff=None, #maximum number of elements in which we stop adding new items
                                     min_mass_diff=0,
                                     num_workers = 15):

        print('Starting computation of molecule pairs')
        print(datetime.now())
        # order the spectrums by mass
        all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)
        
        # get mz
        total_mz = np.array([s.precursor_mz for s in all_spectrums])

        #indexes=[]
        indexes_np = np.zeros((max_combinations, 3 ))
        counter_indexes=0
        # Iterate through the list to form pairsi

        print('Computing all the tanimoto results')
        if use_tqdm:
            # Initialize tqdm with the total number of iterations
            progress_bar = tqdm(total=max_combinations, desc="Processing")

        futures=[]
        

        # Compute all the fingerprints:
        print('Compute all the fingerprints')
        fingerprints = TrainUtils.compute_all_fingerprints(all_spectrums)

        # get random indexes for the first part of the pair
        random_i_np = np.random.randint(0, len(all_spectrums)-2, max_combinations)
        list_index_i = np.zeros(max_combinations)
        list_index_j = np.zeros(max_combinations)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            num_workers = executor._max_workers  # Accessing the internal attribute
            print(f"Number of workers: {num_workers}")
            
            for counter_indexes in range(max_combinations):

                i = random_i_np[counter_indexes]       
                diff_total_max = (total_mz - (all_spectrums[i].precursor_mz+ max_mass_diff))
                diff_total_min = (total_mz - (all_spectrums[i].precursor_mz+ min_mass_diff))
                min_mz_index=  np.where((diff_total_min > 0))[0] 
                max_mz_index = np.where((diff_total_max > 0))[0] # get list

                min_mz_index = min_mz_index[0] if len(min_mz_index) > 0 else 0
                max_mz_index = max_mz_index[0] if len(max_mz_index) > 0 else len(all_spectrums)-1
                
                # get the other index
                j = random.randint(min_mz_index, max_mz_index)

                # Submit the task to the executor
                futures.append(executor.submit(Tanimoto.compute_tanimoto,
                                               fingerprints[i],
                                               fingerprints[j],))
                list_index_i[counter_indexes]=i
                list_index_j[counter_indexes]=j
                if use_tqdm:
                                    progress_bar.update(1)

        
        concurrent.futures.wait(futures)
        # Retrieve the results from futures
        counter_indexes=0
        for future, i, j in zip(futures, list_index_i, list_index_j):
                tani = future.result()
                if tani is not None:
                    indexes_np[counter_indexes, 0] = i
                    indexes_np[counter_indexes, 1] = j
                    indexes_np[counter_indexes, 2] = tani
                    counter_indexes = counter_indexes +1

        indexes_np = indexes_np[0:counter_indexes]
        print(f'Number of effective pairs retrieved: {counter_indexes} ')
        #molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes_np)
        '''
        # create dataset
        print('Creating molecule pairs')
        molecule_pairs=[]

        if use_tqdm:
            iterator_extraction = tqdm(indexes)
        else:
            iterator_extraction = indexes
        for i, j, tani in iterator_extraction:
            molecule_pair = MoleculePair(
                        vector_0=None,
                        vector_1=None,
                        smiles_0=all_spectrums[i].smiles,
                        smiles_1=all_spectrums[j].smiles,
                        similarity=tani,
                        global_feats_0=TrainUtils.get_global_variables(all_spectrums[i]),
                        global_feats_1=TrainUtils.get_global_variables(all_spectrums[j]),
                        index_in_spectrum_0=i, #index in the spectrum list used as input
                        index_in_spectrum_1=j,
                        spectrum_object_0= all_spectrums[i],
                         spectrum_object_1 = all_spectrums[j],
                         params_0= all_spectrums[i].params,
                         params_1= all_spectrums[j].params,)
            molecule_pairs.append(molecule_pair)
        '''
        print('Ending computation of molecule pairs')
        print(datetime.now())
        return molecular_pair_set
    

    
    #@staticmethod
    #def get_min_bin(molecule_pairs, number_bins):
    #    similarities = [m.similarity for m in molecule_pairs]
    #    hist, _ = np.histogram(similarities, bins=number_bins)
    #    return np.min(hist)
    
    @staticmethod
    def divide_data_into_bins(molecule_pairs, number_bins, 
                              bin_sim_1=True, #if you want to try sim=1 as a different bin
                              ):
         # Initialize lists to store values for each bin
        binned_molecule_pairs = []

        # Group the values into the corresponding bins, adding one for sim=1
        if bin_sim_1:
            number_bins_effective = number_bins+1
        else:
            number_bins_effective = number_bins

        for p in range(number_bins_effective):
            low = p*(1/number_bins_effective)

            if bin_sim_1:
                high =  (p+1)*(1/number_bins_effective)
            else:
                 if (p == (number_bins_effective-1)):
                      high= 1+ 0.1
                 else:
                      high =  (p+1)*(1/number_bins_effective)
            
            temp_molecule_pairs = [m for m in molecule_pairs if ((m.similarity>=low) and (m.similarity<high))]


            binned_molecule_pairs.append(temp_molecule_pairs)
        

        # get minimum bin size
        min_bin = min([len(b) for b in binned_molecule_pairs])
        return binned_molecule_pairs, min_bin

    @staticmethod
    def uniformise(molecule_pairs, number_bins=3, return_binned_list=False, 
                   bin_sim_1=True, #if you want to treat sim=1 as another bin
                   ):   
        '''
        get a uniform distribution of labels between 0 and 1
        ''' 
        #min_bin = TrainUtils.get_min_bin(molecule_pairs, number_bins)
        binned_molecule_pairs, min_bin = TrainUtils.divide_data_into_bins(molecule_pairs, number_bins, bin_sim_1=bin_sim_1)
        uniform_molecule_pairs= []

        for target_molecule_pairs in binned_molecule_pairs:
            # select some random samples
            #print('*')
            #print(len(target_molecule_pairs))
            #print(min_bin)
            sampled_molecule_pairs = random.sample(target_molecule_pairs, min_bin)
            # add to the final list
            uniform_molecule_pairs = uniform_molecule_pairs + sampled_molecule_pairs

        # insert spectrum vectors
        uniform_molecule_pairs = TrainUtils.insert_spectrum_vector_into_molecule_pairs(uniform_molecule_pairs)
        if return_binned_list:
            return uniform_molecule_pairs, binned_molecule_pairs
        else:
            return uniform_molecule_pairs
    
    @staticmethod
    def insert_spectrum_vector_into_molecule_pairs(molecule_pairs):
        pp = Preprocessor()
        for m in molecule_pairs:
            m.vector_0 = pp.return_vector_and_preprocess(m.spectrum_object_0)
            m.vector_1 = pp.return_vector_and_preprocess(m.spectrum_object_1)
        return molecule_pairs


    @staticmethod
    def get_data_from_indexes(spectrums, indexes):
        return ([(spectrums[p[0]].spectrum_vector,   TrainUtils.get_global_variables(spectrums[p[0]]),
                          spectrums[p[1]].spectrum_vector, TrainUtils.get_global_variables(spectrums[p[1]]),) for p in indexes])
        


    @staticmethod
    def get_global_variables(spectrum):
        '''
        get global variables from a spectrum such as precursor mass
        '''
        list_global_variables = [spectrum.precursor_mz, spectrum.precursor_charge]
        return np.array(list_global_variables)
