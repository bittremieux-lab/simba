from itertools import combinations
from tqdm import tqdm
from src.tanimoto import Tanimoto
import numpy as np
import random
from src.molecule_pair import MoleculePair
from  src.preprocessor import Preprocessor

class TrainUtils:

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
    def compute_all_tanimoto_results(all_spectrums, max_combinations=1000000, limit_low_tanimoto=True):
        num_samples= len(all_spectrums)
        molecule_pairs=[]
        all_indices = list(range(num_samples))
        index_used = [random.sample(all_indices, 2) for _ in range(max_combinations)] #combinations

        for i, j in tqdm(index_used):

            tani = Tanimoto.compute_tanimoto(all_spectrums[i].params['smiles'],  all_spectrums[j].params['smiles'])
            if tani is not None:
                # in the case we want to reduce the number of low tanimoto values, we can add only if the tanimoto value is high or the number of samples is small 
                if not(limit_low_tanimoto) or (len(molecule_pairs)<100000) or (tani>0.3):  
                    molecule_pair = MoleculePair( vector_0=None, 
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

        return molecule_pairs
    

    
    @staticmethod
    def get_min_bin(molecule_pairs, number_bins):
        similarities = [m.similarity for m in molecule_pairs]
        hist, _ = np.histogram(similarities, bins=number_bins)
        return np.min(hist)
    
    @staticmethod
    def divide_data_into_bins(molecule_pairs, number_bins):
         # Initialize lists to store values for each bin
        binned_molecule_pairs = []

        # Group the values into the corresponding bins
        for p in range(number_bins):
            low = p*(1/number_bins)
            if p == (number_bins-1): # if it is the last bin takes everything including 1
                high= 1+ 0.1
            else:
                high =  (p+1)*(1/number_bins)
            
            temp_molecule_pairs = [m for m in molecule_pairs if ((m.similarity>=low) and (m.similarity<high))]


            binned_molecule_pairs.append(temp_molecule_pairs)
        return binned_molecule_pairs

    @staticmethod
    def uniformise(molecule_pairs, number_bins=3):   
        '''
        get a uniform distribution of labels between 0 and 1
        ''' 
        min_bin = TrainUtils.get_min_bin(molecule_pairs, number_bins)
        binned_molecule_pairs = TrainUtils.divide_data_into_bins(molecule_pairs, number_bins)
        uniform_molecule_pairs= []

        for target_molecule_pairs in binned_molecule_pairs:
            # select some random samples
            sampled_molecule_pairs = random.sample(target_molecule_pairs, min_bin)
            # add to the final list
            uniform_molecule_pairs = uniform_molecule_pairs + sampled_molecule_pairs

        # insert spectrum vectors
        uniform_molecule_pairs = TrainUtils.insert_spectrum_vector_into_molecule_pairs(uniform_molecule_pairs)
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