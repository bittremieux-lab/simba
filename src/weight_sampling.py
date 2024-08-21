import numpy as np
import math
from src.ordinal_classification.ordinal_classification import OrdinalClassification

class WeightSampling:
    """
    functions to execute weighted sampling for tackling unbalance in similarities
    """

    @staticmethod
    def compute_weights(binned_list):
        freq = np.array([len(r) for r in binned_list])
       
        # remove 1 from using the last bin for sim=1
        #index_half = int((len(binned_list)) / 2)

        # for the ranges that are lower than 0.5 treat them as an only range
        # sum_freq_low_range=sum(freq[0:index_half])
        # freq_low_range= sum_freq_low_range/(len(freq[0:index_half]))

        #freq[0:index_half] = freq_low_range
        weights=np.zeros(len(binned_list))
        for index,f in enumerate(freq):
            if f != 0:
                weights[index] = np.sum(freq)/f
            else:
                weights[index]=0
        

        # deprecated
        # for the ranges that are lower than 0.5, put half the weight on the highest range
        # weights[0:index_half] = 0.5 * weights[0:index_half]

        # for the weights below 0.5 increase the wiehgt to reduce false positives:
        #weights[0:index_half] = 2 * weights[0:index_half]

        # normalize the weights
        weights = weights / np.sum(weights)
        # weights= [1 for w in weights]

        # the last bin corresponds to sim=1. So if there are 6 bins, actually there are 5 bins between 0 and 1
        #bin_size = 1 / (len(binned_list) - 1)

        # currently, we do not bin sim=1
        bin_size= 1/len(binned_list)
        range_weights = np.arange(0, len(binned_list)) * bin_size
        return weights, range_weights

    @staticmethod
    def compute_weights_categories(binned_list):
        freq = np.array([len(r) for r in binned_list])
        weights = np.sum(freq) / freq
        weights = weights / np.sum(weights)
        bin_size= 1/(len(binned_list)-1)
        range_weights = np.arange(0, len(binned_list)) * bin_size
        return weights, range_weights

    #@staticmethod
    #def compute_sample_weights(molecule_pairs, weights):

        # get similarities
    #    sim = molecule_pairs.indexes_tani[:, 2]
    #    # sim = [m.similarity for m in molecule_pairs]
    #    #index = [math.floor(s * (len(weights) - 1)) for s in sim]
    #    index = [math.floor(s * (len(weights))) if s != 1 else (len(weights)-1) for s in sim  ]
    #    weights_sample = np.array([weights[ind] for ind in index])
    #    weights_sample = weights_sample / (sum(weights_sample))
    #    return weights_sample

    @staticmethod
    def compute_sample_weights(molecule_pairs, weights, use_molecule_pair_object=True, targets=None, 
                        bining_sim1=False,
                        normalize=True):
        # get similarities
        if use_molecule_pair_object:
            sim = molecule_pairs.indexes_tani[:, 2]
        else:
            sim = targets
        
        # Calculate the index using vectorized operations
        if bining_sim1:
            effective_range= len(weights) - 1
        else:
            effective_range = len(weights)

        #print(f'sim: {sim}')
        #print(f'effective range: {effective_range}')
        indices = np.floor(sim * (effective_range)).astype(int)

        if not(bining_sim1):
            indices[indices == effective_range] = len(weights) - 1
        
        # Map the indices to weights and normalize

        #print(f'Weights: {weights}')
        #print(f'indices: {indices}')
        #print(f'{weights.shape}')
        #print(f'{indices.shape}')
        weights_sample = weights[indices]
        if normalize:
            weights_sample /= weights_sample.sum()
        
        return weights_sample

    
    
    @staticmethod
    def compute_sample_weights_categories(molecule_pairs, weights):
        # get similarities
        sim = molecule_pairs.indexes_tani[:, 2]
        
        # Calculate the index using vectorized operations
        #indices = np.ceil(sim * (len(weights)-1)).astype(int)
        indices = OrdinalClassification.custom_random(sim * (len(weights)-1)).astype(int)
        indices[indices == len(weights)] = len(weights) - 1
        
        # Map the indices to weights and normalize
        weights_sample = weights[indices]
        weights_sample /= weights_sample.sum()
        
        return weights_sample