import numpy as np
import math


class WeightSampling:
    """
    functions to execute weighted sampling for tackling unbalance in similarities
    """

    @staticmethod
    def compute_weights(binned_list):
        freq = np.array([len(r) for r in binned_list])
        weights = np.sum(freq) / freq

        # remove 1 from using the last bin for sim=1

        index_half = int((len(binned_list) - 1) / 2)

        # deprecated
        # for the ranges that are lower than 0.5, put half the weight on the highest range
        #weights[0:index_half] = 0.5 * weights[0:index_half]

        #for the ranges that are lower than 0.5 treat them as an only range
        sum_weights_low_range=sum(weights[0:index_half])
        weight_low_range= sum_weights_low_range/(len(weights[0:index_half]))
        weights[0:index_half] = weight_low_range

        # normalize the weights
        weights = weights / np.sum(weights)

        # the last bin corresponds to sim=1. So if there are 6 bins, actually there are 5 bins between 0 and 1
        bin_size = 1 / (len(binned_list) - 1)
        range_weights = np.arange(0, len(binned_list)) * bin_size
        return weights, range_weights

    @staticmethod
    def compute_sample_weights(molecule_pairs, weights):
        sim = [m.similarity for m in molecule_pairs]
        index = [math.floor(s * (len(weights) - 1)) for s in sim]
        weights_sample = np.array([weights[ind] for ind in index])
        weights_sample = weights_sample / (sum(weights_sample))
        return weights_sample
