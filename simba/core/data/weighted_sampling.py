import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from simba.utils.binning import round_to_ordinal


class SimilarityWeightSampler:
    """
    Class for computing weights for weighted random sampling based on binned similarity values.
    """

    @staticmethod
    def compute_weights(binned_list):
        freq = np.array([len(r) for r in binned_list])

        weights = np.zeros(len(binned_list))
        for index, f in enumerate(freq):
            if f != 0:
                weights[index] = np.sum(freq) / f
            else:
                weights[index] = 0

        # normalize the weights
        weights = weights / np.sum(weights)

        # compute the range of similarity values for each bin
        bin_size = 1 / len(binned_list)
        range_weights = np.arange(0, len(binned_list)) * bin_size
        return weights, range_weights

    @staticmethod
    def compute_weights_categories(binned_list):
        freq = np.array([len(r) for r in binned_list])

        weights = np.sum(freq) / freq
        weights = weights / np.sum(weights)
        bin_size = 1 / (len(binned_list) - 1)
        range_weights = np.arange(0, len(binned_list)) * bin_size
        return weights, range_weights

    @staticmethod
    def compute_sample_weights(
        molecule_pairs,
        weights,
        use_molecule_pair_object=True,
        targets=None,
        bining_sim1=False,
        normalize=True,
    ):
        # get similarities
        if use_molecule_pair_object:
            sim = molecule_pairs.pair_distances[:, 2]
        else:
            sim = targets

        # Calculate the index using vectorized operations
        effective_range = len(weights) - 1 if bining_sim1 else len(weights)

        indices = np.floor(sim * (effective_range)).astype(int)

        # make sure the indexes are not below 0
        indices[indices < 0] = 0

        if not (bining_sim1):
            indices[indices == effective_range] = len(weights) - 1

        # Map the indices to weights and normalize

        weights_sample = weights[indices]
        if normalize:
            weights_sample /= weights_sample.sum()

        return weights_sample

    @staticmethod
    def compute_sample_weights_categories(molecule_pairs, weights):
        # get similarities
        sim = molecule_pairs.pair_distances[:, 2]

        # Calculate the index using vectorized operations
        indices = round_to_ordinal(sim * (len(weights) - 1)).astype(int)
        indices[indices == len(weights)] = len(weights) - 1

        # Map the indices to weights and normalize
        weights_sample = weights[indices]
        weights_sample /= weights_sample.sum()

        return weights_sample


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement,
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
