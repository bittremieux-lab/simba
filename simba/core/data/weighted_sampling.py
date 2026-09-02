import numpy as np
from simba.utils.binning import round_to_ordinal
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


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


def _alias_setup(probs: np.ndarray):
    """Walker's alias method table construction (O(n), run once).

    Returns (prob, alias) such that each slot i draws outcome i with
    probability prob[i], else outcome alias[i].
    """
    n = len(probs)
    prob = np.zeros(n)
    alias = np.zeros(n, dtype=np.int64)
    scaled = probs * n
    small = list(np.where(scaled < 1)[0])
    large = list(np.where(scaled >= 1)[0])
    while small and large:
        small_idx = small.pop()
        large_idx = large.pop()
        prob[small_idx] = scaled[small_idx]
        alias[small_idx] = large_idx
        scaled[large_idx] -= 1 - scaled[small_idx]
        if scaled[large_idx] < 1:
            small.append(large_idx)
        else:
            large.append(large_idx)
    for remaining in small + large:
        prob[remaining] = 1
    return prob, alias


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.weights.numpy() / torch.sum(self.weights).numpy()
        self.alias_prob, self.alias_idx = _alias_setup(p)

    def __iter__(self):
        n = len(self.alias_prob)
        col = np.random.randint(0, n, size=self.num_samples)
        r = np.random.random_sample(self.num_samples)
        use_alias = r >= self.alias_prob[col]
        result = np.where(use_alias, self.alias_idx[col], col)
        return iter(torch.from_numpy(result).tolist())
