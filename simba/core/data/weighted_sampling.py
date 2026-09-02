import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


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
