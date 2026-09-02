"""Tests for simba/core/data/weighted_sampling.py"""

import numpy as np
import torch

from simba.core.data.weighted_sampling import CustomWeightedRandomSampler


class TestCustomWeightedRandomSampler:
    def test_matches_expected_distribution(self):
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0])
        sampler = CustomWeightedRandomSampler(
            weights=weights, num_samples=200_000, replacement=True
        )

        draws = list(iter(sampler))
        fractions = np.bincount(draws, minlength=5) / len(draws)
        expected = (weights / weights.sum()).numpy()

        np.testing.assert_allclose(fractions, expected, atol=0.01)

    def test_draws_differ_across_epochs(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        sampler = CustomWeightedRandomSampler(
            weights=weights, num_samples=1000, replacement=True
        )

        draws_epoch_0 = list(iter(sampler))
        draws_epoch_1 = list(iter(sampler))

        assert draws_epoch_0 != draws_epoch_1
