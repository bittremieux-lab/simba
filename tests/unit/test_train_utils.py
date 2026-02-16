"""Tests for simba/train_utils.py"""

import numpy as np
import pandas as pd
import pytest

from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.training.train_utils import TrainUtils


@pytest.fixture
def sample_spectra(create_test_spectrum):
    return [
        create_test_spectrum(identifier="spec1", smiles="CCO", bms="scaffold1"),
        create_test_spectrum(identifier="spec2", smiles="CCCO", bms="scaffold1"),
        create_test_spectrum(identifier="spec3", smiles="CCCCO", bms="scaffold2"),
        create_test_spectrum(identifier="spec4", smiles="CCCCCO", bms="scaffold3"),
        create_test_spectrum(identifier="spec5", smiles="CCCCCCO", bms="scaffold3"),
        create_test_spectrum(identifier="spec6", smiles="CCCCCCCO", bms="scaffold4"),
        create_test_spectrum(identifier="spec7", smiles="CCCCCCCCO", bms="scaffold4"),
        create_test_spectrum(identifier="spec8", smiles="CCCCCCCCCO", bms="scaffold5"),
    ]


@pytest.fixture
def sample_molecule_pairs(sample_spectra):
    pair_distances = np.array(
        [
            [0, 1, 0.9],
            [0, 2, 0.85],
            [0, 3, 0.8],
            [1, 2, 0.75],
            [1, 3, 0.7],
            [2, 3, 0.65],
            [3, 4, 0.6],
            [4, 5, 0.5],
            [5, 6, 0.4],
            [6, 7, 0.3],
            [0, 4, 0.2],
            [1, 5, 0.15],
            [2, 6, 0.1],
            [3, 7, 0.05],
        ]
    )
    df_smiles = pd.DataFrame(
        {
            "indexes": [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
            ]
        },
        index=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    return MoleculePairsOpt(
        unique_spectra=sample_spectra,
        pair_distances=pair_distances,
        original_spectra=sample_spectra,
        df_smiles=df_smiles,
    )


class TestTrainUtils:
    def test_compute_unique_combinations(self, sample_molecule_pairs):
        result = TrainUtils.compute_unique_combinations(
            sample_molecule_pairs, high_sim=1
        )

        assert isinstance(result, MoleculePairsOpt)
        assert len(result.spectra) == len(sample_molecule_pairs.spectra)
        assert result.pair_distances.shape[1] == 3

    def test_train_val_test_split_bms(self, sample_spectra):
        train, val, test = TrainUtils.train_val_test_split_bms(
            sample_spectra, val_split=0.25, test_split=0.25, seed=42
        )

        assert len(train) + len(val) + len(test) == len(sample_spectra)
        assert len(val) > 0
        assert len(test) > 0

    def test_get_combination_indexes(self):
        indexes = TrainUtils.get_combination_indexes(
            num_samples=5, combination_length=2
        )

        assert isinstance(indexes, list)
        assert all(len(idx) == 2 for idx in indexes)
        expected_combinations = 5 * (5 - 1) // 2
        assert len(indexes) == expected_combinations

    def test_generate_random_combinations(self):
        combinations = list(
            TrainUtils.generate_random_combinations(num_samples=10, num_combinations=5)
        )

        assert len(combinations) == 5
        assert all(len(c) == 2 for c in combinations)
        assert all(c[0] != c[1] for c in combinations)

    def test_get_unique_spectra(self, sample_spectra):
        unique_spectra, mapping = TrainUtils.get_unique_spectra(sample_spectra)

        assert len(unique_spectra) <= len(sample_spectra)
        assert len(mapping) == len(unique_spectra)

    def test_get_unique_spectra_with_duplicates(self, create_test_spectrum):
        # Test with duplicate SMILES to verify deduplication
        spectra_with_duplicates = [
            create_test_spectrum(identifier="spec1", smiles="CCO", bms="scaffold1"),
            create_test_spectrum(identifier="spec2", smiles="CCO", bms="scaffold1"),
            create_test_spectrum(identifier="spec3", smiles="CCCO", bms="scaffold2"),
            create_test_spectrum(identifier="spec4", smiles="CCCO", bms="scaffold2"),
            create_test_spectrum(identifier="spec5", smiles="CCCCO", bms="scaffold3"),
        ]

        unique_spectra, mapping = TrainUtils.get_unique_spectra(spectra_with_duplicates)

        # Should have 3 unique SMILES: CCO, CCCO, CCCCO
        assert len(unique_spectra) == 3
        # Mapping should reflect the reduced set
        assert len(mapping) == 3

    def test_uniformise(self, sample_molecule_pairs):
        result = TrainUtils.uniformise(
            sample_molecule_pairs,
            number_bins=2,
            return_binned_list=False,
            bin_sim_1=False,
            seed=42,
        )

        assert isinstance(result, MoleculePairsOpt)
        assert len(result.pair_distances) > 0

    def test_uniformise_with_ordinal(self, sample_molecule_pairs):
        result = TrainUtils.uniformise(
            sample_molecule_pairs,
            number_bins=2,
            ordinal_classification=True,
            bin_sim_1=False,
            seed=42,
        )

        assert isinstance(result, MoleculePairsOpt)

    def test_divide_data_into_bins(self, sample_molecule_pairs):
        binned_pairs, min_bin = TrainUtils.divide_data_into_bins(
            sample_molecule_pairs, number_bins=2, bin_sim_1=False
        )

        assert len(binned_pairs) == 2
        assert isinstance(min_bin, int)
        assert min_bin >= 0
        assert all(isinstance(bp, MoleculePairsOpt) for bp in binned_pairs)

    def test_divide_data_into_bins_with_sim_1(self, sample_molecule_pairs):
        binned_pairs, min_bin = TrainUtils.divide_data_into_bins(
            sample_molecule_pairs, number_bins=2, bin_sim_1=True
        )

        assert len(binned_pairs) == 3
        assert isinstance(min_bin, int)

    def test_divide_data_into_bins_categories(self, sample_molecule_pairs):
        binned_pairs, min_bin = TrainUtils.divide_data_into_bins_categories(
            sample_molecule_pairs, number_bins=2, bin_sim_1=False
        )

        assert len(binned_pairs) == 2
        assert isinstance(min_bin, int)
        assert min_bin >= 0

    def test_uniformise_with_return_binned_list(self, sample_molecule_pairs):
        result, binned_list = TrainUtils.uniformise(
            sample_molecule_pairs,
            number_bins=2,
            return_binned_list=True,
            bin_sim_1=False,
            seed=42,
        )

        assert isinstance(result, MoleculePairsOpt)
        assert isinstance(binned_list, list)
        assert len(binned_list) == 2
        assert all(isinstance(item, MoleculePairsOpt) for item in binned_list)

    def test_compute_all_fingerprints(self, sample_spectra):
        fingerprints = TrainUtils.compute_all_fingerprints(sample_spectra)

        assert len(fingerprints) == len(sample_spectra)
        assert all(fp is not None for fp in fingerprints)

    def test_train_val_test_split_bms_no_val_test(self, sample_spectra):
        train, val, test = TrainUtils.train_val_test_split_bms(
            sample_spectra, val_split=0.0, test_split=0.0, seed=42
        )

        assert len(train) == len(sample_spectra)
        assert len(val) == 0
        assert len(test) == 0
