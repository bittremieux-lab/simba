"""Unit tests for SIMBA molecular networking."""

import numpy as np
import pytest

from simba.workflows.molecular_networking import (
    _build_scores,
    mces_to_similarity,
)


class TestMcesToSimilarity:
    def test_fixed_normalization_zero_distance(self):
        mces = np.zeros((3, 3))
        result = mces_to_similarity(mces)
        np.testing.assert_array_equal(result, np.ones((3, 3)))

    def test_fixed_normalization_max_distance(self):
        mces = np.full((2, 2), 20.0)
        result = mces_to_similarity(mces, mces_max=20.0)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))

    def test_fixed_normalization_clamps_below_zero(self):
        mces = np.array([[25.0]])
        result = mces_to_similarity(mces, mces_max=20.0)
        assert result[0, 0] == 0.0

    def test_fixed_normalization_midpoint(self):
        mces = np.array([[10.0]])
        result = mces_to_similarity(mces, mces_max=20.0)
        assert result[0, 0] == pytest.approx(0.5)

    def test_fixed_normalization_custom_max(self):
        mces = np.array([[5.0]])
        result = mces_to_similarity(mces, mces_max=10.0)
        assert result[0, 0] == pytest.approx(0.5)

    def test_output_always_in_unit_interval(self):
        rng = np.random.default_rng(42)
        mces = rng.uniform(0, 30, (10, 10))
        result = mces_to_similarity(mces)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetric_input_gives_symmetric_output(self):
        rng = np.random.default_rng(0)
        m = rng.uniform(0, 20, (5, 5))
        sym = (m + m.T) / 2
        result = mces_to_similarity(sym)
        np.testing.assert_array_almost_equal(result, result.T)


class TestBuildScores:
    def test_build_scores_shape(self, create_test_spectrum):
        spectra = [
            create_test_spectrum(mgf_index=i, precursor_mz=float(100 + i * 10))
            for i in range(4)
        ]
        sim = np.eye(4)
        scores, nodes = _build_scores(spectra, sim, "simba_similarity")

        assert len(nodes) == 4
        assert scores.n_rows == 4
        assert scores.n_cols == 4

    def test_node_ids_are_mgf_position(self, create_test_spectrum):
        spectra = [
            create_test_spectrum(mgf_index=i, precursor_mz=float(100 + i * 10))
            for i in range(3)
        ]
        scores, nodes = _build_scores(spectra, np.eye(3), "simba_similarity")
        assert [n.get("spectrum_id") for n in nodes] == ["0", "1", "2"]

    def test_node_ids_use_mgf_index_not_list_position(self, create_test_spectrum):
        # Simulate 3 spectra surviving filtering from a larger MGF (indices 5, 7, 9)
        spectra = [
            create_test_spectrum(mgf_index=2 * i + 5, precursor_mz=float(100 + i * 10))
            for i in range(3)
        ]
        scores, nodes = _build_scores(spectra, np.eye(3), "simba_similarity")
        assert [n.get("spectrum_id") for n in nodes] == ["5", "7", "9"]

    def test_network_builds_from_scores(self, create_test_spectrum):
        from matchms.networking import SimilarityNetwork

        spectra = [
            create_test_spectrum(mgf_index=i, precursor_mz=float(100 + i * 10))
            for i in range(4)
        ]
        sim = np.ones((4, 4)) * 0.9
        np.fill_diagonal(sim, 1.0)
        scores, _ = _build_scores(spectra, sim, "simba_similarity")

        network = SimilarityNetwork(
            identifier_key="spectrum_id",
            top_n=4,
            max_links=3,
            score_cutoff=0.5,
        )
        network.create_network(scores, score_name="simba_similarity")
        assert network.graph.number_of_nodes() == 4
        assert network.graph.number_of_edges() > 0
