"""Integration tests for SIMBA molecular networking."""

import numpy as np
import pytest

from simba.core.models.simba_model import Simba
from simba.workflows.molecular_networking import (
    _build_scores,
    mces_to_similarity,
)
from simba.workflows.utils import load_spectra


pytestmark = pytest.mark.integration


class TestMolecularNetworkingWorkflow:
    def test_all_vs_all_prediction_shape(
        self, sample_mgf_casmi, mock_model, hydra_config
    ):
        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        assert len(spectra) >= 2

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )
        sim_ed, sim_mces = simba.predict(spectra, spectra)

        n = len(spectra)
        assert sim_ed.shape == (n, n)
        assert sim_mces.shape == (n, n)

    def test_embedding_cached_for_symmetric_call(
        self, sample_mgf_casmi, mock_model, hydra_config
    ):
        """Second predict call with the same spectra should hit the cache."""
        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )

        simba.predict(spectra, spectra)
        initial_cache_size = len(simba._embedding_cache)

        simba.predict(spectra, spectra)
        assert len(simba._embedding_cache) == initial_cache_size

    def test_similarity_matrix_in_unit_interval(
        self, sample_mgf_casmi, mock_model, hydra_config
    ):
        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )
        _, sim_mces = simba.predict(spectra, spectra)

        similarity = mces_to_similarity(sim_mces)

        assert np.all(similarity >= 0.0)
        assert np.all(similarity <= 1.0)

    def test_matchms_scores_wrapper(self, sample_mgf_casmi, mock_model, hydra_config):
        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        n = len(spectra)
        sim = np.ones((n, n)) * 0.8
        scores, nodes = _build_scores(spectra, sim, "simba_similarity")

        assert scores.n_rows == n
        assert scores.n_cols == n
        assert len(nodes) == n
        assert all(node.get("spectrum_id") is not None for node in nodes)

    def test_network_has_nodes_and_edges(
        self, sample_mgf_casmi, mock_model, hydra_config
    ):
        from matchms.networking import SimilarityNetwork

        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )
        _, sim_mces = simba.predict(spectra, spectra)

        similarity = mces_to_similarity(sim_mces)
        scores, _ = _build_scores(spectra, similarity, "simba_similarity")

        network = SimilarityNetwork(
            identifier_key="spectrum_id",
            top_n=len(spectra),
            max_links=len(spectra) - 1,
            score_cutoff=0.0,
        )
        network.create_network(scores, score_name="simba_similarity")

        assert network.graph.number_of_nodes() == len(spectra)
        assert network.graph.number_of_edges() >= 0

    def test_network_export_graphml(
        self, sample_mgf_casmi, mock_model, hydra_config, temp_dir
    ):
        from matchms.networking import SimilarityNetwork

        spectra = load_spectra(
            sample_mgf_casmi, hydra_config, min_peaks=5, n_samples=100
        )
        n = len(spectra)
        sim = np.ones((n, n)) * 0.9
        np.fill_diagonal(sim, 1.0)

        scores, _ = _build_scores(spectra, sim, "simba_similarity")
        network = SimilarityNetwork(
            identifier_key="spectrum_id",
            top_n=n,
            max_links=n - 1,
            score_cutoff=0.5,
        )
        network.create_network(scores, score_name="simba_similarity")

        output_file = str(temp_dir / "network.graphml")
        network.export_to_file(output_file, graph_format="graphml")

        assert (temp_dir / "network.graphml").exists()
