"""Integration tests for SIMBA analog discovery.

Tests cover analog discovery workflow including:
- Distance matrix computation between query and library spectra
- Combined ranking using MCES and edit distance metrics
- Top-k analog extraction
- Basic ranking computation

Based on notebooks/final_tutorials/run_analog_discovery.ipynb
"""

import numpy as np
import pytest

from simba.analog_discovery.simba_analog_discovery import AnalogDiscovery
from simba.core.models.simba_model import Simba
from simba.workflows.utils import load_spectra


pytestmark = pytest.mark.integration


class TestAnalogDiscovery:
    """Test analog discovery workflow."""

    def test_compute_distance_matrix(self, sample_mgf_casmi, mock_model, hydra_config):
        """Test computing distance matrices between query and library spectra."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )

        query_spectra = spectra[:1]
        library_spectra = spectra

        sim_ed, sim_mces = simba.predict(query_spectra, library_spectra)

        assert sim_ed.shape == (len(query_spectra), len(library_spectra))
        assert sim_mces.shape == (len(query_spectra), len(library_spectra))
        assert isinstance(sim_ed, np.ndarray)
        assert isinstance(sim_mces, np.ndarray)

    def test_ranking_combined_metrics(self, sample_mgf_casmi, mock_model, hydra_config):
        """Test ranking using combined MCES and edit distance metrics."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )

        query_spectra = spectra[:1]
        library_spectra = spectra

        sim_ed, sim_mces = simba.predict(query_spectra, library_spectra)
        ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)

        assert ranking.shape == (len(query_spectra), len(library_spectra))
        assert isinstance(ranking, np.ndarray)
        assert np.all(ranking >= 0) and np.all(ranking <= 1)

    def test_find_top_k_analogs(self, sample_mgf_casmi, mock_model, hydra_config):
        """Test extracting top-k analog matches."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )

        query_spectra = spectra[:1]
        library_spectra = spectra

        sim_ed, sim_mces = simba.predict(query_spectra, library_spectra)
        ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)

        k = 2
        top_k_indices = np.argsort(ranking[0])[-k:][::-1]

        assert len(top_k_indices) == k
        assert all(0 <= idx < len(library_spectra) for idx in top_k_indices)


class TestAnalogDiscoveryComponents:
    """Unit tests for analog discovery components."""

    def test_analog_discovery_import(self):
        """Test that AnalogDiscovery module can be imported."""
        assert AnalogDiscovery is not None

    def test_compute_ranking_basic(self):
        """Test basic ranking computation with synthetic data."""
        sim_mces = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
        sim_ed = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])

        ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)

        assert ranking.shape == sim_mces.shape
        assert isinstance(ranking, np.ndarray)
        assert np.all(ranking >= 0) and np.all(ranking <= 1)
