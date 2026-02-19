"""Unit tests for ground truth computations."""

import numpy as np
import pytest

from simba.core.chemistry.similarity_metrics import MolecularSimilarityMetrics
from simba.workflows.utils import load_spectra


pytestmark = pytest.mark.unit


class TestGroundTruthTanimoto:
    """Test Tanimoto similarity ground truth computations."""

    def test_compute_tanimoto_identical_molecules(
        self, sample_mgf_casmi, hydra_config, mocker
    ):
        """Test Tanimoto computation for identical molecules returns 1.0."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=2,
            use_gnps_format=False,
        )

        assert len(spectra) >= 1

        result = MolecularSimilarityMetrics.compute_tanimoto([spectra[0]], [spectra[0]])

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_compute_tanimoto_different_molecules(self, sample_mgf_casmi, hydra_config):
        """Test Tanimoto computation for different molecules."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=10,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_tanimoto([spectra[0]], [spectra[1]])

        assert result.shape == (1, 1)
        assert 0.0 <= result[0, 0] <= 1.0
        assert isinstance(result, np.ndarray)


class TestGroundTruthEditDistance:
    """Test Edit Distance ground truth computations."""

    def test_compute_edit_distance_identical_molecules(
        self, sample_mgf_casmi, hydra_config
    ):
        """Test Edit Distance computation for identical molecules returns 0.0."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=2,
            use_gnps_format=False,
        )

        assert len(spectra) >= 1

        result = MolecularSimilarityMetrics.compute_edit_distance(
            [spectra[0]], [spectra[0]]
        )

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert isinstance(result, np.ndarray)

    def test_compute_edit_distance_different_molecules(
        self, sample_mgf_casmi, hydra_config
    ):
        """Test Edit Distance computation for different molecules."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=10,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_edit_distance(
            [spectra[0]], [spectra[1]]
        )

        assert result.shape == (1, 1)
        assert result[0, 0] >= 0.0
        assert isinstance(result, np.ndarray)

    def test_compute_edit_distance_max_value_default(
        self, sample_mgf_casmi, hydra_config
    ):
        """Test Edit Distance computation with default max_value=5."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=5,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_edit_distance(
            [spectra[0]], [spectra[1]], max_value=5
        )

        assert result.shape == (1, 1)
        assert result[0, 0] <= 5.0
        assert isinstance(result, np.ndarray)

    def test_compute_edit_distance_max_value_custom(
        self, sample_mgf_casmi, hydra_config
    ):
        """Test Edit Distance computation with custom max_value=10."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=5,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_edit_distance(
            [spectra[0]], [spectra[1]], max_value=10
        )

        assert result.shape == (1, 1)
        assert result[0, 0] <= 10.0
        assert isinstance(result, np.ndarray)

    def test_compute_edit_distance_multiple_pairs(self, sample_mgf_casmi, hydra_config):
        """Test Edit Distance computation for multiple molecule pairs."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=10,
            use_gnps_format=False,
        )

        assert len(spectra) >= 3

        result = MolecularSimilarityMetrics.compute_edit_distance(
            [spectra[0], spectra[1]], [spectra[1], spectra[2]]
        )

        assert result.shape == (2, 2)
        assert np.all(result >= 0.0)
        assert isinstance(result, np.ndarray)


class TestGroundTruthMCES:
    """Test MCES (Maximum Common Edge Subgraph) ground truth computations."""

    def test_compute_mces_identical_molecules(self, sample_mgf_casmi, hydra_config):
        """Test MCES computation for identical molecules returns 0.0."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=2,
            use_gnps_format=False,
        )

        assert len(spectra) >= 1

        result = MolecularSimilarityMetrics.compute_mces(
            [spectra[0]], [spectra[0]], threshold=20
        )

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert isinstance(result, np.ndarray)

    def test_compute_mces_different_molecules(self, sample_mgf_casmi, hydra_config):
        """Test MCES computation for different molecules."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=2,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_mces(
            [spectra[0]], [spectra[1]], threshold=20
        )

        assert result.shape == (1, 1)
        assert result[0, 0] >= 0.0
        assert isinstance(result, np.ndarray)

    def test_compute_mces_threshold_handling(self, sample_mgf_casmi, hydra_config):
        """Test MCES computation with custom threshold."""
        # config removed - using hydra_config
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=2,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        result = MolecularSimilarityMetrics.compute_mces(
            [spectra[0]], [spectra[1]], threshold=30
        )

        assert result.shape == (1, 1)
        assert result[0, 0] >= 0.0
        assert isinstance(result, np.ndarray)
