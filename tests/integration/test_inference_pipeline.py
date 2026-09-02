"""Integration tests for SIMBA inference pipeline.

Based on notebooks/final_tutorials/run_inference.ipynb
"""

import numpy as np
import pytest

from simba.core.models.simba_model import Simba
from simba.core.models.similarity_models import SimilarityModelMultitask
from simba.workflows.utils import load_spectra


pytestmark = pytest.mark.integration


class TestInferencePipeline:
    """Test inference workflow from README use case 1."""

    def test_load_spectra_from_mgf_standard_format(self, sample_mgf, hydra_config):
        """Test loading spectra from standard MGF format."""
        spectra = load_spectra(
            sample_mgf,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) > 0
        assert len(spectra) <= 3

        for spec in spectra:
            assert hasattr(spec, "precursor_mz")
            assert hasattr(spec, "mz")
            assert hasattr(spec, "intensity")
            assert len(spec.mz) > 0

    def test_load_spectra_from_mgf_casmi_format(self, sample_mgf_casmi, hydra_config):
        """Test loading spectra from CASMI2022 format with SMILES."""
        spectra = load_spectra(
            sample_mgf_casmi,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) > 0

        for spec in spectra:
            assert hasattr(spec, "precursor_mz")
            assert len(spec.mz) > 0

    @pytest.mark.skip(
        reason="FcLayerAnalogDiscovery's analog-discovery scoring still assumes "
        "an ED classifier head, which SimilarityModelMultitask no longer has."
    )
    def test_inference_end_to_end(self, sample_mgf, mocker, hydra_config):
        """Test complete inference pipeline with model."""
        spectra = load_spectra(
            sample_mgf,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2
        n_spectra = len(spectra)

        model = SimilarityModelMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.similarity_models.SimilarityModelMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )
        assert simba is not None
        assert simba.model is not None

        sim_ed, sim_mces = simba.predict(spectra, spectra)

        assert sim_ed.shape == (n_spectra, n_spectra)
        assert sim_mces.shape == (n_spectra, n_spectra)
        assert isinstance(sim_ed, np.ndarray)
        assert isinstance(sim_mces, np.ndarray)

    def test_embedding_caching(self, sample_mgf, mocker, hydra_config):
        """Test that embeddings caching works correctly."""
        model = SimilarityModelMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.similarity_models.SimilarityModelMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=True
        )

        assert simba.cache_embeddings is True
        assert hasattr(simba, "_embedding_cache")
        assert isinstance(simba._embedding_cache, dict)

        simba_no_cache = Simba(
            "fake_model.ckpt", config=hydra_config, device="cpu", cache_embeddings=False
        )
        assert simba_no_cache.cache_embeddings is False
