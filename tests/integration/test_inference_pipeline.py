"""Integration tests for SIMBA inference pipeline.

Based on notebooks/final_tutorials/run_inference.ipynb
"""

import numpy as np
import pytest

from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.models.embedder_multitask import EmbedderMultitask
from simba.core.models.simba_model import Simba


pytestmark = pytest.mark.integration


class TestInferencePipeline:
    """Test inference workflow from README use case 1."""

    def test_load_spectra_from_mgf_standard_format(self, sample_mgf, hydra_config):
        """Test loading spectra from standard MGF format."""
        spectra = PreprocessingSimba.load_spectra(
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
        spectra = PreprocessingSimba.load_spectra(
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

    def test_inference_end_to_end(self, sample_mgf, mocker, hydra_config):
        """Test complete inference pipeline with model."""
        spectra = PreprocessingSimba.load_spectra(
            sample_mgf,
            hydra_config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2
        n_spectra = len(spectra)

        model = EmbedderMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            n_classes=hydra_config.model.tasks.edit_distance.n_classes,
            use_gumbel=hydra_config.model.tasks.edit_distance.use_gumbel,
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
            use_edit_distance_regresion=hydra_config.model.tasks.edit_distance.use_regression,
            use_fingerprints=hydra_config.model.tasks.fingerprints.enabled,
            USE_LEARNABLE_MULTITASK=hydra_config.model.multitasking.learnable,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
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
        model = EmbedderMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            n_classes=hydra_config.model.tasks.edit_distance.n_classes,
            use_gumbel=hydra_config.model.tasks.edit_distance.use_gumbel,
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
            use_edit_distance_regresion=hydra_config.model.tasks.edit_distance.use_regression,
            use_fingerprints=hydra_config.model.tasks.fingerprints.enabled,
            USE_LEARNABLE_MULTITASK=hydra_config.model.multitasking.learnable,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
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
