"""Tests for simba/core/models/similarity_models.py::SimilarityModelMultitask"""

import pytest
import torch

from simba.core.models.similarity_models import SimilarityModelMultitask


class TestEmbedderMultitask:
    @pytest.fixture
    def embedder_config(self):
        return {
            "d_model": 128,
            "n_layers": 2,
            "dropout": 0.1,
            "weights": None,
            "lr": 0.001,
            "use_element_wise": True,
            "use_cosine_distance": True,
            "use_precursor_mz_for_model": True,
            "use_adduct": True,
            "use_ce": False,
            "use_ion_activation": False,
            "use_ion_method": False,
        }

    @pytest.fixture
    def embedder(self, embedder_config):
        return SimilarityModelMultitask(**embedder_config)

    @pytest.fixture
    def sample_batch(self):
        batch_size = 2
        n_peaks = 10
        n_adducts = 48  # Length of ADDUCT_TO_MASS dictionary

        return {
            "mz_0": torch.randn(batch_size, n_peaks),
            "intensity_0": torch.randn(batch_size, n_peaks).abs(),
            "mz_1": torch.randn(batch_size, n_peaks),
            "intensity_1": torch.randn(batch_size, n_peaks).abs(),
            "precursor_mass_0": torch.randn(batch_size, 1),
            "precursor_charge_0": torch.ones(batch_size, 1),
            "precursor_mass_1": torch.randn(batch_size, 1),
            "precursor_charge_1": torch.ones(batch_size, 1),
            "adduct_0": torch.zeros(batch_size, n_adducts),  # One-hot encoded
            "adduct_1": torch.zeros(batch_size, n_adducts),  # One-hot encoded
            "ionmode_0": torch.ones(batch_size, 1),
            "ionmode_1": torch.ones(batch_size, 1),
            "ce_0": torch.ones(batch_size, 1) * 30.0,
            "ce_1": torch.ones(batch_size, 1) * 30.0,
            "ion_activation_0": torch.zeros(batch_size, 1),
            "ion_activation_1": torch.zeros(batch_size, 1),
            "ion_method_0": torch.zeros(batch_size, 1),
            "ion_method_1": torch.zeros(batch_size, 1),
            "similarity": torch.tensor([0.8, 0.6]),
            "similarity_2": torch.tensor([0.7, 0.5]),
            "mces": torch.tensor([0.7, 0.5]),
            "mol_idx_0": torch.tensor([10, 20]),
            "mol_idx_1": torch.tensor([10, 20]),
        }

    def test_init_basic(self, embedder_config):
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.use_cosine_distance is True
        assert embedder.use_mces_bucket_head is False

    def test_configure_optimizers(self, embedder):
        optimizer = embedder.configure_optimizers()

        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_compute_from_embeddings(self, embedder, embedder_config):
        batch_size = 4
        d_model = embedder_config["d_model"]

        emb0 = torch.randn(batch_size, d_model)
        emb1 = torch.randn(batch_size, d_model)

        result = embedder.compute_from_embeddings(emb0, emb1)

        assert len(result) == 1
        (emb_sim_2,) = result
        assert emb_sim_2.shape[0] == batch_size

    def test_forward_basic(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)

        assert len(result) == 1
        (emb_sim_2,) = result
        assert emb_sim_2.shape[0] == sample_batch["mz_0"].shape[0]
        assert not torch.isnan(emb_sim_2).any()

    def test_forward_with_return_spectrum_output(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch, return_spectrum_output=True)

        assert len(result) == 3
        emb_sim_2, emb0, emb1 = result
        batch_size = sample_batch["mz_0"].shape[0]
        assert emb_sim_2.shape[0] == batch_size
        assert emb0.shape[0] == batch_size
        assert emb1.shape[0] == batch_size

    def test_forward_without_adduct(self, embedder_config, sample_batch):
        embedder_config["use_adduct"] = False
        embedder = SimilarityModelMultitask(**embedder_config)

        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)

        assert len(result) == 1
        (emb_sim_2,) = result
        assert emb_sim_2.shape[0] == sample_batch["mz_0"].shape[0]

    def test_training_step_basic(self, embedder, sample_batch):
        result = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(result, dict)
        assert not torch.isnan(result["loss"])

    def test_validation_step(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            loss = embedder.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, dict)
        assert "loss" in loss
        assert "mces_pred" in loss and "mces_target" in loss
        assert not torch.isnan(loss["loss"])

    def test_training_step_with_contrastive_loss(self, embedder_config, sample_batch):
        embedder_config["use_contrastive_loss"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        result = embedder.training_step(sample_batch, batch_idx=0)

        assert not torch.isnan(result["loss"])

    def test_validation_step_with_contrastive_loss(self, embedder_config, sample_batch):
        embedder_config["use_contrastive_loss"] = True
        embedder = SimilarityModelMultitask(**embedder_config)
        embedder.eval()

        with torch.no_grad():
            loss = embedder.validation_step(sample_batch, batch_idx=0)

        assert not torch.isnan(loss["loss"])

    def test_contrastive_loss_info_nce_basic(self, embedder_config):
        embedder_config["use_contrastive_loss"] = True
        embedder = SimilarityModelMultitask(**embedder_config)
        d_model = embedder_config["d_model"]

        emb0 = torch.randn(4, d_model)
        emb1 = torch.randn(4, d_model)
        mol_idx_0 = torch.tensor([1, 2, 3, 4])
        mol_idx_1 = torch.tensor([1, 2, 3, 4])

        loss, n_pairs = embedder._contrastive_loss_info_nce(
            emb0, emb1, mol_idx_0, mol_idx_1
        )

        assert n_pairs == 4
        assert loss is not None
        assert not torch.isnan(loss)

    def test_contrastive_loss_info_nce_too_few_pairs(self, embedder_config):
        embedder_config["use_contrastive_loss"] = True
        embedder = SimilarityModelMultitask(**embedder_config)
        d_model = embedder_config["d_model"]

        emb0 = torch.randn(4, d_model)
        emb1 = torch.randn(4, d_model)
        mol_idx_0 = torch.tensor([1, 2, 3, 4])
        mol_idx_1 = torch.tensor([1, 20, 30, 40])

        loss, n_pairs = embedder._contrastive_loss_info_nce(
            emb0, emb1, mol_idx_0, mol_idx_1
        )

        assert n_pairs == 1
        assert loss is None

    def test_contrastive_projection_head_off_by_default(self, embedder_config):
        embedder_config["use_contrastive_loss"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.contrastive_use_projection_head is False
        assert not hasattr(embedder, "contrastive_projection")

    def test_contrastive_loss_info_nce_with_projection_head(self, embedder_config):
        embedder_config["use_contrastive_loss"] = True
        embedder_config["contrastive_use_projection_head"] = True
        embedder = SimilarityModelMultitask(**embedder_config)
        d_model = embedder_config["d_model"]

        assert hasattr(embedder, "contrastive_projection")

        emb0 = torch.randn(4, d_model)
        emb1 = torch.randn(4, d_model)
        mol_idx_0 = torch.tensor([1, 2, 3, 4])
        mol_idx_1 = torch.tensor([1, 2, 3, 4])

        loss, n_pairs = embedder._contrastive_loss_info_nce(
            emb0, emb1, mol_idx_0, mol_idx_1
        )

        assert n_pairs == 4
        assert loss is not None
        assert not torch.isnan(loss)

    def test_test_step(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.test_step(sample_batch, batch_idx=0)

        # test_step may not be defined, returns None
        # If it is defined, it should return a tensor
        if result is not None:
            assert isinstance(result, torch.Tensor)
            assert not torch.isnan(result)
