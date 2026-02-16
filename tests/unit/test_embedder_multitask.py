"""Tests for simba/ordinal_classification/embedder_multitask.py"""

import pytest
import torch
import torch.nn as nn

from simba.core.models.ordinal.embedder_multitask import (
    CustomizedCrossEntropyLoss,
    EmbedderMultitask,
)


class TestCustomizedCrossEntropyLoss:
    def test_init(self):
        loss_fn = CustomizedCrossEntropyLoss(n_classes=6)

        assert loss_fn.n_classes == 6
        assert loss_fn.penalty_matrix.shape == (6, 6)
        assert torch.all(loss_fn.penalty_matrix >= 0)
        assert torch.all(loss_fn.penalty_matrix <= 1)

    def test_forward_correct_prediction(self):
        loss_fn = CustomizedCrossEntropyLoss(n_classes=6)

        # Logits strongly favor class 2
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([2])

        loss = loss_fn.forward(logits, target)

        # Loss should be non-negative (penalty matrix is normalized)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_wrong_prediction(self):
        loss_fn = CustomizedCrossEntropyLoss(n_classes=6)

        # Logits favor class 0, but target is class 5 (far away)
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([5])

        loss = loss_fn.forward(logits, target)

        # Loss should be larger for distant wrong prediction
        assert loss.item() > 0.5

    def test_forward_batch(self):
        loss_fn = CustomizedCrossEntropyLoss(n_classes=6)

        batch_size = 4
        logits = torch.randn(batch_size, 6)
        target = torch.tensor([0, 1, 2, 3])

        loss = loss_fn.forward(logits, target)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_penalty_matrix_symmetry(self):
        loss_fn = CustomizedCrossEntropyLoss(n_classes=6)

        # Diagonal should have maximum values (correct predictions)
        diagonal = torch.diag(loss_fn.penalty_matrix)
        assert torch.all(diagonal >= loss_fn.penalty_matrix)


class TestEmbedderMultitask:
    @pytest.fixture
    def embedder_config(self):
        return {
            "d_model": 128,
            "n_layers": 2,
            "n_classes": 6,
            "use_gumbel": False,
            "dropout": 0.1,
            "weights": None,
            "lr": 0.001,
            "use_element_wise": True,
            "use_cosine_distance": True,
            "weights_sim2": None,
            "use_edit_distance_regresion": False,
            "use_mces20_log_loss": True,
            "use_fingerprints": False,
            "use_precursor_mz_for_model": True,
            "tau_gumbel_softmax": 10,
            "gumbel_reg_weight": 0.1,
            "USE_LEARNABLE_MULTITASK": True,
            "use_adduct": True,
            "use_ce": False,
            "use_ion_activation": False,
            "use_ion_method": False,
        }

    @pytest.fixture
    def embedder(self, embedder_config):
        return EmbedderMultitask(**embedder_config)

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
            "similarity": torch.tensor([0.8, 0.6]),
            "similarity_2": torch.tensor([0.7, 0.5]),
        }

    def test_init_basic(self, embedder_config):
        embedder = EmbedderMultitask(**embedder_config)

        assert embedder.classifier is not None
        assert isinstance(embedder.classifier, nn.Linear)
        assert embedder.loss_fn is not None
        assert embedder.regression_loss is not None

    def test_init_with_gumbel(self, embedder_config):
        embedder_config["use_gumbel"] = True
        embedder = EmbedderMultitask(**embedder_config)

        assert embedder.use_gumbel is True
        assert embedder.tau_gumbel_softmax == 10
        assert embedder.gumbel_reg_weight == 0.1

    def test_init_with_fingerprints(self, embedder_config):
        embedder_config["use_fingerprints"] = True
        embedder = EmbedderMultitask(**embedder_config)

        assert embedder.use_fingerprints is True
        assert embedder.linear_fingerprint_0 is not None
        assert embedder.linear_fingerprint_1 is not None

    def test_init_with_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = EmbedderMultitask(**embedder_config)

        assert embedder.use_edit_distance_regresion is True
        assert embedder.linear1_cossim is not None

    def test_calculate_weight_loss2_with_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = EmbedderMultitask(**embedder_config)

        weight = embedder.calculate_weight_loss2()
        assert weight == 1

    def test_calculate_weight_loss2_without_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = False
        embedder = EmbedderMultitask(**embedder_config)

        weight = embedder.calculate_weight_loss2()
        assert weight == 200

    def test_compute_adjacent_diffs(self, embedder):
        batch_size = 4
        n_classes = 6
        gumbel_probs = torch.rand(batch_size, n_classes)
        gumbel_probs = gumbel_probs / gumbel_probs.sum(dim=1, keepdim=True)

        result = embedder.compute_adjacent_diffs(gumbel_probs, batch_size)

        # Result is a scalar (averaged over batch)
        assert result.dim() == 0
        assert result.item() >= 0

    def test_ordinal_loss(self, embedder):
        logits = torch.randn(4, 6)
        target = torch.tensor([0.0, 1.0, 2.0, 3.0])

        loss = embedder.ordinal_loss(logits, target)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_gumbel_softmax(self, embedder):
        logits = torch.randn(4, 6)

        result = embedder.gumbel_softmax(logits, temperature=1.0, hard=True)

        assert result.shape == logits.shape
        # Hard gumbel should be one-hot
        assert torch.allclose(result.sum(dim=1), torch.ones(4))

    def test_ordinal_cross_entropy(self, embedder):
        pred = torch.randn(4, 6)
        # Target must be integer indices (0 to 5 for 6 classes)
        target = torch.tensor([0, 2, 4, 5])

        loss = embedder.ordinal_cross_entropy(pred, target)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, embedder):
        optimizer = embedder.configure_optimizers()

        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_compute_from_embeddings(self, embedder):
        batch_size = 4
        d_model = embedder.linear1.in_features

        emb0 = torch.randn(batch_size, d_model)
        emb1 = torch.randn(batch_size, d_model)

        result = embedder.compute_from_embeddings(emb0, emb1)

        assert len(result) == 2
        emb, emb_sim_2 = result
        assert emb.shape[0] == batch_size
        assert emb_sim_2.shape[0] == batch_size

    def test_forward_basic(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)

        assert len(result) == 2
        emb, emb_sim_2 = result
        assert emb.shape[0] == sample_batch["mz_0"].shape[0]
        assert emb_sim_2.shape[0] == sample_batch["mz_0"].shape[0]
        assert not torch.isnan(emb).any()
        assert not torch.isnan(emb_sim_2).any()

    def test_forward_with_return_spectrum_output(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch, return_spectrum_output=True)

        assert len(result) == 4
        emb, emb_sim_2, emb0, emb1 = result
        batch_size = sample_batch["mz_0"].shape[0]
        assert emb.shape[0] == batch_size
        assert emb_sim_2.shape[0] == batch_size
        assert emb0.shape[0] == batch_size
        assert emb1.shape[0] == batch_size

    def test_forward_with_fingerprints(self, embedder_config, sample_batch):
        embedder_config["use_fingerprints"] = True
        embedder = EmbedderMultitask(**embedder_config)

        # Add fingerprints to batch
        batch_size = sample_batch["mz_0"].shape[0]
        sample_batch["fingerprint_0"] = torch.randn(batch_size, 2048)
        sample_batch["fingerprint_1"] = torch.randn(batch_size, 2048)

        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)

        assert len(result) == 2
        emb, emb_sim_2 = result
        assert emb.shape[0] == batch_size

    def test_forward_without_adduct(self, embedder_config, sample_batch):
        # Test with use_adduct=False but keep USE_LEARNABLE_MULTITASK=True
        embedder_config["use_adduct"] = False
        embedder = EmbedderMultitask(**embedder_config)

        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)

        assert len(result) == 2
        emb, emb_sim_2 = result
        assert emb.shape[0] == sample_batch["mz_0"].shape[0]

    def test_training_step_basic(self, embedder, sample_batch):
        # Add required fields for training_step
        sample_batch["ed"] = torch.tensor([2, 3])  # Edit distance targets
        sample_batch["mces"] = torch.tensor([0.7, 0.5])  # MCES targets

        loss = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_training_step_with_gumbel(self, embedder_config, sample_batch):
        embedder_config["use_gumbel"] = True
        embedder = EmbedderMultitask(**embedder_config)

        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        loss = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(loss)

    def test_training_step_with_edit_distance_regression(
        self, embedder_config, sample_batch
    ):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = EmbedderMultitask(**embedder_config)

        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        loss = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(loss)

    def test_validation_step(self, embedder, sample_batch):
        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        embedder.eval()
        with torch.no_grad():
            loss = embedder.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(loss)

    def test_test_step(self, embedder, sample_batch):
        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        embedder.eval()
        with torch.no_grad():
            result = embedder.test_step(sample_batch, batch_idx=0)

        # test_step may not be defined, returns None
        # If it is defined, it should return a tensor
        if result is not None:
            assert isinstance(result, torch.Tensor)
            # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
            assert not torch.isnan(result)
