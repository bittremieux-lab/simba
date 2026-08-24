"""Tests for simba/ordinal_classification/embedder_multitask.py"""

import pytest
import torch
import torch.nn as nn

from simba.core.models.similarity_models import (
    CustomizedCrossEntropyLoss,
    SimilarityModelMultitask,
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
            "mol_idx_0": torch.tensor([0, 1]),
            "mol_idx_1": torch.tensor([1, 2]),
            "spec_idx_0": torch.tensor([0, 1]),
            "spec_idx_1": torch.tensor([1, 2]),
            "smiles_0": ["CCO", "CCN"],
            "smiles_1": ["CCN", "CCC"],
        }

    def test_init_basic(self, embedder_config):
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.classifier is not None
        assert isinstance(embedder.classifier, nn.Linear)
        assert embedder.loss_fn is not None
        assert embedder.regression_loss is not None

    def test_init_with_gumbel(self, embedder_config):
        embedder_config["use_gumbel"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.use_gumbel is True
        assert embedder.tau_gumbel_softmax == 10
        assert embedder.gumbel_reg_weight == 0.1

    def test_init_with_fingerprints(self, embedder_config):
        embedder_config["use_fingerprints"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.use_fingerprints is True
        assert embedder.linear_fingerprint_0 is not None
        assert embedder.linear_fingerprint_1 is not None

    def test_init_with_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        assert embedder.use_edit_distance_regresion is True
        assert embedder.linear1_cossim is not None

    def test_calculate_weight_loss2_with_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        weight = embedder.calculate_weight_loss2()
        assert weight == 1

    def test_calculate_weight_loss2_without_edit_distance(self, embedder_config):
        embedder_config["use_edit_distance_regresion"] = False
        embedder = SimilarityModelMultitask(**embedder_config)

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
        embedder = SimilarityModelMultitask(**embedder_config)

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
        embedder = SimilarityModelMultitask(**embedder_config)

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

        out = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(out, dict)
        assert isinstance(out["loss"], torch.Tensor)
        assert not torch.isnan(out["loss"])
        assert "mces_pred" in out
        assert "mces_target" in out

    def test_training_step_with_gumbel(self, embedder_config, sample_batch):
        embedder_config["use_gumbel"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        out = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(out, dict)
        assert isinstance(out["loss"], torch.Tensor)
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(out["loss"])

    def test_training_step_with_edit_distance_regression(
        self, embedder_config, sample_batch
    ):
        embedder_config["use_edit_distance_regresion"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        out = embedder.training_step(sample_batch, batch_idx=0)

        assert isinstance(out, dict)
        assert isinstance(out["loss"], torch.Tensor)
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(out["loss"])

    def test_validation_step(self, embedder, sample_batch):
        # Add required fields
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        embedder.eval()
        with torch.no_grad():
            loss = embedder.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, dict)
        assert "loss" in loss
        assert "mces_pred" in loss and "mces_target" in loss
        # No ED outputs -- the ED head isn't scored in validation_step anymore
        assert "ed_pred" not in loss and "ed_target" not in loss
        # Pair-identity fields needed downstream for the per-pair CSV dump
        for key in (
            "mol_idx_0",
            "mol_idx_1",
            "spec_idx_0",
            "spec_idx_1",
            "smiles_0",
            "smiles_1",
        ):
            assert key in loss
        # Note: loss can be negative when USE_LEARNABLE_MULTITASK=True due to learnable weights
        assert not torch.isnan(loss["loss"])

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

    def test_use_edit_distance_false_skips_ed_computation(
        self, embedder_config, sample_batch
    ):
        """When ED is fully disabled, its head shouldn't run at all -- no
        batch["ed"] needed, emb (logits1) is None, not just excluded from
        the loss sum."""
        embedder_config["use_edit_distance"] = False
        embedder = SimilarityModelMultitask(**embedder_config)

        emb0 = torch.randn(4, embedder_config["d_model"])
        emb1 = torch.randn(4, embedder_config["d_model"])
        emb, emb_sim_2 = embedder.compute_from_embeddings(emb0, emb1)
        assert emb is None

        # No "ed" key in the batch at all -- should not be read/needed.
        sample_batch["mces"] = torch.tensor([0.7, 0.5])
        out_train = embedder.training_step(sample_batch, batch_idx=0)
        assert not torch.isnan(out_train["loss"])
        assert "ed_pred" not in out_train

        embedder.eval()
        with torch.no_grad():
            out_val = embedder.validation_step(sample_batch, batch_idx=0)
        assert not torch.isnan(out_val["loss"])

    def test_step_reuses_precomputed_logits_list(self, embedder, sample_batch):
        """training_step/validation_step should only call the encoder once
        per batch (via forward), passing the same logits_list into step()
        instead of forward()-ing a second time."""
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        call_count = {"n": 0}
        original_forward = embedder.forward

        def counting_forward(*args, **kwargs):
            call_count["n"] += 1
            return original_forward(*args, **kwargs)

        embedder.forward = counting_forward
        embedder.training_step(sample_batch, batch_idx=0)
        assert call_count["n"] == 1

        call_count["n"] = 0
        embedder.eval()
        with torch.no_grad():
            embedder.validation_step(sample_batch, batch_idx=0)
        assert call_count["n"] == 1

    def test_loss_components_and_sigma_logged(self, embedder, sample_batch):
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        logged = {}
        embedder.log = lambda name, value, **kw: logged.__setitem__(name, value)

        embedder.training_step(sample_batch, batch_idx=0)

        for key in ("loss_ed", "loss_mces", "log_sigma1", "log_sigma2"):
            assert key in logged, f"{key} was not logged"
        assert "loss_mces_bucket" not in logged
        assert "log_sigma3" not in logged


class TestMcesBucketHead:
    """Optional second target (model.tasks.mces_bucket): a CORN-style
    ordinal classification head trained in parallel on MCES, on top of
    whatever the primary task/head_mode is. Disabled by default (see
    TestEmbedderMultitask above, which covers use_mces_bucket_head=False
    unaffected by any of this)."""

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
            "use_mces_bucket_head": True,
        }

    @pytest.fixture
    def embedder(self, embedder_config):
        return SimilarityModelMultitask(**embedder_config)

    @pytest.fixture
    def sample_batch(self):
        batch_size = 2
        n_peaks = 10
        n_adducts = 48

        return {
            "mz_0": torch.randn(batch_size, n_peaks),
            "intensity_0": torch.randn(batch_size, n_peaks).abs(),
            "mz_1": torch.randn(batch_size, n_peaks),
            "intensity_1": torch.randn(batch_size, n_peaks).abs(),
            "precursor_mass_0": torch.randn(batch_size, 1),
            "precursor_charge_0": torch.ones(batch_size, 1),
            "precursor_mass_1": torch.randn(batch_size, 1),
            "precursor_charge_1": torch.ones(batch_size, 1),
            "adduct_0": torch.zeros(batch_size, n_adducts),
            "adduct_1": torch.zeros(batch_size, n_adducts),
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
            "mol_idx_0": torch.tensor([0, 1]),
            "mol_idx_1": torch.tensor([1, 2]),
            "spec_idx_0": torch.tensor([0, 1]),
            "spec_idx_1": torch.tensor([1, 2]),
            "smiles_0": ["CCO", "CCN"],
            "smiles_1": ["CCN", "CCC"],
        }

    def test_init(self, embedder):
        assert embedder.use_mces_bucket_head is True
        assert isinstance(embedder.mces_bucket_head, nn.Linear)
        # 4 edges -> 6 classes (singleton 0 + 4 finite bins + open-ended top)
        assert embedder.mces_bucket_n_classes == 6
        assert embedder.mces_bucket_head.out_features == 5  # n_classes - 1
        assert hasattr(embedder, "log_sigma3")

    def test_disabled_head_has_no_new_attributes(self):
        config = {
            "d_model": 128,
            "n_layers": 2,
            "n_classes": 6,
            "use_gumbel": False,
            "lr": 0.001,
        }
        embedder = SimilarityModelMultitask(**config)
        assert embedder.use_mces_bucket_head is False
        assert not hasattr(embedder, "mces_bucket_head")
        assert not hasattr(embedder, "log_sigma3")

    def test_target_bins_match_requested_scheme(self, embedder):
        # 0, (0,2], (2,4], (4,6], (6,8], (8,inf) -> classes 0..5 -- (0,1] and
        # (1,2] were originally separate but merged into one (0,2] bin since
        # (1,2] was empty in experiment 013's validation set.
        raw = torch.tensor(
            [0.0, 0.3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.1, 50.0]
        )
        expected = torch.tensor([0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        bins = embedder._mces_bucket_target_bins(raw)
        assert torch.equal(bins, expected)

    def test_compute_from_embeddings_returns_three(self, embedder):
        batch_size = 4
        emb0 = torch.randn(batch_size, 128)
        emb1 = torch.randn(batch_size, 128)

        result = embedder.compute_from_embeddings(emb0, emb1)

        assert len(result) == 3
        emb, emb_sim_2, emb_sim_3 = result
        assert emb_sim_3.shape == (batch_size, 5)

    def test_forward_basic_returns_three(self, embedder, sample_batch):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch)
        assert len(result) == 3

    def test_forward_with_return_spectrum_output_returns_five(
        self, embedder, sample_batch
    ):
        embedder.eval()
        with torch.no_grad():
            result = embedder.forward(sample_batch, return_spectrum_output=True)
        assert len(result) == 5
        emb, emb_sim_2, emb_sim_3, emb0, emb1 = result
        batch_size = sample_batch["mz_0"].shape[0]
        assert emb0.shape[0] == batch_size
        assert emb1.shape[0] == batch_size

    def test_validation_step_reports_bucket_pred_and_target(
        self, embedder, sample_batch
    ):
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        embedder.eval()
        with torch.no_grad():
            out = embedder.validation_step(sample_batch, batch_idx=0)

        assert "mces_bucket_pred" in out and "mces_bucket_target" in out
        assert out["mces_bucket_pred"].shape == (2,)
        assert not torch.isnan(out["loss"])

    def test_training_step_loss_not_nan(self, embedder, sample_batch):
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        out = embedder.training_step(sample_batch, batch_idx=0)

        assert not torch.isnan(out["loss"])

    def test_loss_mces_bucket_and_sigma3_logged(self, embedder, sample_batch):
        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        logged = {}
        embedder.log = lambda name, value, **kw: logged.__setitem__(name, value)

        embedder.training_step(sample_batch, batch_idx=0)

        for key in ("loss_mces_bucket", "log_sigma3", "loss_ed", "loss_mces"):
            assert key in logged, f"{key} was not logged"

    def test_bucket_only_learnable_weight_isolated_from_primary_task(
        self, embedder_config, sample_batch
    ):
        """mces_bucket_learnable_weight=True, USE_LEARNABLE_MULTITASK=False:
        log_sigma3 should exist and get logged, and the combined loss should
        equal (loss1 + weight_loss2 * loss2) -- the exact same primary-task
        formula as the plain fixed-weight path -- plus the bucket term under
        its own learnable log_sigma3, evaluated from one single forward pass
        (eval mode, no dropout) so there's no cross-call randomness to
        confound the comparison."""
        embedder_config["USE_LEARNABLE_MULTITASK"] = False
        embedder_config["mces_bucket_learnable_weight"] = True
        embedder = SimilarityModelMultitask(**embedder_config)

        assert hasattr(embedder, "log_sigma3")
        assert not hasattr(embedder, "log_sigma1")
        assert not hasattr(embedder, "log_sigma2")

        sample_batch["ed"] = torch.tensor([2, 3])
        sample_batch["mces"] = torch.tensor([0.7, 0.5])

        embedder.eval()
        with torch.no_grad():
            logits_list = embedder(sample_batch)
            loss = embedder.step(sample_batch, batch_idx=0, logits_list=logits_list)

            logits1, logits2, logits3 = logits_list
            target1 = sample_batch["ed"].long()
            target2 = sample_batch["mces"].float()
            loss1 = embedder.customised_ce(logits1, target1)
            squared_diff = (logits2.view(-1, 1) - target2.view(-1, 1)) ** 2
            loss2 = squared_diff.mean()
            weight_loss2 = embedder.calculate_weight_loss2()
            raw_mces_target = (1.0 - target2) * embedder.mces_max_value
            bucket_bins = embedder._mces_bucket_target_bins(raw_mces_target)
            loss3 = embedder._corn_loss_generic(
                logits3, bucket_bins, embedder.mces_bucket_n_classes
            )
            expected = (loss1 + weight_loss2 * loss2) + (
                torch.exp(-embedder.log_sigma3) * loss3 + embedder.log_sigma3
            )

        assert not torch.isnan(loss)
        assert torch.allclose(loss, expected, atol=1e-4)
