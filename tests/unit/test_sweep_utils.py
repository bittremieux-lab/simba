"""Unit tests for hyperparameter sweep utilities.

Tests cover the helper functions in simba.commands.sweep_train that
can be tested without running training: distribution building, param
sampling, and config validation.
"""

import optuna
import pytest
from omegaconf import OmegaConf

from simba.commands.sweep_train import (
    _build_distributions,
    _sample_params,
)


@pytest.fixture
def params_cfg():
    """Minimal sweep.params config covering all four types."""
    return OmegaConf.create(
        {
            "optimizer.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "model.transformer.d_model": {
                "type": "categorical",
                "choices": [128, 256, 512],
            },
            "model.transformer.n_layers": {"type": "int", "low": 2, "high": 6},
            "training.gradient_clip_val": {"type": "uniform", "low": 0.5, "high": 2.0},
        }
    )


class TestBuildDistributions:
    def test_returns_dict_with_all_params(self, params_cfg):
        dists = _build_distributions(params_cfg)
        assert set(dists.keys()) == set(params_cfg.keys())

    def test_loguniform_type(self, params_cfg):
        dists = _build_distributions(params_cfg)
        assert isinstance(
            dists["optimizer.lr"],
            optuna.distributions.LogUniformDistribution,
        )

    def test_categorical_type(self, params_cfg):
        dists = _build_distributions(params_cfg)
        assert isinstance(
            dists["model.transformer.d_model"],
            optuna.distributions.CategoricalDistribution,
        )
        assert list(dists["model.transformer.d_model"].choices) == [128, 256, 512]

    def test_int_type(self, params_cfg):
        dists = _build_distributions(params_cfg)
        assert isinstance(
            dists["model.transformer.n_layers"],
            optuna.distributions.IntUniformDistribution,
        )

    def test_uniform_type(self, params_cfg):
        dists = _build_distributions(params_cfg)
        assert isinstance(
            dists["training.gradient_clip_val"],
            optuna.distributions.UniformDistribution,
        )

    def test_unknown_type_raises(self):
        bad_cfg = OmegaConf.create(
            {"some.param": {"type": "unknown", "low": 0, "high": 1}}
        )
        with pytest.raises(ValueError, match="Unknown param type"):
            _build_distributions(bad_cfg)

    def test_empty_config_returns_empty_dict(self):
        assert _build_distributions(OmegaConf.create({})) == {}


class TestSampleParams:
    def test_all_params_sampled(self, params_cfg):
        study = optuna.create_study()
        trial = study.ask()
        sampled = _sample_params(trial, params_cfg)
        assert set(sampled.keys()) == set(params_cfg.keys())

    def test_sampled_values_in_bounds(self, params_cfg):
        study = optuna.create_study()
        for _ in range(5):
            trial = study.ask()
            sampled = _sample_params(trial, params_cfg)
            study.tell(trial, 1.0)

            assert 1e-5 <= sampled["optimizer.lr"] <= 1e-2
            assert sampled["model.transformer.d_model"] in [128, 256, 512]
            assert 2 <= sampled["model.transformer.n_layers"] <= 6
            assert 0.5 <= sampled["training.gradient_clip_val"] <= 2.0

    def test_enqueued_values_used_first(self, params_cfg):
        """Enqueued starting params must be returned by _sample_params unchanged."""
        starting = {
            "optimizer.lr": 0.001,
            "model.transformer.d_model": 256,
            "model.transformer.n_layers": 4,
            "training.gradient_clip_val": 1.0,
        }
        study = optuna.create_study()
        study.enqueue_trial(starting)
        trial = study.ask()
        sampled = _sample_params(trial, params_cfg)

        assert sampled["optimizer.lr"] == pytest.approx(0.001)
        assert sampled["model.transformer.d_model"] == 256
        assert sampled["model.transformer.n_layers"] == 4
        assert sampled["training.gradient_clip_val"] == pytest.approx(1.0)
