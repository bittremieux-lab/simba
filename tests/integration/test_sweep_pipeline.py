"""Integration tests for the hyperparameter sweep pipeline.

Training is mocked so these tests run quickly without a GPU or real data.
They verify sweep orchestration: trial execution, trials.json persistence,
resume semantics, and starting_params enqueueing.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from simba.commands.sweep_train import _sample_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SWEEP_DEFAULT_YAML = (
    Path(__file__).parent.parent.parent / "simba" / "configs" / "sweep" / "default.yaml"
)


def _make_cfg(tmp_path, n_trials=2, resume=False, extra=None):
    """Build a minimal OmegaConf config for sweep tests.

    Loads sweep.params from the real default.yaml (preserves dotted keys),
    then overlays test-specific settings.  Inference is disabled by setting
    checkpoints.save_checkpoints=false so no Hydra compose() is needed.
    """
    output_dir = tmp_path / "sweep_out"

    # Load actual sweep defaults (preserves "optimizer.lr" as a dotted key)
    sweep_cfg = OmegaConf.load(_SWEEP_DEFAULT_YAML)

    # Base cfg with just the fields sweep_train reads from cfg
    base = OmegaConf.create({
        "paths": {
            "preprocessing_dir_train": str(tmp_path),
            "preprocessing_dir": None,
        },
        "checkpoints": {
            "save_checkpoints": False,  # skip inference — no compose() needed
        },
    })
    cfg = OmegaConf.merge(base, sweep_cfg)

    # Override sweep settings for the test
    OmegaConf.update(cfg, "sweep.output_dir", str(output_dir), merge=True)
    OmegaConf.update(cfg, "sweep.n_trials", n_trials, merge=True)
    OmegaConf.update(cfg, "sweep.resume", resume, merge=True)
    OmegaConf.update(cfg, "sweep.study_name", "test_study", merge=True)

    if extra:
        for k, v in extra.items():
            OmegaConf.update(cfg, k, v, merge=True)

    return cfg, output_dir


def _fake_run_trial(return_value=0.5):
    """Mock _run_trial: writes params.json, returns fixed loss, no training."""
    def _inner(base_cfg, trial, output_dir, params_cfg):
        trial_dir = output_dir / "checkpoints" / str(trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)
        sampled = _sample_params(trial, params_cfg)
        (trial_dir / "params.json").write_text(json.dumps(sampled))
        return return_value
    return _inner


def _run_sweep(cfg):
    """Invoke sweep_train's original body, bypassing @hydra.main."""
    from simba.commands.sweep_train import sweep_train
    sweep_train.__wrapped__(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSweepOrchestration:
    """Core sweep loop: execution, persistence, resume."""

    def test_trials_json_created(self, tmp_path):
        cfg, output_dir = _make_cfg(tmp_path, n_trials=2)
        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        trials = json.loads((output_dir / "trials.json").read_text())
        assert len(trials) == 2
        assert all(t["status"] == "completed" for t in trials)

    def test_each_trial_record_has_params_and_value(self, tmp_path):
        cfg, output_dir = _make_cfg(tmp_path, n_trials=2)
        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        for t in json.loads((output_dir / "trials.json").read_text()):
            assert "params" in t and "value" in t
            assert "optimizer.lr" in t["params"]

    def test_params_json_written_per_checkpoint(self, tmp_path):
        cfg, output_dir = _make_cfg(tmp_path, n_trials=2)
        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        for i in range(2):
            p = output_dir / "checkpoints" / str(i) / "params.json"
            assert p.exists(), f"params.json missing for trial {i}"
            assert "optimizer.lr" in json.loads(p.read_text())

    def test_fresh_start_removes_stale_checkpoints(self, tmp_path):
        cfg, output_dir = _make_cfg(tmp_path, n_trials=1)
        stale = output_dir / "checkpoints" / "99"
        stale.mkdir(parents=True)
        (stale / "dummy.txt").write_text("stale")

        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        assert not stale.exists()

    def test_resume_appends_to_prior_trials(self, tmp_path):
        output_dir = tmp_path / "sweep_out"
        output_dir.mkdir(parents=True)

        prior = [{
            "trial_number": 0,
            "params": {
                "optimizer.lr": 0.001,
                "model.transformer.d_model": 128,
                "model.transformer.n_layers": 4,
                "training.gradient_clip_val": 1.0,
            },
            "value": 0.8,
            "status": "completed",
            "timestamp": "2026-01-01T00:00:00",
        }]
        (output_dir / "trials.json").write_text(json.dumps(prior))

        cfg, _ = _make_cfg(tmp_path, n_trials=1, resume=True,
                           extra={"sweep.output_dir": str(output_dir)})
        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        trials = json.loads((output_dir / "trials.json").read_text())
        assert len(trials) == 2  # 1 prior + 1 new


class TestStartingParams:
    """sweep.starting_params: known values run as first trial(s)."""

    def test_starting_params_used_for_first_trial(self, tmp_path):
        starting = [{
            "optimizer.lr": 0.001,
            "model.transformer.d_model": 256,
            "model.transformer.n_layers": 5,
            "training.gradient_clip_val": 1.0,
        }]
        cfg, output_dir = _make_cfg(tmp_path, n_trials=1,
                                    extra={"sweep.starting_params": starting})
        with patch("simba.commands.sweep_train._run_trial", side_effect=_fake_run_trial()):
            _run_sweep(cfg)

        params = json.loads((output_dir / "checkpoints" / "0" / "params.json").read_text())
        assert params["optimizer.lr"] == pytest.approx(0.001, rel=1e-3)
        assert params["model.transformer.d_model"] == 256
        assert params["model.transformer.n_layers"] == 5

    def test_starting_params_ignored_on_resume(self, tmp_path):
        """Enqueued starting values must not apply when resume=true."""
        output_dir = tmp_path / "sweep_out"
        output_dir.mkdir(parents=True)
        prior = [{
            "trial_number": 0,
            "params": {
                "optimizer.lr": 0.001,
                "model.transformer.d_model": 128,
                "model.transformer.n_layers": 3,
                "training.gradient_clip_val": 0.5,
            },
            "value": 0.7,
            "status": "completed",
            "timestamp": "2026-01-01T00:00:00",
        }]
        (output_dir / "trials.json").write_text(json.dumps(prior))

        # lr=99.9 is outside [1e-5, 1e-2] — if it appears, starting_params ran
        starting = [{
            "optimizer.lr": 99.9,
            "model.transformer.d_model": 512,
            "model.transformer.n_layers": 8,
            "training.gradient_clip_val": 2.0,
        }]
        cfg, _ = _make_cfg(tmp_path, n_trials=1, resume=True, extra={
            "sweep.output_dir": str(output_dir),
            "sweep.starting_params": starting,
        })

        sampled_lrs = []

        def _capture(base_cfg, trial, out_dir, params_cfg):
            trial_dir = out_dir / "checkpoints" / str(trial.number)
            trial_dir.mkdir(parents=True, exist_ok=True)
            s = _sample_params(trial, params_cfg)
            sampled_lrs.append(s["optimizer.lr"])
            (trial_dir / "params.json").write_text(json.dumps(s))
            return 0.5

        with patch("simba.commands.sweep_train._run_trial", side_effect=_capture):
            _run_sweep(cfg)

        assert all(lr <= 1e-2 for lr in sampled_lrs), \
            "starting_params should be ignored on resume"

