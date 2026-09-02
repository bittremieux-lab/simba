"""Hyperparameter sweep for SIMBA — self-contained Optuna loop."""

import itertools
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf


_CONFIG_PATH = str(Path(__file__).parent.parent / "configs")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_distributions(params_cfg: DictConfig) -> dict:
    """Build Optuna distribution objects from sweep.params config."""
    distributions = {}
    for name, p in params_cfg.items():
        t = p.type
        if t == "loguniform":
            distributions[name] = optuna.distributions.LogUniformDistribution(
                float(p.low), float(p.high)
            )
        elif t == "uniform":
            distributions[name] = optuna.distributions.UniformDistribution(
                float(p.low), float(p.high)
            )
        elif t == "int":
            distributions[name] = optuna.distributions.IntUniformDistribution(
                int(p.low), int(p.high)
            )
        elif t == "categorical":
            distributions[name] = optuna.distributions.CategoricalDistribution(
                list(p.choices)
            )
        else:
            raise ValueError(
                f"Unknown param type '{t}' for '{name}'. "
                "Use: loguniform, uniform, int, categorical"
            )
    return distributions


def _sample_params(trial: optuna.Trial, params_cfg: DictConfig) -> dict:
    """Sample all search-space params from the Optuna trial."""
    sampled = {}
    for name, p in params_cfg.items():
        t = p.type
        if t == "loguniform":
            sampled[name] = trial.suggest_loguniform(name, float(p.low), float(p.high))
        elif t == "uniform":
            sampled[name] = trial.suggest_uniform(name, float(p.low), float(p.high))
        elif t == "int":
            sampled[name] = trial.suggest_int(name, int(p.low), int(p.high))
        elif t == "categorical":
            sampled[name] = trial.suggest_categorical(name, list(p.choices))
    return sampled


def _run_trial(
    base_cfg: DictConfig, trial: optuna.Trial, output_dir: Path, params_cfg: DictConfig
) -> float:
    """Sample hyperparameters, train one trial, return best validation loss."""
    import click

    from simba.core.training.train_utils import TrainUtils
    from simba.workflows.training import (
        create_dataloaders,
        load_dataset,
        prepare_data,
        setup_callbacks,
        setup_model,
    )
    from simba.workflows.training import train as run_training

    # Sample all params from the config-defined search space
    cfg = deepcopy(base_cfg)
    sampled = _sample_params(trial, params_cfg)

    for param_name, value in sampled.items():
        OmegaConf.update(cfg, param_name, value, merge=True)

    trial_dir = output_dir / "checkpoints" / str(trial.number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg, "paths.checkpoint_dir", str(trial_dir), merge=True)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Trial {trial.number}")
    for k, v in sampled.items():
        click.echo(f"  {k} = {v}")
    click.echo(f"  checkpoint → {trial_dir}")
    click.echo(f"{'=' * 60}")

    (mol_train, mol_val, mol_test, mol_test_uni) = load_dataset(cfg)
    (dataset_train, train_sampler, dataset_val, val_sampler) = prepare_data(
        mol_train, mol_val, mol_test, mol_test_uni, cfg
    )
    dataloader_train, dataloader_val = create_dataloaders(
        cfg, dataset_train, train_sampler, dataset_val, val_sampler
    )

    n_batches = len(dataloader_train)
    if n_batches == 0:
        raise RuntimeError("No training batches found.")
    if n_batches < cfg.training.val_check_interval:
        OmegaConf.update(
            cfg, "training.val_check_interval", max(1, n_batches // 2), merge=True
        )

    mces_sampled: list[float] = []
    for batch in itertools.islice(dataloader_train, 100):
        mces_sampled += list(batch["mces"].reshape(-1))

    counting_mces, _ = TrainUtils.count_ranges(
        np.array(mces_sampled), number_bins=5, bin_sim_1=False, max_value=1
    )
    weights_mces = np.array(
        [np.sum(counting_mces) / c if c != 0 else 0 for c in counting_mces]
    )
    weights_mces = weights_mces / np.sum(weights_mces)

    chk_cb, chk_n_cb, loss_cb, early_stop_cb, progress_log_cb, val_metrics_cb = (
        setup_callbacks(cfg)
    )
    model = setup_model(cfg, weights_mces)
    trainer = run_training(
        model,
        dataloader_train,
        dataloader_val,
        cfg,
        chk_cb,
        chk_n_cb,
        loss_cb,
        early_stop_cb,
        progress_log_cb,
        val_metrics_cb,
    )

    val_loss = float(trainer.callback_metrics.get("validation_loss", math.inf))
    click.echo(f"Trial {trial.number} finished — val_loss={val_loss:.6f}")

    # Save params alongside the checkpoint — this is the ground-truth record of
    # which hyperparams produced this checkpoint, surviving any resume/restart.
    with open(trial_dir / "params.json", "w") as f:
        json.dump(sampled, f, indent=2)

    return val_loss


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base=None)
def sweep_train(cfg: DictConfig) -> None:
    """Run a self-contained Optuna hyperparameter sweep with JSON-based resume."""
    import click

    n_trials: int = int(OmegaConf.select(cfg, "sweep.n_trials", default=10))
    output_dir = Path(
        str(
            OmegaConf.select(
                cfg,
                "sweep.output_dir",
                default=f"./sweeps/optuna_{datetime.now():%Y%m%d_%H%M%S}",
            )
        )
    )
    resume: bool = bool(OmegaConf.select(cfg, "sweep.resume", default=False))
    study_name: str = str(
        OmegaConf.select(cfg, "sweep.study_name", default="simba_hyperparam_search")
    )

    params_cfg = OmegaConf.select(cfg, "sweep.params")
    if params_cfg is None:
        raise ValueError(
            "No search space defined. Load a sweep config:\n"
            "  simba-sweep +sweep=default ...\n"
            "Or add sweep.params.* overrides directly."
        )
    distributions = _build_distributions(params_cfg)

    preprocessing_dir = cfg.paths.preprocessing_dir_train or cfg.paths.preprocessing_dir
    if not preprocessing_dir:
        raise ValueError("paths.preprocessing_dir_train is required.")
    if not Path(preprocessing_dir).exists():
        raise FileNotFoundError(f"Preprocessing dir not found: {preprocessing_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_file = output_dir / "trials.json"
    checkpoints_dir = output_dir / "checkpoints"

    # Load previous results when resuming
    previous_trials: list[dict] = []
    if resume:
        if not trials_file.exists():
            raise FileNotFoundError(
                f"sweep.resume=true but no trials.json in {output_dir}.\n"
                "Run without resume first."
            )
        with open(trials_file) as f:
            previous_trials = json.load(f)
        click.echo(f"Resuming: loaded {len(previous_trials)} trials from {trials_file}")
    else:
        click.echo(f"Fresh sweep — output: {output_dir}")
        # Remove stale per-trial checkpoint dirs so no old checkpoint is reused
        # with the wrong hyperparams from a new trial with the same index.
        import shutil

        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            click.echo(
                f"Removed stale checkpoints from previous run: {checkpoints_dir}"
            )

    # Create in-memory Optuna study (no SQLite, no SQLAlchemy)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed study with previous results so TPE retains all prior knowledge
    for prev in previous_trials:
        if prev.get("status") != "completed":
            continue
        frozen = optuna.trial.create_trial(
            params=prev["params"],
            distributions=distributions,
            value=prev["value"],
        )
        study.add_trial(frozen)

    click.echo(f"Study seeded with {len(study.trials)} completed prior trials.")

    # Enqueue user-specified starting points — these run as the first trial(s)
    # before TPE takes over. Only applied on fresh sweeps (not resume).
    starting_params_cfg = OmegaConf.select(cfg, "sweep.starting_params")
    if starting_params_cfg and not resume:
        starting_points = OmegaConf.to_container(starting_params_cfg, resolve=True)
        if isinstance(starting_points, list):
            for point in starting_points:
                study.enqueue_trial(point)
            click.echo(
                f"Enqueued {len(starting_points)} starting point(s) to run first."
            )
        else:
            click.echo(
                "Warning: sweep.starting_params must be a list of dicts — skipping."
            )

    click.echo(f"Running {n_trials} new trials...\n")

    all_trials = list(previous_trials)

    for _ in range(n_trials):
        trial = study.ask()
        try:
            val_loss = _run_trial(cfg, trial, output_dir, params_cfg)
            study.tell(trial, val_loss)
            record = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": val_loss,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            click.echo(f"Trial {trial.number} FAILED: {e}", err=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            record = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": math.inf,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
            }

        all_trials.append(record)
        # Persist after every trial so progress survives crashes
        with open(trials_file, "w") as f:
            json.dump(all_trials, f, indent=2)

    click.echo("\n" + "=" * 60)
    click.echo(f"Sweep complete — {n_trials} new trials done.")
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    if not completed_trials:
        click.echo("No completed trials.")
        click.echo(f"Trials file   : {trials_file}")
        click.echo("=" * 60)
        return

    best = study.best_trial
    click.echo(f"Best val_loss : {best.value:.6f}  (trial #{best.number})")
    click.echo(f"Best params   : {best.params}")
    click.echo(f"Trials file   : {trials_file}")
    click.echo("=" * 60)

    # ── Inference on the best trial's checkpoint ──────────────────────────
    save_checkpoints = bool(
        OmegaConf.select(cfg, "checkpoints.save_checkpoints", default=True)
    )
    if not save_checkpoints:
        click.echo("\n(Skipping inference — checkpoints.save_checkpoints=false)")
    else:
        from hydra import compose

        from simba.workflows.inference import inference as run_inference_wf

        best_trial_dir = output_dir / "checkpoints" / str(best.number)
        inference_out = output_dir / "best_inference"
        inference_out.mkdir(parents=True, exist_ok=True)

        preprocessing_dir = str(
            OmegaConf.select(cfg, "paths.preprocessing_dir_train")
            or OmegaConf.select(cfg, "paths.preprocessing_dir")
        )
        pickle_file = str(
            OmegaConf.select(
                cfg,
                "paths.preprocessing_pickle_file",
                default="mapping_unique_smiles.pkl",
            )
        )
        accelerator = str(OmegaConf.select(cfg, "hardware.accelerator", default="auto"))

        # Read the params that were actually used to train the best trial's.
        best_params_file = best_trial_dir / "params.json"
        if best_params_file.exists():
            with open(best_params_file) as f:
                best_params = json.load(f)
        else:
            click.echo(
                "Warning: params.json not found for best trial, falling back to study params"
            )
            best_params = best.params

        # Build overrides exactly as the standalone `simba inference` command would,
        # plus the best trial's hyperparams so the model architecture matches the checkpoint.
        inf_overrides = [
            f"paths.checkpoint_dir={best_trial_dir}",
            f"paths.preprocessing_dir={preprocessing_dir}",
            f"paths.output_dir={inference_out}",
            f"inference.preprocessing_pickle={pickle_file}",
            f"hardware.accelerator={accelerator}",
            f"inference.accelerator={accelerator}",
            "inference.uniformize_testing=false",
        ]
        for param_name, value in best_params.items():
            inf_overrides.append(f"{param_name}={value}")

        inf_cfg = compose(config_name="config", overrides=inf_overrides)

        click.echo("\n" + "=" * 60)
        click.echo(f"Running inference on best trial (#{best.number}) checkpoint...")
        click.echo(f"  checkpoint : {best_trial_dir}")
        click.echo(f"  output     : {inference_out}")
        click.echo("=" * 60)

        try:
            metrics = run_inference_wf(inf_cfg)
            metrics_file = inference_out / "metrics.json"

            def _to_json(v):
                if hasattr(v, "tolist"):
                    return v.tolist()
                if hasattr(v, "item"):
                    return v.item()
                return v

            with open(metrics_file, "w") as f:
                json.dump({k: _to_json(v) for k, v in metrics.items()}, f, indent=2)
            click.echo("\nInference metrics (best trial):")
            for k, v in metrics.items():
                click.echo(f"  {k}: {v}")
            click.echo(f"\nMetrics saved → {metrics_file}")
            click.echo(f"Plots saved   → {inference_out}")
        except Exception as exc:
            import traceback

            click.echo(f"\nInference failed: {exc}", err=True)
            traceback.print_exc()


if __name__ == "__main__":
    sweep_train()
