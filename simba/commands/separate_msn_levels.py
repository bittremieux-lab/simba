"""Inference command for SIMBA CLI."""

import sys
from pathlib import Path

import click
from hydra import compose, initialize_config_dir

import simba


# Fix import paths for old checkpoints
sys.modules["src"] = simba


@click.command(name="separate_msn_levels")
@click.argument("overrides", nargs=-1)
def separate_msn_levels(overrides: tuple[str, ...]) -> None:
    """Run inference on test data using a trained SIMBA model.

    Configuration is loaded from YAML files with Hydra overrides.
    Required paths can be specified via command line using Hydra syntax.

    Examples:

    \b
    # Basic inference with Hydra config overrides
    simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./preprocessed_data

    \b
    # Use last checkpoint instead of best model
    simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./data \\
        inference.use_last_model=true

    \b
    # Fast dev inference with smaller batch size
    simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./data \\
        inference=fast_dev

    \b
    # Custom batch size and no uniformization
    simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./data \\
        inference.batch_size=32 inference.uniformize_during_testing=0
    """
    import os
    import platform

    from simba.utils.config_utils import get_config_path
    from simba.utils.logger_setup import logger

    # from simba.workflows.inference import inference as run_inference
    from simba.workflows.separate_msn_levels import (
        separate_msn_levels as separate_msn_levels,
    )

    # Enable MPS fallback on macOS for unsupported ops
    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    click.echo("Loading configuration...")

    # Load Hydra config with CLI overrides
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=list(overrides))

    # Validate required paths
    checkpoint_dir = cfg.paths.checkpoint_dir
    preprocessing_dir = cfg.paths.preprocessing_dir_train or cfg.paths.preprocessing_dir

    if not checkpoint_dir:
        raise click.UsageError(
            "❌ Error: Checkpoint directory is required.\n"
            "Please specify: paths.checkpoint_dir=<path>\n"
            "  simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./data"
        )

    if not preprocessing_dir:
        raise click.UsageError(
            "❌ Error: Preprocessing directory is required.\n"
            "Please specify either paths.preprocessing_dir or paths.preprocessing_dir_train:\n"
            "  simba inference paths.checkpoint_dir=./checkpoints paths.preprocessing_dir=./data"
        )

    checkpoint_dir = Path(checkpoint_dir).resolve()
    preprocessing_dir = Path(preprocessing_dir).resolve()

    if not checkpoint_dir.exists():
        raise click.UsageError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not preprocessing_dir.exists():
        raise click.UsageError(
            f"Preprocessing directory not found: {preprocessing_dir}"
        )

    # Check for model checkpoint
    best_model_path = checkpoint_dir / "best_model.ckpt"
    last_model_path = checkpoint_dir / "last.ckpt"

    if not best_model_path.exists() and not last_model_path.exists():
        raise click.UsageError(
            f"No model checkpoint found in {checkpoint_dir}\n"
            f"Expected either 'best_model.ckpt' or 'last.ckpt'"
        )

    # Set output directory
    output_dir = cfg.paths.get("output_dir", None)
    if output_dir:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = checkpoint_dir

    click.echo(f"Checkpoint directory: {checkpoint_dir}")
    click.echo(f"Preprocessing data: {preprocessing_dir}")
    click.echo(f"Output directory: {output_dir}")

    # Validate dataset exists
    dataset_path = Path(preprocessing_dir) / cfg.inference.preprocessing_pickle
    if not dataset_path.exists():
        raise click.UsageError(
            f"Dataset file not found: {dataset_path}\n"
            f"Expected '{cfg.inference.preprocessing_pickle}' in preprocessing directory.\n"
            f"You can override with: inference.preprocessing_pickle=<filename>"
        )

    click.echo(f"\nDataset: {dataset_path}")
    click.echo(f"Batch size: {cfg.inference.batch_size}")
    click.echo(f"Accelerator: {cfg.hardware.accelerator}")
    click.echo(f"Use last model: {cfg.inference.use_last_model}")
    click.echo(f"Uniformize testing: {cfg.inference.uniformize_testing}\n")

    try:
        # Run inference workflow
        separate_msn_levels(cfg)

    except Exception as e:
        logger.error(f"Separa msn levels failed: {e}")
        raise click.ClickException(str(e)) from e
