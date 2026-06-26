"""Train command for SIMBA CLI."""

from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from simba.utils.config_utils import get_config_path, get_model_paths


@click.command()
@click.argument("overrides", nargs=-1)
def train(overrides: tuple[str, ...]) -> None:
    """Train a SIMBA model on MS/MS spectral data.

    Configuration is loaded from YAML files (configs/config.yaml).
    You can override any parameter via command line using Hydra syntax.

    Examples:

    \b
    # Basic training with default config
    simba train

    \b
    # Override specific parameters
    simba train training.training.epochs=50 hardware.accelerator=gpu

    \b
    # Override paths
    simba train paths.preprocessing_dir=/path/to/data

    \b
    # Multiple overrides
    simba train training.training.epochs=100 training.training.batch_size=64
    """
    click.echo("Loading configuration...")

    # Load Hydra config with CLI overrides
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=list(overrides))

    # Call actual training logic
    _train_with_hydra(cfg)


def _train_with_hydra(cfg: DictConfig) -> None:
    """Internal training function using Hydra configuration."""
    # Validate required paths
    preprocessing_dir = cfg.paths.preprocessing_dir_train or cfg.paths.preprocessing_dir
    if not preprocessing_dir:
        click.echo(
            "❌ Error: Preprocessing directory is required.\n"
            "Please specify either preprocessing_dir or preprocessing_dir_train:\n"
            "  simba train paths.preprocessing_dir_train=./preprocessed_data\n"
            "or\n"
            "  simba train paths.preprocessing_dir=./preprocessed_data",
            err=True,
        )
        raise click.Abort()

    # Check if preprocessing directory exists
    preprocessing_path = Path(preprocessing_dir)
    if not preprocessing_path.exists():
        click.echo(
            f"❌ Error: Preprocessing directory not found: {preprocessing_path}\n"
            "Please run preprocessing first:\n"
            "  simba preprocess paths.spectra_path=data/spectra.mgf "
            "paths.preprocessing_dir={preprocessing_path}",
            err=True,
        )
        raise click.Abort()

    # Check if preprocessing pickle file exists
    mapping_file = preprocessing_path / cfg.paths.preprocessing_pickle_file
    if not mapping_file.exists():
        click.echo(
            f"❌ Error: Preprocessing file not found: {mapping_file}\n"
            "The preprocessing directory seems incomplete.\n"
            "Please run preprocessing again with paths.overwrite=true",
            err=True,
        )
        raise click.Abort()

    if not cfg.paths.checkpoint_dir:
        click.echo(
            "❌ Error: checkpoint_dir is required.\n"
            "Please specify the output directory for model checkpoints:\n"
            "  simba train paths.checkpoint_dir=./checkpoints",
            err=True,
        )
        raise click.Abort()

    # Get model paths (checkpoint_dir, best_model_path, pretrained_path)
    paths = get_model_paths(cfg)

    click.echo("Starting SIMBA training...")
    click.echo(f"Checkpoint directory: {paths['checkpoint_dir']}")
    click.echo(f"Preprocessing directory: {preprocessing_dir}")
    click.echo(f"Epochs: {cfg.training.epochs}")
    click.echo(f"Accelerator: {cfg.hardware.accelerator}")
    click.echo(f"Batch size: {cfg.training.batch_size}")

    # Import training functions from workflows module
    from simba.workflows.training import (
        create_dataloaders,
        load_dataset,
        prepare_data,
        setup_callbacks,
        setup_model,
    )
    from simba.workflows.training import (
        train as run_training,
    )

    try:
        # Create checkpoint directory
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

        # Load dataset
        click.echo("Loading dataset...")
        (
            molecule_pairs_train,
            molecule_pairs_val,
            molecule_pairs_test,
            uniformed_molecule_pairs_test,
        ) = load_dataset(cfg)

        # Prepare training data
        click.echo("Preparing training data...")

        (
            dataset_train,
            train_sampler,
            dataset_val,
            val_sampler,
            weights_ed,
            bins_ed,
        ) = prepare_data(
            molecule_pairs_train,
            molecule_pairs_val,
            molecule_pairs_test,
            uniformed_molecule_pairs_test,
            cfg,
        )

        # Create dataloaders
        dataloader_train, dataloader_val = create_dataloaders(
            cfg, dataset_train, train_sampler, dataset_val, val_sampler
        )

        # Check training dataset size and adjust val_check_interval if needed
        num_train_batches = len(dataloader_train)

        if num_train_batches == 0:
            raise click.ClickException(
                "No training batches found in the preprocessed dataset. "
                "The dataset may be empty or all samples were filtered out. "
                "Please check your preprocessing outputs and ensure the dataset contains valid training pairs."
            )

        if num_train_batches < cfg.training.val_check_interval:
            click.echo(
                f"Debug: Original val_check_interval={cfg.training.val_check_interval}, "
                f"num_train_batches={num_train_batches}"
            )
            cfg.training.val_check_interval = max(1, num_train_batches // 2)
            click.echo(
                f"Adjusted val_check_interval to {cfg.training.val_check_interval} "
                f"(training dataset has only {num_train_batches} batches)"
            )

        # Setup model and callbacks
        click.echo("Initializing model...")
        checkpoint_callback, checkpoint_n_steps_callback, losscallback = (
            setup_callbacks(cfg)
        )

        # Get weights for MCES from first 100 batches (same as original script)
        click.echo("Computing MCES weights from training data...")

        import itertools

        import numpy as np

        from simba.core.training.train_utils import TrainUtils

        mces_sampled = []
        for batch in itertools.islice(dataloader_train, 100):
            mces_sampled = mces_sampled + list(batch["mces"].reshape(-1))

        mces_sampled = np.array(mces_sampled)
        counting_mces, bins_mces = TrainUtils.count_ranges(
            mces_sampled,
            number_bins=5,
            bin_sim_1=False,
            max_value=1,
        )

        # Calculate weights directly (same as original script)
        weights_mces = np.array(
            [np.sum(counting_mces) / c if c != 0 else 0 for c in counting_mces]
        )
        weights_mces = weights_mces / np.sum(weights_mces)

        model = setup_model(cfg, weights_mces)

        # Train model
        click.echo(f"Starting training for {cfg.training.epochs} epochs...")
        run_training(
            model,
            dataloader_train,
            dataloader_val,
            cfg,
            checkpoint_callback,
            checkpoint_n_steps_callback,
            losscallback,
        )

        click.echo("Training completed successfully!")
        click.echo(f"Model saved to: {paths['checkpoint_dir']}")

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        raise
