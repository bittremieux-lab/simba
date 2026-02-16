"""Preprocess command for SIMBA CLI."""

from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


@click.command()
@click.argument("overrides", nargs=-1, type=str)
def preprocess(overrides: tuple[str, ...]):
    """Preprocess MS/MS spectral data for SIMBA training.

    This command converts raw mass spectrometry data (.mgf format) into
    training-ready format by computing structural similarity metrics between
    molecules. The output includes numpy arrays with indexes and distances,
    plus a pickle file mapping spectra to molecular structures (SMILES).

    The preprocessing computes:
    - Edit distance between molecular structures
    - MCES (Maximum Common Edge Substructure) distance
    - Train/validation/test splits

    OVERRIDES: Hydra-style configuration overrides (key=value pairs).

    Examples:

        # Basic preprocessing with default settings
        simba preprocess

        # Override paths
        simba preprocess \\
            paths.spectra_path=data/spectra.mgf \\
            paths.preprocessing_dir=./preprocessed_data

        # Process all spectra with 4 worker processes
        simba preprocess \\
            paths.spectra_path=data/spectra.mgf \\
            preprocessing.max_spectra_train=1000000 \\
            hardware.num_workers=4

        # Custom splits and mapping filename
        simba preprocess \\
            paths.spectra_path=data/spectra.mgf \\
            preprocessing.val_split=0.15 \\
            preprocessing.test_split=0.15 \\
            paths.preprocessing_pickle_file=custom_mapping.pkl
    """
    # Initialize Hydra configuration
    from simba.utils.config_utils import get_config_path

    config_path = get_config_path()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        _preprocess_with_hydra(cfg)


def _preprocess_with_hydra(cfg: DictConfig):
    """Run preprocessing with Hydra configuration.

    This is a thin wrapper that validates CLI inputs and delegates to the workflow.

    Args:
        cfg: Hydra configuration object
    """
    click.echo("Starting SIMBA preprocessing...")

    # Validate required paths
    if not cfg.paths.spectra_path:
        click.echo(
            "❌ Error: spectra_path is required.\n"
            "Please specify the path to your .mgf file:\n"
            "  simba preprocess paths.spectra_path=data/spectra.mgf",
            err=True,
        )
        raise click.Abort()

    if not cfg.paths.preprocessing_dir:
        click.echo(
            "❌ Error: preprocessing_dir is required.\n"
            "Please specify the output directory:\n"
            "  simba preprocess paths.preprocessing_dir=./preprocessed_data",
            err=True,
        )
        raise click.Abort()

    # Check if spectra file exists
    spectra_path = Path(cfg.paths.spectra_path)
    if not spectra_path.exists():
        click.echo(
            f"❌ Error: Spectra file not found: {spectra_path}\n"
            "Please check the path and try again.",
            err=True,
        )
        raise click.Abort()

    click.echo(f"Input spectra: {cfg.paths.spectra_path}")
    click.echo(f"Output workspace: {cfg.paths.preprocessing_dir}")
    click.echo(f"Max training spectra: {cfg.preprocessing.max_spectra_train}")
    click.echo(f"Number of workers: {cfg.preprocessing.num_workers}")

    # Validate splits
    if cfg.preprocessing.val_split + cfg.preprocessing.test_split >= 1.0:
        click.echo(
            "❌ Error: val_split + test_split must be less than 1.0",
            err=True,
        )
        raise click.Abort()

    # Import and run the preprocessing workflow
    from simba.workflows.preprocessing import preprocess as run_preprocessing

    try:
        run_preprocessing(cfg)

        click.echo("\n✅ Preprocessing completed successfully!")
        click.echo(f"Output saved to: {cfg.paths.preprocessing_dir}")
        click.echo(f"Mapping file: {cfg.paths.preprocessing_pickle_file}")
        click.echo("\n📝 Next step: Train a model using 'simba train' command")

    except Exception as e:
        click.echo(f"\n❌ Error during preprocessing: {e}", err=True)
        raise click.Abort() from e
