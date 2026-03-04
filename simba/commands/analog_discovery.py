"""Analog discovery command for SIMBA CLI."""

from pathlib import Path

import click


@click.command(name="analog-discovery")
@click.option(
    "--model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the trained SIMBA model checkpoint (e.g., best_model.ckpt).",
)
@click.option(
    "--query-spectra",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to query spectra file (.mgf or .pkl format).",
)
@click.option(
    "--reference-spectra",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to reference library spectra file (.mgf or .pkl format).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to save analog discovery results and plots.",
)
@click.argument("overrides", nargs=-1, type=str)
def analog_discovery(
    model_path: Path,
    query_spectra: Path,
    reference_spectra: Path,
    output_dir: Path,
    overrides: tuple[str, ...],
) -> None:
    """Find structural analogs in a reference library using SIMBA.

    This command performs analog discovery by:
    1. Loading query and reference spectra
    2. Computing SIMBA predictions (edit distance and MCES)
    3. Ranking matches based on structural similarity
    4. Saving top matches and visualizations

    Configuration is loaded from YAML files and can be overridden via command line.

    Examples:

    \b
    # Basic analog discovery
    simba analog-discovery \\
        --model-path ./models/best_model.ckpt \\
        --query-spectra ./data/casmi2022.mgf \\
        --reference-spectra ./data/massspecgym.mgf \\
        --output-dir ./results

    \b
    # Fast dev mode for testing
    simba analog-discovery analog_discovery=fast_dev \\
        --model-path ./models/best_model.ckpt \\
        --query-spectra ./data/query.mgf \\
        --reference-spectra ./data/ref.mgf \\
        --output-dir ./results

    \b
    # Analyze specific query with custom settings
    simba analog-discovery \\
        --model-path ./models/best_model.ckpt \\
        --query-spectra ./data/query.mgf \\
        --reference-spectra ./data/ref.mgf \\
        --output-dir ./results \\
        analog_discovery.query_index=5 \\
        analog_discovery.top_k=20 \\
        analog_discovery.compute_ground_truth=true

    \b
    # GPU inference with larger batch
    simba analog-discovery \\
        --model-path ./models/best_model.ckpt \\
        --query-spectra ./data/query.mgf \\
        --reference-spectra ./data/ref.mgf \\
        --output-dir ./results \\
        analog_discovery.device=gpu \\
        analog_discovery.batch_size=64
    """
    from hydra import compose, initialize_config_dir

    from simba.utils.config_utils import get_config_path

    # Initialize Hydra configuration
    config_path = get_config_path()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        _analog_discovery_with_hydra(
            cfg, model_path, query_spectra, reference_spectra, output_dir
        )


def _analog_discovery_with_hydra(
    cfg,
    model_path: Path,
    query_spectra: Path,
    reference_spectra: Path,
    output_dir: Path,
) -> None:
    """Run analog discovery with Hydra configuration."""
    from omegaconf import OmegaConf

    # Validate paths
    model_path = model_path.resolve()
    query_spectra = query_spectra.resolve()
    reference_spectra = reference_spectra.resolve()
    output_dir = output_dir.resolve()

    if not model_path.exists():
        click.echo(f"❌ Error: Model file not found: {model_path}", err=True)
        raise click.Abort()

    if not query_spectra.exists():
        click.echo(f"❌ Error: Query spectra file not found: {query_spectra}", err=True)
        raise click.Abort()

    if not reference_spectra.exists():
        click.echo(
            f"❌ Error: Reference spectra file not found: {reference_spectra}", err=True
        )
        raise click.Abort()

    # Add paths to config dynamically
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.paths.model_path = str(model_path)
    cfg.paths.query_spectra = str(query_spectra)
    cfg.paths.reference_spectra = str(reference_spectra)
    cfg.paths.output_dir = str(output_dir)
    OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    click.echo("=" * 70)
    click.echo("SIMBA Analog Discovery")
    click.echo("=" * 70)

    # Print configuration
    click.echo(f"\n📂 Loading query spectra from: {query_spectra}")
    click.echo(f"\n📂 Loading reference spectra from: {reference_spectra}")

    # Import and run workflow
    from simba.workflows.analog_discovery import run_analog_discovery

    try:
        # Run workflow
        result = run_analog_discovery(cfg)

        # Print summary statistics
        stats = result["statistics"]
        click.echo("\n" + "=" * 70)
        click.echo("SUMMARY STATISTICS")
        click.echo("=" * 70)
        click.echo(f"\nTotal queries processed: {stats['total_queries']}")
        click.echo(f"Total matches found: {stats['total_matches']}")

        click.echo("\nRanking scores:")
        click.echo(f"  Mean: {stats['ranking_scores']['mean']:.4f}")
        click.echo(f"  Median: {stats['ranking_scores']['median']:.4f}")
        click.echo(f"  Min: {stats['ranking_scores']['min']:.4f}")
        click.echo(f"  Max: {stats['ranking_scores']['max']:.4f}")

        click.echo("\nPredicted Edit Distance:")
        click.echo(f"  Mean: {stats['predicted_edit_distance']['mean']:.2f}")
        click.echo(f"  Median: {stats['predicted_edit_distance']['median']:.2f}")

        click.echo("\nPredicted MCES Distance:")
        click.echo(f"  Mean: {stats['predicted_mces_distance']['mean']:.2f}")
        click.echo(f"  Median: {stats['predicted_mces_distance']['median']:.2f}")

        if "ground_truth_edit_distance" in stats:
            click.echo("\nGround Truth Edit Distance:")
            click.echo(f"  Mean: {stats['ground_truth_edit_distance']['mean']:.2f}")
            click.echo(f"  Median: {stats['ground_truth_edit_distance']['median']:.2f}")

            click.echo("\nGround Truth MCES Distance:")
            click.echo(f"  Mean: {stats['ground_truth_mces_distance']['mean']:.2f}")
            click.echo(f"  Median: {stats['ground_truth_mces_distance']['median']:.2f}")

        click.echo("\n" + "=" * 70)
        click.echo("✓ Analog discovery complete!")
        click.echo(f"📁 All results saved to: {output_dir}")
        click.echo("=" * 70)

    except Exception as e:
        import traceback
        click.echo(f"\n❌ Error during analog discovery: {e}", err=True)
        click.echo("\nFull traceback:", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.Abort() from e
