"""Molecular networking command for SIMBA CLI."""

from pathlib import Path

import click


@click.command(name="molecular-network")
@click.option(
    "--model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the trained SIMBA model checkpoint (e.g., best_model.ckpt).",
)
@click.option(
    "--input-spectra",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the input spectra file (.mgf format).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to save the network and intermediate files.",
)
@click.argument("overrides", nargs=-1, type=str)
def molecular_network(
    model_path: Path,
    input_spectra: Path,
    output_dir: Path,
    overrides: tuple[str, ...],
) -> None:
    """Build a molecular network from a single set of spectra using SIMBA.

    Computes all-vs-all pairwise SIMBA predictions, converts predicted MCES
    distances to normalized similarities (sim = 1 - mces / max_mces), and
    outputs a spectral network compatible with Cytoscape and matchms.

    Input spectra are assumed to be deduplicated (one spectrum per compound).
    Duplicate spectra from the same compound will appear as separate nodes.

    Output files:
    \b
      molecular_network.<format>   — spectral network (default: graphml)
      similarity_mces.npy          — raw MCES distance matrix [N x N], reusable
                                     via molecular_network.precomputed_mces=<path>
      network_plot.png             — visualisation (only when molecular_network.plot=true)

    Pass ``molecular_network.precomputed_mces=<path>`` to reuse a previously
    saved ``similarity_mces.npy`` and skip model inference entirely.

    Examples:

    \b
    # Basic molecular networking
    simba molecular-network \\
        --model-path ./models/best_model.ckpt \\
        --input-spectra ./data/query.mgf \\
        --output-dir ./network_output

    \b
    # Custom score cutoff
    simba molecular-network \\
        --model-path ./models/best_model.ckpt \\
        --input-spectra ./data/query.mgf \\
        --output-dir ./network_output \\
        molecular_network.score_cutoff=0.6

    \b
    # Export as GEXF (e.g. for Gephi)
    simba molecular-network \\
        --model-path ./models/best_model.ckpt \\
        --input-spectra ./data/query.mgf \\
        --output-dir ./network_output \\
        molecular_network.graph_format=gexf
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from simba.utils.config_utils import get_config_path

    config_path = get_config_path()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))

        model_path = model_path.resolve()
        input_spectra = input_spectra.resolve()
        output_dir = output_dir.resolve()

        OmegaConf.set_struct(cfg, False)
        cfg.paths.model_path = str(model_path)
        cfg.paths.input_spectra = str(input_spectra)
        cfg.paths.output_dir = str(output_dir)
        OmegaConf.set_struct(cfg, True)

        click.echo("=" * 70)
        click.echo("SIMBA Molecular Networking")
        click.echo("=" * 70)
        click.echo(f"\nInput spectra : {input_spectra}")
        click.echo(f"Model         : {model_path}")
        click.echo(f"Output dir    : {output_dir}")

        from simba.workflows.molecular_networking import run_molecular_networking

        try:
            result = run_molecular_networking(cfg)
        except Exception as exc:
            click.echo(f"\nError: {exc}", err=True)
            raise click.Abort() from exc

        click.echo("\n" + "=" * 70)
        click.echo("DONE")
        click.echo("=" * 70)
        click.echo(f"Nodes : {result['n_nodes']}")
        click.echo(f"Edges : {result['n_edges']}")
        click.echo(f"Network file : {result['output_file']}")
