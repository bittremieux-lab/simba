"""SIMBA Command Line Interface."""

import click

from simba.commands.analog_discovery import analog_discovery
from simba.commands.inference import inference
from simba.commands.molecular_networking import molecular_network
from simba.commands.preprocess import preprocess
from simba.commands.train import train
from simba.commands.metadata_analysis import metadata_analysis
from simba.commands.separate_msn_levels import separate_msn_levels
@click.group()
@click.version_option(package_name="simba-ms")
def cli():
    """SIMBA: Spectral Identification of Molecule Bio-Analogues.

    A transformer-based neural network for predicting chemical structural
    similarity from tandem mass spectrometry (MS/MS) spectra.
    """
    pass


# Register commands
cli.add_command(analog_discovery)
cli.add_command(inference)
cli.add_command(molecular_network)
cli.add_command(preprocess)
cli.add_command(train)
cli.add_command(metadata_analysis)
cli.add_command(separate_msn_levels)


if __name__ == "__main__":
    cli()
