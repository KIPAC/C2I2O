import click
import tables_io

from c2i2o import __version__
from c2i2o.generation.c2i_generation import C2IGenerationParams
from c2i2o.utility.yaml_utils import read_yaml_file_as

from . import options


@click.group()
@click.version_option(__version__)
def c2i() -> None:
    """Cosmology to intermediates command line interface"""


@click.group()
@click.version_option(__version__)
def cosmo() -> None:
    """Cosmology command line interface"""


@cosmo.command(name="generate")
@options.cosmo_parameter_file()
@options.config_file()
def generate_cosmo_parameters(
    cosmo_parameter_file: str,
    config_file: str,
) -> None:
    """Generate cosmology parameters from priors"""

    # Read the config file with the priors and cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    # Generate the cosmological paramters
    cosmo_params = c2i_generation_parameters.generate_cosmology_parameters()

    # Write the outputs
    tables_io.write(cosmo_params, cosmo_parameter_file)


@c2i.command(name="generate")
@options.cosmo_parameter_file()
@options.intermediates_file()
@options.config_file()
def cosmology_to_intermediates(
    cosmo_parameter_file: str,
    intermediates_file: str,
    config_file: str,
) -> None:
    """Generate cosmology parameters from priors and compute intermediates"""

    # Read the config file with the priors and cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    cosmo_params, intermediates = c2i_generation_parameters.generate_intermediates()

    # Write the outputs
    tables_io.write(cosmo_params, cosmo_parameter_file)
    tables_io.write(intermediates, intermediates_file)


@c2i.command(name="compute")
@options.cosmo_parameter_file()
@options.intermediates_file()
@options.config_file()
def compute_intermediates(
    cosmo_parameter_file: str,
    intermediates_file: str,
    config_file: str,
) -> None:
    """Read cosmology parameters and compute intermediates"""

    # Read the config file with the cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    # Read the cosmological parameters
    comos_params = tables_io.read(cosmo_parameter_file)
    # Compute the intermediates
    intermediates = c2i_generation_parameters.compute_intermediates(comos_params)

    # Write the outputs
    tables_io.write(intermediates, intermediates_file)
