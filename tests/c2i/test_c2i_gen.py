import os

import tables_io

from c2i2o.generation.c2i_generation import C2IGenerationParams
from c2i2o.utility.yaml_utils import read_yaml_file_as


def test_c2i_generation() -> None:
    """Generate both cosmology parameters and intermediates"""

    config_file: str = "tests/c2i/c2i.yaml"

    cosmo_parameter_file: str = "tests/c2i/cosmo.hdf5"
    intermediates_file: str = "tests/c2i/intermed.hdf5"

    # Read the config file with the cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    # Generate the cosmology parameters and intermediates
    cosmo_params, intermediates = c2i_generation_parameters.generate_intermediates()

    # Write the outputs
    tables_io.write(cosmo_params, cosmo_parameter_file)
    tables_io.write(intermediates, intermediates_file)

    os.unlink(cosmo_parameter_file)
    os.unlink(intermediates_file)


def test_c2i_generation_v2() -> None:
    """Generate both cosmology parameters and intermediates
    but in two steps
    """

    config_file: str = "tests/c2i/c2i.yaml"

    cosmo_parameter_file: str = "tests/c2i/cosmo.hdf5"

    # Read the config file with the cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    # Generate the cosmological parameters
    cosmo_params = c2i_generation_parameters.generate_cosmology_parameters()

    # Write the outputs
    tables_io.write(cosmo_params, cosmo_parameter_file)

    # read the inputs
    cosmo_params_read = tables_io.read(cosmo_parameter_file)

    # Compute the intermediates
    _intermediates = c2i_generation_parameters.compute_intermediates(cosmo_params_read)

    os.unlink(cosmo_parameter_file)


def test_cosmo_generation() -> None:
    """Generate both cosmology parameters"""

    config_file: str = "tests/c2i/c2i.yaml"

    cosmo_parameter_file: str = "tests/c2i/cosmo.hdf5"

    # Read the config file with the cosmology configuration
    c2i_generation_parameters = read_yaml_file_as(C2IGenerationParams, config_file)
    # Read the cosmological parameters
    cosmo_params = c2i_generation_parameters.generate_cosmology_parameters()

    # Write the outputs
    tables_io.write(cosmo_params, cosmo_parameter_file)

    os.unlink(cosmo_parameter_file)
