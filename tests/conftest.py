"""Shared pytest fixtures for c2i2o tests."""

import numpy as np
import pytest
from c2i2o.core.intermediates import PowerSpectrum
from c2i2o.core.observables import GalaxyClusteringObservable
from c2i2o.core.parameters import CosmologicalParameters, ParameterSpace


@pytest.fixture
def default_cosmology() -> CosmologicalParameters:
    """
    Provide default cosmological parameters.

    Returns
    -------
    CosmologicalParameters
        Fiducial cosmology similar to Planck 2018
    """
    return CosmologicalParameters(
        omega_m=0.315,
        omega_b=0.049,
        h=0.674,
        sigma_8=0.811,
        n_s=0.965,
    )


@pytest.fixture
def parameter_space() -> ParameterSpace:
    """
    Provide a parameter space for testing.

    Returns
    -------
    ParameterSpace
        Parameter space with reasonable bounds
    """
    return ParameterSpace(
        bounds={
            "omega_m": (0.1, 0.5),
            "omega_b": (0.02, 0.08),
            "h": (0.6, 0.8),
            "sigma_8": (0.6, 1.0),
            "n_s": (0.9, 1.1),
        }
    )


@pytest.fixture
def power_spectrum() -> PowerSpectrum:
    """
    Provide a sample power spectrum.

    Returns
    -------
    PowerSpectrum
        Mock power spectrum for testing
    """
    k = np.logspace(-3, 1, 50)
    # Simple power-law power spectrum for testing
    pk = 1000 * (k / 0.05) ** (-2.5)
    return PowerSpectrum(k=k, pk=pk, redshift=0.0)


@pytest.fixture
def galaxy_clustering_data() -> GalaxyClusteringObservable:
    """
    Provide mock galaxy clustering data.

    Returns
    -------
    GalaxyClusteringObservable
        Mock clustering observable
    """
    scales = np.logspace(-1, 2, 20)
    # Mock correlation function with some noise
    rng = np.random.default_rng(42)
    xi = 10 * (scales / 10) ** (-1.8) + rng.normal(0, 0.5, len(scales))
    uncertainty = 0.5 * np.ones_like(xi)

    return GalaxyClusteringObservable(
        scales=scales,
        data=xi,
        uncertainty=uncertainty,
        measurement_type="correlation",
        redshift=0.5,
    )


@pytest.fixture
def random_seed() -> int:
    """Provide a fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """
    Provide a numpy random number generator.

    Parameters
    ----------
    random_seed : int
        Random seed

    Returns
    -------
    np.random.Generator
        Random number generator
    """
    return np.random.default_rng(random_seed)
