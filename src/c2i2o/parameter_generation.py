"""Parameter generation for c2i2o.

This module provides utilities for generating parameter samples from
combined univariate and multivariate distributions, with support for
YAML configuration and HDF5 output.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tables_io
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from c2i2o.core.multi_distribution import MultiDistributionSet
from c2i2o.core.parameter_space import ParameterSpace


class ParameterGenerator(BaseModel):
    """Generator for cosmological parameter samples.

    This class combines univariate distributions (ParameterSpace) and
    multivariate distributions (MultiDistributionSet) to generate parameter
    samples. It supports scaling of distribution widths and can save
    configurations to YAML and samples to HDF5.

    Attributes
    ----------
    num_samples
        Number of samples to generate.
    scale_factor
        Universal scaling factor applied to all distribution widths.
        Values > 1 increase uncertainty, < 1 decrease it.
    parameter_space
        Univariate parameter distributions.
    multi_distribution_set
        Multivariate parameter distributions with correlations.

    Examples
    --------
    >>> # Create parameter generator
    >>> param_space = ParameterSpace(
    ...     parameters={
    ...         "n_s": Norm(loc=0.96, scale=0.01),
    ...     }
    ... )
    >>> multi_dist = MultiDistributionSet(
    ...     distributions=[
    ...         MultiGauss(
    ...             mean=np.array([0.3, 0.8]),
    ...             cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
    ...             param_names=["omega_m", "sigma_8"]
    ...         )
    ...     ]
    ... )
    >>> generator = ParameterGenerator(
    ...     num_samples=1000,
    ...     scale_factor=1.5,
    ...     parameter_space=param_space,
    ...     multi_distribution_set=multi_dist
    ... )
    >>> samples = generator.generate(random_state=42)
    >>> samples.keys()
    dict_keys(['n_s', 'omega_m', 'sigma_8'])

    Notes
    -----
    The scale_factor is applied by scaling the `scale` parameter for
    univariate distributions and the covariance matrix for multivariate
    distributions.
    """

    num_samples: int = Field(..., gt=0, description="Number of samples to generate")
    scale_factor: float = Field(default=1.0, gt=0, description="Scaling factor for distribution widths")
    parameter_space: ParameterSpace = Field(..., description="Univariate parameter distributions")
    multi_distribution_set: MultiDistributionSet = Field(
        ..., description="Multivariate parameter distributions"
    )

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples_positive(cls, v: int) -> int:
        """Ensure num_samples is positive.

        Parameters
        ----------
        v
            Number of samples to validate.

        Returns
        -------
            Validated number of samples.

        Raises
        ------
        ValueError
            If num_samples is not positive.
        """
        if v <= 0:  # pragma: no cover
            raise ValueError(f"num_samples must be positive, got {v}")
        return v

    @field_validator("scale_factor")
    @classmethod
    def validate_scale_factor_positive(cls, v: float) -> float:
        """Ensure scale_factor is positive.

        Parameters
        ----------
        v
            Scale factor to validate.

        Returns
        -------
            Validated scale factor.

        Raises
        ------
        ValueError
            If scale_factor is not positive.
        """
        if v <= 0:  # pragma: no cover
            raise ValueError(f"scale_factor must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_no_name_collisions(self) -> "ParameterGenerator":
        """Ensure no parameter name collisions between distributions.

        Returns
        -------
            Validated ParameterGenerator instance.

        Raises
        ------
        ValueError
            If any parameter names are duplicated between ParameterSpace
            and MultiDistributionSet.
        """
        # Get parameter names from ParameterSpace
        param_space_names = set(self.parameter_space.parameters.keys())

        # Get parameter names from MultiDistributionSet
        multi_dist_names: set[str] = set()
        for i, dist in enumerate(self.multi_distribution_set.distributions):
            if dist.param_names is not None:
                multi_dist_names.update(dist.param_names)
            else:
                # Add default names
                for j in range(dist.n_dim):
                    multi_dist_names.add(f"dist{i}_param{j}")

        # Check for collisions
        collisions = param_space_names & multi_dist_names
        if collisions:
            raise ValueError(f"Parameter name collision detected between distributions: {collisions}")

        return self

    def _scale_parameter_space(self) -> ParameterSpace:
        """Create a scaled copy of the parameter space.

        Returns
        -------
            ParameterSpace with scaled distribution widths.

        Notes
        -----
        Only distributions with a `scale` parameter are affected.
        FixedDistribution instances are not modified.
        """
        if self.scale_factor == 1.0:
            return self.parameter_space

        # Create new parameters dict with scaled distributions
        scaled_params = {}
        for name, dist in self.parameter_space.parameters.items():
            # Create a copy with scaled parameters
            dist_dict = dist.model_dump()

            # Scale the 'scale' parameter if it exists
            if "scale" in dist_dict and dist.dist_type != "fixed":
                dist_dict["scale"] = dist_dict["scale"] * self.scale_factor

            # Reconstruct the distribution
            scaled_params[name] = type(dist)(**dist_dict)

        return ParameterSpace(parameters=scaled_params)

    def _scale_multi_distribution_set(self) -> MultiDistributionSet:
        """Create a scaled copy of the multi-distribution set.

        Returns
        -------
            MultiDistributionSet with scaled covariance matrices.

        Notes
        -----
        Covariance matrices are scaled by scale_factor^2 to preserve
        the scaling of standard deviations.
        """
        if self.scale_factor == 1.0:
            return self.multi_distribution_set

        # Create new distributions list with scaled covariances
        scaled_dists = []
        for dist in self.multi_distribution_set.distributions:
            dist_dict = dist.model_dump()

            # Scale covariance matrix by scale_factor^2
            dist_dict["cov"] = np.array(dist_dict["cov"]) * (self.scale_factor**2)

            # Reconstruct the distribution
            scaled_dists.append(type(dist)(**dist_dict))

        return MultiDistributionSet(distributions=scaled_dists)

    def generate(self, random_state: int | None = None) -> dict[str, np.ndarray]:
        """Generate parameter samples.

        Parameters
        ----------
        random_state
            Random seed for reproducibility.

        Returns
        -------
            Dictionary mapping parameter names to sample arrays.
            Each array has shape (num_samples,).

        Examples
        --------
        >>> generator = ParameterGenerator(...)
        >>> samples = generator.generate(random_state=42)
        >>> len(samples['omega_m'])
        1000
        """
        # Scale distributions
        scaled_param_space = self._scale_parameter_space()
        scaled_multi_dist = self._scale_multi_distribution_set()

        # Sample from both
        samples_univariate = scaled_param_space.sample(n_samples=self.num_samples, random_state=random_state)
        samples_multivariate = scaled_multi_dist.sample(n_samples=self.num_samples, random_state=random_state)

        # Combine samples
        return {**samples_univariate, **samples_multivariate}

    def to_yaml(self, filepath: str | Path) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        filepath
            Path to output YAML file.

        Examples
        --------
        >>> generator = ParameterGenerator(...)
        >>> generator.to_yaml("config.yaml")

        Notes
        -----
        NumPy arrays are converted to lists for YAML serialization.
        """
        filepath = Path(filepath)

        # Convert to dict, handling NumPy arrays
        data = self.model_dump()

        # Custom YAML representer for NumPy arrays
        def numpy_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
            return dumper.represent_list(data.tolist())

        yaml.add_representer(np.ndarray, numpy_representer)

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "ParameterGenerator":
        """Load configuration from YAML file.

        Parameters
        ----------
        filepath
            Path to input YAML file.

        Returns
        -------
            ParameterGenerator instance.

        Examples
        --------
        >>> generator = ParameterGenerator.from_yaml("config.yaml")
        >>> samples = generator.generate(random_state=42)

        Notes
        -----
        Lists in YAML are automatically converted to NumPy arrays where needed
        by Pydantic validation.
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def generate_to_hdf5(
        self,
        filepath: str | Path,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Generate samples and write directly to HDF5 file.

        Parameters
        ----------
        filepath
            Path to output HDF5 file.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments passed to tables_io.write().

        Examples
        --------
        >>> generator = ParameterGenerator(...)
        >>> generator.generate_to_hdf5("samples.h5", random_state=42)

        Notes
        -----
        Uses tables_io.write() for efficient HDF5 writing.
        The samples are stored as a dictionary of arrays in the specified group.
        """
        filepath = Path(filepath)

        # Generate samples
        samples = self.generate(random_state=random_state)

        # Write to HDF5
        tables_io.write(samples, str(filepath), **kwargs)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"
