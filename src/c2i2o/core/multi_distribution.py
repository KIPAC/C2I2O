"""Multi-dimensional probability distributions for c2i2o.

This module provides classes for multi-dimensional probability distributions
with support for correlations via covariance matrices. These are useful for
modeling correlated cosmological parameters or uncertainties.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy import linalg, stats


class MultiDistributionBase(BaseModel, ABC):
    """Abstract base class for multi-dimensional probability distributions.

    This class provides common functionality for multi-variate distributions,
    including parameter validation and covariance matrix handling.

    Attributes
    ----------
    dist_type
        String identifier for the distribution type.
    mean
        Mean values for each dimension.
    cov
        Covariance matrix (n_dim × n_dim). Must be symmetric and positive definite.
    param_names
        Optional names for each parameter/dimension.

    Notes
    -----
    Subclasses must implement sample() and log_prob() methods.
    The covariance matrix is validated to ensure it is symmetric and
    positive definite.
    """

    dist_type: str = Field(..., description="Type identifier for the distribution")
    mean: np.ndarray = Field(..., description="Mean values (n_dim,)")
    cov: np.ndarray = Field(..., description="Covariance matrix (n_dim, n_dim)")
    param_names: list[str] | None = Field(default=None, description="Optional names for each parameter")

    @field_validator("mean")
    @classmethod
    def validate_mean_1d(cls, v: np.ndarray) -> np.ndarray:
        """Validate that mean is a 1D array."""
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError(f"Mean must be 1D array, got shape {v.shape}")
        return v

    @field_validator("cov")
    @classmethod
    def validate_cov_matrix(cls, v: np.ndarray) -> np.ndarray:
        """Validate that covariance matrix is 2D, symmetric, and positive definite."""
        v = np.asarray(v)

        # Check 2D
        if v.ndim != 2:
            raise ValueError(f"Covariance must be 2D array, got shape {v.shape}")

        # Check square
        if v.shape[0] != v.shape[1]:
            raise ValueError(f"Covariance must be square, got shape {v.shape}")

        # Check symmetric
        if not np.allclose(v, v.T):
            raise ValueError("Covariance matrix must be symmetric")

        # Check positive definite by attempting Cholesky decomposition
        try:
            linalg.cholesky(v, lower=True)
        except linalg.LinAlgError as e:
            raise ValueError("Covariance matrix must be positive definite") from e

        return v

    @field_validator("param_names")
    @classmethod
    def validate_param_names_length(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that param_names length matches dimensions if provided."""
        if v is not None and "mean" in info.data:
            mean = np.asarray(info.data["mean"])
            if len(v) != len(mean):
                raise ValueError(
                    f"Number of param_names ({len(v)}) must match " f"number of dimensions ({len(mean)})"
                )
        return v

    @property
    def n_dim(self) -> int:
        """Number of dimensions.

        Returns
        -------
            Number of dimensions in the distribution.
        """
        return len(self.mean)

    @property
    def std(self) -> np.ndarray:
        """Standard deviations for each dimension.

        Returns
        -------
            Array of standard deviations (square root of diagonal of covariance).
        """
        return cast(np.ndarray, np.sqrt(np.diag(self.cov)))

    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix.

        Returns
        -------
            Correlation matrix derived from covariance matrix.
        """
        std = self.std
        return self.cov / np.outer(std, std)

    @abstractmethod
    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional sampling parameters.

        Returns
        -------
            Array of samples with shape (n_samples, n_dim).
        """
        ...

    @abstractmethod
    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
            Shape should be (n_points, n_dim).
        **kwargs
            Additional parameters.

        Returns
        -------
            Log probability density values with shape (n_points,).
        """
        ...

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class MultiGauss(MultiDistributionBase):
    """Multi-dimensional Gaussian (normal) distribution.

    This class implements a multi-variate normal distribution with arbitrary
    covariance structure, allowing for correlated parameters.

    Parameters
    ----------
    dist_type
        Must be "multi_gauss".
    mean
        Mean values for each dimension (n_dim,).
    cov
        Covariance matrix (n_dim, n_dim).
    param_names
        Optional names for each parameter.

    Examples
    --------
    >>> # 2D Gaussian with correlation
    >>> mean = np.array([0.3, 0.8])
    >>> cov = np.array([[0.01, 0.005],
    ...                 [0.005, 0.02]])
    >>> dist = MultiGauss(mean=mean, cov=cov,
    ...                   param_names=["omega_m", "sigma_8"])
    >>> samples = dist.sample(1000, random_state=42)
    >>> samples.shape
    (1000, 2)
    >>> log_p = dist.log_prob(samples)
    >>> log_p.shape
    (1000,)

    Notes
    -----
    Uses scipy.stats.multivariate_normal for sampling and probability
    calculations.
    """

    dist_type: Literal["multi_gauss"] = "multi_gauss"

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the multivariate Gaussian distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Array of samples with shape (n_samples, n_dim).

        Examples
        --------
        >>> mean = np.array([0.0, 0.0])
        >>> cov = np.eye(2)
        >>> dist = MultiGauss(mean=mean, cov=cov)
        >>> samples = dist.sample(100, random_state=42)
        >>> samples.shape
        (100, 2)
        """
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(self.mean, self.cov, size=n_samples)

    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Log probability density values. Shape is (n_points,) or scalar.

        Examples
        --------
        >>> mean = np.array([0.0, 0.0])
        >>> cov = np.eye(2)
        >>> dist = MultiGauss(mean=mean, cov=cov)
        >>> x = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> log_p = dist.log_prob(x)
        >>> log_p.shape
        (2,)
        """
        x = np.asarray(x)
        return cast(np.ndarray, stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov))

    def prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute probability density.

        Parameters
        ----------
        x
            Values at which to evaluate probability.
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Probability density values. Shape is (n_points,) or scalar.
        """
        x = np.asarray(x)
        return cast(np.ndarray, stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov))


class MultiLogNormal(MultiDistributionBase):
    """Multi-dimensional log-normal distribution.

    This class implements a multi-variate log-normal distribution where the
    logarithm of the variables follows a multi-variate normal distribution.
    Useful for parameters that are positive and multiplicative.

    Parameters
    ----------
    dist_type
        Must be "multi_lognormal".
    mean
        Mean values in log-space for each dimension (n_dim,).
    cov
        Covariance matrix in log-space (n_dim, n_dim).
    param_names
        Optional names for each parameter.

    Examples
    --------
    >>> # 2D log-normal with correlation
    >>> mean_log = np.array([0.0, 0.0])  # exp(0) = 1.0 in real space
    >>> cov_log = np.array([[0.1, 0.05],
    ...                     [0.05, 0.2]])
    >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log,
    ...                       param_names=["A_s", "n_s"])
    >>> samples = dist.sample(1000, random_state=42)
    >>> # Samples are positive
    >>> assert np.all(samples > 0)

    Notes
    -----
    The mean and covariance are specified in log-space. The actual samples
    returned are in real space (exponential of the underlying Gaussian).
    """

    dist_type: Literal["multi_lognormal"] = "multi_lognormal"

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the multivariate log-normal distribution.

        Samples are drawn from the underlying Gaussian in log-space, then
        exponentiated to give positive values.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Array of positive samples with shape (n_samples, n_dim).

        Examples
        --------
        >>> mean_log = np.array([0.0, 0.0])
        >>> cov_log = np.eye(2) * 0.1
        >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        >>> samples = dist.sample(100, random_state=42)
        >>> samples.shape
        (100, 2)
        >>> np.all(samples > 0)
        True
        """
        rng = np.random.default_rng(random_state)
        # Sample from underlying Gaussian in log-space
        log_samples = rng.multivariate_normal(self.mean, self.cov, size=n_samples)
        # Exponentiate to get real-space samples
        return np.exp(log_samples)

    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability (must be positive).
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Log probability density values. Shape is (n_points,) or scalar.
            Returns -inf for non-positive values.

        Examples
        --------
        >>> mean_log = np.array([0.0, 0.0])
        >>> cov_log = np.eye(2) * 0.1
        >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        >>> x = np.array([[1.0, 1.0], [2.0, 2.0]])
        >>> log_p = dist.log_prob(x)
        >>> log_p.shape
        (2,)
        """
        x = np.asarray(x)

        # Check for non-positive values
        if np.any(x <= 0):
            # Return -inf for invalid values
            result = np.full(x.shape[:-1] if x.ndim > 1 else (), -np.inf)
            return result

        # Transform to log-space
        log_x = np.log(x)

        # Log probability from underlying Gaussian
        log_p_gaussian = stats.multivariate_normal.logpdf(log_x, mean=self.mean, cov=self.cov)

        # Jacobian correction: product of 1/x_i for all dimensions
        # log|J| = -sum(log(x_i))
        if x.ndim == 1:
            log_jacobian = -np.sum(np.log(x))
        else:
            log_jacobian = -np.sum(np.log(x), axis=1)

        return cast(np.ndarray, log_p_gaussian + log_jacobian)

    def prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute probability density.

        Parameters
        ----------
        x
            Values at which to evaluate probability (must be positive).
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Probability density values. Shape is (n_points,) or scalar.
            Returns 0 for non-positive values.
        """
        return cast(np.ndarray, np.exp(self.log_prob(x, **kwargs)))

    def mean_real_space(self) -> np.ndarray:
        """Compute mean in real space.

        For log-normal distributions, the mean in real space is:
        E[X] = exp(μ + σ²/2)

        Returns
        -------
            Mean values in real space (n_dim,).
        """
        return np.exp(self.mean + np.diag(self.cov) / 2.0)

    def variance_real_space(self) -> np.ndarray:
        """Compute variance in real space for each dimension.

        For log-normal distributions, the variance in real space is:
        Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)

        Returns
        -------
            Variance values in real space (n_dim,).

        Notes
        -----
        This returns the marginal variances. For full covariance in real space,
        the transformation is more complex.
        """
        var_log = np.diag(self.cov)
        return cast(np.ndarray, (np.exp(var_log) - 1.0) * np.exp(2.0 * self.mean + var_log))


__all__ = [
    "MultiDistributionBase",
    "MultiGauss",
    "MultiLogNormal",
]
