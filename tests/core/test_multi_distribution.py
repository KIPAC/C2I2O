"""Tests for c2i2o.core.multi_distribution module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.multi_distribution import (
    MultiDistributionBase,
    MultiGauss,
    MultiLogNormal,
)


class TestMultiDistributionBase:
    """Tests for MultiDistributionBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that MultiDistributionBase cannot be instantiated directly."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        with pytest.raises(TypeError):
            MultiDistributionBase(  # type: ignore
                dist_type="test",
                mean=mean,
                cov=cov,
            )

    def test_mean_must_be_1d(self) -> None:
        """Test that mean must be 1D array."""
        # This would be caught in a concrete subclass
        mean_2d = np.array([[0.0, 0.0], [1.0, 1.0]])
        cov = np.eye(2)

        with pytest.raises(ValidationError, match="must be 1D"):
            MultiGauss(mean=mean_2d, cov=cov)

    def test_cov_must_be_2d(self) -> None:
        """Test that covariance must be 2D array."""
        mean = np.array([0.0, 0.0])
        cov_1d = np.array([1.0, 1.0])

        with pytest.raises(ValidationError, match="must be 2D"):
            MultiGauss(mean=mean, cov=cov_1d)

    def test_cov_must_be_square(self) -> None:
        """Test that covariance must be square matrix."""
        mean = np.array([0.0, 0.0])
        cov_nonsquare = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        with pytest.raises(ValidationError, match="must be square"):
            MultiGauss(mean=mean, cov=cov_nonsquare)

    def test_cov_must_be_symmetric(self) -> None:
        """Test that covariance must be symmetric."""
        mean = np.array([0.0, 0.0])
        cov_asymmetric = np.array([[1.0, 0.5], [0.3, 1.0]])  # Not symmetric

        with pytest.raises(ValidationError, match="must be symmetric"):
            MultiGauss(mean=mean, cov=cov_asymmetric)

    def test_cov_must_be_positive_definite(self) -> None:
        """Test that covariance must be positive definite."""
        mean = np.array([0.0, 0.0])
        cov_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite

        with pytest.raises(ValidationError, match="must be positive definite"):
            MultiGauss(mean=mean, cov=cov_not_pd)

    def test_param_names_length_must_match_dimensions(self) -> None:
        """Test that param_names length must match mean dimension."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        param_names = ["param1"]  # Only 1 name for 2 dimensions

        with pytest.raises(ValidationError, match="must match number of dimensions"):
            MultiGauss(mean=mean, cov=cov, param_names=param_names)


class TestMultiGauss:
    """Tests for MultiGauss distribution."""

    def test_initialization_2d(self) -> None:
        """Test basic 2D initialization."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.dist_type == "multi_gauss"
        np.testing.assert_array_equal(dist.mean, mean)
        np.testing.assert_array_equal(dist.cov, cov)
        assert dist.n_dim == 2

    def test_initialization_with_param_names(self) -> None:
        """Test initialization with parameter names."""
        mean = np.array([0.3, 0.8])
        cov = np.eye(2) * 0.01
        param_names = ["omega_m", "sigma_8"]

        dist = MultiGauss(mean=mean, cov=cov, param_names=param_names)

        assert dist.param_names == param_names

    def test_initialization_with_correlation(self) -> None:
        """Test initialization with correlated parameters."""
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])

        dist = MultiGauss(mean=mean, cov=cov)

        np.testing.assert_array_equal(dist.cov, cov)

    def test_n_dim_property(self) -> None:
        """Test n_dim property."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 3

    def test_std_property(self) -> None:
        """Test std property."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[0.01, 0.0], [0.0, 0.04]])

        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_correlation_property(self) -> None:
        """Test correlation matrix property."""
        mean = np.array([0.0, 0.0])
        std1, std2 = 0.1, 0.2
        corr_12 = 0.5
        cov = np.array([[std1**2, corr_12 * std1 * std2], [corr_12 * std1 * std2, std2**2]])

        dist = MultiGauss(mean=mean, cov=cov)

        expected_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(dist.correlation, expected_corr)

    def test_sample_shape(self) -> None:
        """Test that sample returns correct shape."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(100, random_state=42)

        assert samples.shape == (100, 2)

    def test_sample_reproducible(self) -> None:
        """Test that sampling is reproducible with random_state."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples1 = dist.sample(50, random_state=42)
        samples2 = dist.sample(50, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_statistics(self) -> None:
        """Test that samples have approximately correct statistics."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(10000, random_state=42)

        # Check sample mean
        sample_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_mean, mean, atol=0.05)

        # Check sample covariance
        sample_cov = np.cov(samples.T)
        np.testing.assert_allclose(sample_cov, cov, atol=0.05)

    def test_log_prob_single_point(self) -> None:
        """Test log_prob for single point."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([0.0, 0.0])
        log_p = dist.log_prob(x)

        assert isinstance(log_p, (float, np.floating))

    def test_log_prob_multiple_points(self) -> None:
        """Test log_prob for multiple points."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        log_p = dist.log_prob(x)

        assert log_p.shape == (3,)

    def test_log_prob_at_mean(self) -> None:
        """Test that log_prob is highest at the mean."""
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x_mean = mean
        x_offset = mean + np.array([0.5, 0.5])

        log_p_mean = dist.log_prob(x_mean)
        log_p_offset = dist.log_prob(x_offset)

        assert log_p_mean > log_p_offset

    def test_prob_method(self) -> None:
        """Test prob method."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0]])
        prob = dist.prob(x)

        assert prob.shape == (1,)
        assert prob[0] > 0

    def test_prob_equals_exp_log_prob(self) -> None:
        """Test that prob equals exp(log_prob)."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        prob = dist.prob(x)
        log_prob = dist.log_prob(x)

        np.testing.assert_allclose(prob, np.exp(log_prob))

    def test_high_dimensional_distribution(self) -> None:
        """Test with higher dimensional distribution."""
        n_dim = 5
        mean = np.zeros(n_dim)
        cov = np.eye(n_dim)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 5

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 5)


class TestMultiLogNormal:
    """Tests for MultiLogNormal distribution."""

    def test_initialization_2d(self) -> None:
        """Test basic 2D initialization."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.dist_type == "multi_lognormal"
        np.testing.assert_array_equal(dist.mean, mean_log)
        np.testing.assert_array_equal(dist.cov, cov_log)
        assert dist.n_dim == 2

    def test_initialization_with_param_names(self) -> None:
        """Test initialization with parameter names."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        param_names = ["A_s", "n_s"]

        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=param_names)

        assert dist.param_names == param_names

    def test_sample_shape(self) -> None:
        """Test that sample returns correct shape."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(100, random_state=42)

        assert samples.shape == (100, 2)

    def test_sample_all_positive(self) -> None:
        """Test that all samples are positive."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(1000, random_state=42)

        assert np.all(samples > 0)

    def test_sample_reproducible(self) -> None:
        """Test that sampling is reproducible with random_state."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples1 = dist.sample(50, random_state=42)
        samples2 = dist.sample(50, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_log_prob_single_point(self) -> None:
        """Test log_prob for single point."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([1.0, 1.0])
        log_p = dist.log_prob(x)

        assert isinstance(log_p, (float, np.floating))

    def test_log_prob_multiple_points(self) -> None:
        """Test log_prob for multiple points."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
        log_p = dist.log_prob(x)

        assert log_p.shape == (3,)

    def test_log_prob_negative_values(self) -> None:
        """Test that log_prob returns -inf for negative values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[-1.0, 1.0]])
        log_p = dist.log_prob(x)

        assert log_p[0] == -np.inf

    def test_log_prob_zero_values(self) -> None:
        """Test that log_prob returns -inf for zero values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[0.0, 1.0]])
        log_p = dist.log_prob(x)

        assert log_p[0] == -np.inf

    def test_prob_method(self) -> None:
        """Test prob method."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0]])
        prob = dist.prob(x)

        assert prob.shape == (1,)
        assert prob[0] > 0

    def test_prob_equals_exp_log_prob(self) -> None:
        """Test that prob equals exp(log_prob) for positive values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0], [2.0, 2.0]])
        prob = dist.prob(x)
        log_prob = dist.log_prob(x)

        np.testing.assert_allclose(prob, np.exp(log_prob))

    def test_prob_zero_for_negative_values(self) -> None:
        """Test that prob returns 0 for negative values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[-1.0, 1.0]])
        prob = dist.prob(x)

        assert prob[0] == 0.0

    def test_mean_real_space(self) -> None:
        """Test mean in real space calculation."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        mean_real = dist.mean_real_space()

        # For log-normal: E[X] = exp(μ + σ²/2)
        expected = np.exp(mean_log + np.diag(cov_log) / 2.0)
        np.testing.assert_allclose(mean_real, expected)

    def test_mean_real_space_values(self) -> None:
        """Test mean in real space with known values."""
        mean_log = np.array([0.0, 0.0])
        var_log = 0.1
        cov_log = np.eye(2) * var_log
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        mean_real = dist.mean_real_space()

        # exp(0 + 0.1/2) = exp(0.05) ≈ 1.0513
        expected = np.exp(0.05)
        np.testing.assert_allclose(mean_real, np.array([expected, expected]))

    def test_variance_real_space(self) -> None:
        """Test variance in real space calculation."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        var_real = dist.variance_real_space()

        # For log-normal: Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
        var_log = np.diag(cov_log)
        expected = (np.exp(var_log) - 1.0) * np.exp(2.0 * mean_log + var_log)
        np.testing.assert_allclose(var_real, expected)

    def test_sample_mean_approximates_real_space_mean(self) -> None:
        """Test that sample mean approximates theoretical mean in real space."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.05  # Small variance for better approximation
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(10000, random_state=42)
        sample_mean = np.mean(samples, axis=0)
        theoretical_mean = dist.mean_real_space()

        np.testing.assert_allclose(sample_mean, theoretical_mean, rtol=0.1)

    def test_high_dimensional_lognormal(self) -> None:
        """Test with higher dimensional log-normal distribution."""
        n_dim = 5
        mean_log = np.zeros(n_dim)
        cov_log = np.eye(n_dim) * 0.1

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.n_dim == 5

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 5)
        assert np.all(samples > 0)


class TestMultiDistributionComparison:
    """Tests comparing MultiGauss and MultiLogNormal."""

    def test_lognormal_samples_are_exponential_of_normal(self) -> None:
        """Test that log-normal samples are exp of normal samples."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        # Create distributions
        normal_dist = MultiGauss(mean=mean_log, cov=cov_log)
        lognormal_dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        # Sample with same random state
        normal_samples = normal_dist.sample(100, random_state=42)
        lognormal_samples = lognormal_dist.sample(100, random_state=42)

        # Log-normal samples should be exp of normal samples
        np.testing.assert_allclose(lognormal_samples, np.exp(normal_samples))

    def test_correlation_preserved_in_both_distributions(self) -> None:
        """Test that both distributions preserve correlation structure."""
        mean = np.array([0.0, 0.0])
        corr = 0.7
        std1, std2 = 1.0, 2.0
        cov = np.array([[std1**2, corr * std1 * std2], [corr * std1 * std2, std2**2]])

        gauss_dist = MultiGauss(mean=mean, cov=cov)
        lognorm_dist = MultiLogNormal(mean=mean, cov=cov)

        # Both should have same correlation matrix
        np.testing.assert_allclose(gauss_dist.correlation, lognorm_dist.correlation)


class TestMultiDistributionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_dimensional_gauss(self) -> None:
        """Test 1D MultiGauss distribution."""
        mean = np.array([0.0])
        cov = np.array([[1.0]])

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 1
        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 1)

    def test_single_dimensional_lognormal(self) -> None:
        """Test 1D MultiLogNormal distribution."""
        mean_log = np.array([0.0])
        cov_log = np.array([[0.1]])

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.n_dim == 1
        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 1)
        assert np.all(samples > 0)

    def test_uncorrelated_gauss(self) -> None:
        """Test MultiGauss with diagonal covariance (uncorrelated)."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.diag([0.1, 0.2, 0.3])

        dist = MultiGauss(mean=mean, cov=cov)

        # Correlation matrix should be identity
        np.testing.assert_allclose(dist.correlation, np.eye(3))

    def test_highly_correlated_gauss(self) -> None:
        """Test MultiGauss with high correlation."""
        mean = np.array([0.0, 0.0])
        corr = 0.99
        cov = np.array([[1.0, corr], [corr, 1.0]])

        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(1000, random_state=42)
        sample_corr = np.corrcoef(samples.T)

        np.testing.assert_allclose(sample_corr[0, 1], corr, atol=0.05)

    def test_large_variance_lognormal(self) -> None:
        """Test MultiLogNormal with large variance in log-space."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 2.0  # Large variance

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(1000, random_state=42)

        # Should still be all positive
        assert np.all(samples > 0)
        # Should have wide spread
        assert np.max(samples) / np.min(samples) > 10


class TestMultiDistributionSerialization:
    """Tests for serialization of multi-dimensional distributions."""

    def test_gauss_serialization(self) -> None:
        """Test MultiGauss serialization round-trip."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])
        param_names = ["param1", "param2"]

        dist = MultiGauss(mean=mean, cov=cov, param_names=param_names)

        # Serialize
        data = dist.model_dump()

        assert data["dist_type"] == "multi_gauss"
        np.testing.assert_array_equal(data["mean"], mean)
        np.testing.assert_array_equal(data["cov"], cov)
        assert data["param_names"] == param_names

    def test_lognormal_serialization(self) -> None:
        """Test MultiLogNormal serialization round-trip."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        param_names = ["A_s", "n_s"]

        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=param_names)

        # Serialize
        data = dist.model_dump()

        assert data["dist_type"] == "multi_lognormal"
        np.testing.assert_array_equal(data["mean"], mean_log)
        np.testing.assert_array_equal(data["cov"], cov_log)
        assert data["param_names"] == param_names

    def test_gauss_deserialization(self) -> None:
        """Test reconstructing MultiGauss from serialized data."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])

        dist_original = MultiGauss(mean=mean, cov=cov)
        data = dist_original.model_dump()

        # Reconstruct
        dist_reconstructed = MultiGauss(**data)

        np.testing.assert_array_equal(dist_reconstructed.mean, dist_original.mean)
        np.testing.assert_array_equal(dist_reconstructed.cov, dist_original.cov)

    def test_lognormal_deserialization(self) -> None:
        """Test reconstructing MultiLogNormal from serialized data."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        dist_original = MultiLogNormal(mean=mean_log, cov=cov_log)
        data = dist_original.model_dump()

        # Reconstruct
        dist_reconstructed = MultiLogNormal(**data)

        np.testing.assert_array_equal(dist_reconstructed.mean, dist_original.mean)
        np.testing.assert_array_equal(dist_reconstructed.cov, dist_original.cov)


class TestMultiDistributionIntegration:
    """Integration tests for multi-dimensional distributions."""

    def test_cosmological_parameters_gauss(self) -> None:
        """Test realistic cosmological parameter distribution (Gaussian)."""
        # Approximate posterior for Omega_m and sigma_8
        mean = np.array([0.3, 0.8])
        std_omega_m = 0.01
        std_sigma_8 = 0.02
        corr = -0.5  # Negative correlation typical for these parameters

        cov = np.array(
            [
                [std_omega_m**2, corr * std_omega_m * std_sigma_8],
                [corr * std_omega_m * std_sigma_8, std_sigma_8**2],
            ]
        )

        dist = MultiGauss(
            mean=mean,
            cov=cov,
            param_names=["omega_m", "sigma_8"],
        )

        # Draw samples
        samples = dist.sample(10000, random_state=42)

        # Check that samples are physical (positive)
        assert np.all(samples[:, 0] > 0)  # omega_m > 0
        assert np.all(samples[:, 1] > 0)  # sigma_8 > 0

        # Check correlation
        sample_corr = np.corrcoef(samples.T)[0, 1]
        np.testing.assert_allclose(sample_corr, corr, atol=0.05)

    def test_amplitude_parameters_lognormal(self) -> None:
        """Test realistic amplitude parameters (log-normal)."""
        # Parameters that are naturally positive and multiplicative
        # e.g., A_s (primordial amplitude), tau (optical depth)
        mean_log = np.array([np.log(2.1e-9), np.log(0.06)])
        std_log = np.array([0.02, 0.1])
        corr = 0.3

        cov_log = np.array(
            [
                [std_log[0] ** 2, corr * std_log[0] * std_log[1]],
                [corr * std_log[0] * std_log[1], std_log[1] ** 2],
            ]
        )

        dist = MultiLogNormal(
            mean=mean_log,
            cov=cov_log,
            param_names=["A_s", "tau"],
        )

        # Draw samples
        samples = dist.sample(10000, random_state=42)

        # All samples must be positive
        assert np.all(samples > 0)

        # Check approximate means in real space
        mean_real = dist.mean_real_space()
        sample_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_mean, mean_real, rtol=0.1)

    def test_mixed_prior_workflow(self) -> None:
        """Test workflow with both Gaussian and log-normal priors."""
        # Gaussian prior for omega_m, sigma_8
        gauss_mean = np.array([0.3, 0.8])
        gauss_cov = np.diag([0.01**2, 0.02**2])
        gauss_dist = MultiGauss(
            mean=gauss_mean,
            cov=gauss_cov,
            param_names=["omega_m", "sigma_8"],
        )

        # Log-normal prior for A_s
        lognorm_mean = np.array([np.log(2.1e-9)])
        lognorm_cov = np.array([[0.02**2]])
        lognorm_dist = MultiLogNormal(
            mean=lognorm_mean,
            cov=lognorm_cov,
            param_names=["A_s"],
        )

        # Sample from both
        gauss_samples = gauss_dist.sample(1000, random_state=42)
        lognorm_samples = lognorm_dist.sample(1000, random_state=43)

        # Combine samples
        combined_samples = np.hstack([gauss_samples, lognorm_samples])

        assert combined_samples.shape == (1000, 3)
        assert np.all(combined_samples[:, 2] > 0)  # A_s is positive

    def test_conditional_sampling_workflow(self) -> None:
        """Test workflow that conditions on observed correlations."""
        # Start with uncorrelated prior
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        prior = MultiGauss(mean=mean, cov=cov)

        # Draw samples
        prior_samples = prior.sample(5000, random_state=42)

        # Check samples are approximately uncorrelated
        prior_corr = np.corrcoef(prior_samples.T)[0, 1]
        np.testing.assert_allclose(prior_corr, 0.0, atol=0.05)

        # Create posterior with induced correlation
        posterior_cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        posterior = MultiGauss(mean=mean, cov=posterior_cov)

        posterior_samples = posterior.sample(5000, random_state=42)
        posterior_corr = np.corrcoef(posterior_samples.T)[0, 1]

        # Posterior should have the imposed correlation
        np.testing.assert_allclose(posterior_corr, 0.6, atol=0.05)


class TestMultiDistributionDocumentation:
    """Tests that verify examples in docstrings work."""

    def test_multigauss_docstring_example(self) -> None:
        """Test MultiGauss docstring example."""
        # 2D Gaussian with correlation
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        dist = MultiGauss(mean=mean, cov=cov, param_names=["omega_m", "sigma_8"])

        samples = dist.sample(1000, random_state=42)
        assert samples.shape == (1000, 2)

        log_p = dist.log_prob(samples)
        assert log_p.shape == (1000,)

    def test_multilognormal_docstring_example(self) -> None:
        """Test MultiLogNormal docstring example."""
        # 2D log-normal with correlation
        mean_log = np.array([0.0, 0.0])  # exp(0) = 1.0 in real space
        cov_log = np.array([[0.1, 0.05], [0.05, 0.2]])
        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=["A_s", "n_s"])

        samples = dist.sample(1000, random_state=42)
        # Samples are positive
        assert np.all(samples > 0)

    def test_create_scipy_distribution_docstring_example(self) -> None:
        """Test factory function docstring example."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 2)


class TestMultiDistributionProperties:
    """Tests for distribution properties and derived quantities."""

    def test_gauss_std_diagonal(self) -> None:
        """Test std property for diagonal covariance."""
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.diag([0.01, 0.04, 0.09])
        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_gauss_std_with_correlation(self) -> None:
        """Test std property with non-diagonal covariance."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[0.01, 0.005], [0.005, 0.04]])
        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_correlation_identity_for_uncorrelated(self) -> None:
        """Test correlation matrix is identity for uncorrelated variables."""
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.diag([1.0, 2.0, 3.0])
        dist = MultiGauss(mean=mean, cov=cov)

        np.testing.assert_allclose(dist.correlation, np.eye(3))

    def test_correlation_symmetric(self) -> None:
        """Test correlation matrix is symmetric."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        dist = MultiGauss(mean=mean, cov=cov)

        corr = dist.correlation
        np.testing.assert_allclose(corr, corr.T)

    def test_correlation_diagonal_ones(self) -> None:
        """Test correlation matrix has ones on diagonal."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        dist = MultiGauss(mean=mean, cov=cov)

        corr = dist.correlation
        np.testing.assert_allclose(np.diag(corr), np.ones(2))

    def test_lognormal_mean_real_space_formula(self) -> None:
        """Test log-normal mean formula: E[X] = exp(μ + σ²/2)."""
        mean_log = np.array([1.0, 2.0])
        var_log = np.array([0.1, 0.2])
        cov_log = np.diag(var_log)

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        mean_real = dist.mean_real_space()

        expected = np.exp(mean_log + var_log / 2.0)
        np.testing.assert_allclose(mean_real, expected)

    def test_lognormal_variance_real_space_formula(self) -> None:
        """Test log-normal variance formula."""
        mean_log = np.array([0.0, 0.0])
        var_log = np.array([0.1, 0.2])
        cov_log = np.diag(var_log)

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        var_real = dist.variance_real_space()

        expected = (np.exp(var_log) - 1.0) * np.exp(2.0 * mean_log + var_log)
        np.testing.assert_allclose(var_real, expected)
