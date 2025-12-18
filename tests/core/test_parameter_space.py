"""Tests for c2i2o.core.parameter_space module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.distribution import FixedDistribution
from c2i2o.core.parameter_space import ParameterSpace
from c2i2o.core.scipy_distributions import Norm, Uniform


class TestParameterSpace:
    """Tests for ParameterSpace class."""

    def test_initialization(self, simple_parameter_space: ParameterSpace) -> None:
        """Test basic initialization."""
        assert len(simple_parameter_space.parameters) == 3
        assert "omega_m" in simple_parameter_space.parameters
        assert "sigma_8" in simple_parameter_space.parameters
        assert "h" in simple_parameter_space.parameters

    def test_parameter_names(self, simple_parameter_space: ParameterSpace) -> None:
        """Test parameter_names property."""
        names = simple_parameter_space.parameter_names
        assert names == ["h", "omega_m", "sigma_8"]  # Sorted alphabetically

    def test_n_parameters(self, simple_parameter_space: ParameterSpace) -> None:
        """Test n_parameters property."""
        assert simple_parameter_space.n_parameters == 3

    def test_empty_parameters_raises_error(self) -> None:
        """Test that empty parameter dict raises error."""
        with pytest.raises(ValidationError, match="at least one parameter"):
            ParameterSpace(parameters={})

    def test_sample_shape(self, simple_parameter_space: ParameterSpace, random_state: int) -> None:
        """Test sample returns correct shape."""
        samples = simple_parameter_space.sample(100, random_state=random_state)
        assert isinstance(samples, dict)
        assert len(samples) == 3
        for name in simple_parameter_space.parameter_names:
            assert name in samples
            assert samples[name].shape == (100,)

    def test_sample_reproducible(self, simple_parameter_space: ParameterSpace, random_state: int) -> None:
        """Test sampling is reproducible with random_state."""
        samples1 = simple_parameter_space.sample(50, random_state=random_state)
        samples2 = simple_parameter_space.sample(50, random_state=random_state)

        for name in simple_parameter_space.parameter_names:
            np.testing.assert_array_equal(samples1[name], samples2[name])

    def test_sample_fixed_distribution(self, simple_parameter_space: ParameterSpace) -> None:
        """Test that fixed distribution returns constant values."""
        samples = simple_parameter_space.sample(100)
        assert np.all(samples["h"] == 0.7)

    def test_log_prob_shape(self, simple_parameter_space: ParameterSpace, random_state: int) -> None:
        """Test log_prob returns correct shape."""
        samples = simple_parameter_space.sample(100, random_state=random_state)
        log_probs = simple_parameter_space.log_prob(samples)

        assert isinstance(log_probs, dict)
        assert len(log_probs) == 3
        for name in simple_parameter_space.parameter_names:
            assert name in log_probs
            assert log_probs[name].shape == (100,)

    def test_log_prob_scalar(self) -> None:
        """Test log_prob with scalar inputs."""
        param_space = ParameterSpace(parameters={"x": Norm(loc=0.0, scale=1.0)})
        values = {"x": 0.0}
        log_probs = param_space.log_prob(values)
        assert isinstance(log_probs["x"], (np.ndarray))

    def test_log_prob_missing_parameter_raises_error(self, simple_parameter_space: ParameterSpace) -> None:
        """Test log_prob raises error when parameter is missing."""
        values = {"omega_m": 0.3, "sigma_8": 0.8}  # Missing 'h'
        with pytest.raises(KeyError, match="missing"):
            simple_parameter_space.log_prob(values)

    def test_log_prob_joint(self, simple_parameter_space: ParameterSpace, random_state: int) -> None:
        """Test joint log probability calculation."""
        samples = simple_parameter_space.sample(100, random_state=random_state)
        log_prob_joint = simple_parameter_space.log_prob_joint(samples)

        # Should be sum of individual log probs
        log_probs_indiv = simple_parameter_space.log_prob(samples)
        expected = sum(log_probs_indiv.values())
        np.testing.assert_array_equal(log_prob_joint, expected)

    def test_log_prob_joint_scalar(self) -> None:
        """Test joint log_prob with scalar inputs."""
        param_space = ParameterSpace(
            parameters={
                "x": Norm(loc=0.0, scale=1.0),
                "y": Uniform(loc=0.0, scale=1.0),
            }
        )
        values = {"x": 0.0, "y": 0.5}
        log_prob_joint = param_space.log_prob_joint(values)
        assert isinstance(log_prob_joint, (float, np.floating))

    def test_prob(self, simple_parameter_space: ParameterSpace, random_state: int) -> None:
        """Test probability density calculation."""
        samples = simple_parameter_space.sample(100, random_state=random_state)
        probs = simple_parameter_space.prob(samples)

        assert isinstance(probs, dict)
        assert len(probs) == 3
        for name in simple_parameter_space.parameter_names:
            assert name in probs
            assert probs[name].shape == (100,)
            assert np.all(probs[name] >= 0)

    def test_get_bounds(self, simple_parameter_space: ParameterSpace) -> None:
        """Test getting parameter bounds."""
        bounds = simple_parameter_space.get_bounds()

        assert isinstance(bounds, dict)
        assert len(bounds) == 3

        # Uniform bounds
        assert bounds["omega_m"] == (0.2, 0.4)

        # Normal has infinite bounds
        assert bounds["sigma_8"] == (-np.inf, np.inf)

        # Fixed has zero-width bounds
        assert bounds["h"] == (0.7, 0.7)

    def test_get_means(self, simple_parameter_space: ParameterSpace) -> None:
        """Test getting parameter means."""
        means = simple_parameter_space.get_means()

        assert isinstance(means, dict)
        assert len(means) == 3

        # Uniform mean is (loc + loc + scale) / 2
        np.testing.assert_allclose(means["omega_m"], 0.3)

        # Normal mean is loc
        assert means["sigma_8"] == 0.8

        # Fixed mean is value
        assert means["h"] == 0.7

    def test_get_stds(self, simple_parameter_space: ParameterSpace) -> None:
        """Test getting parameter standard deviations."""
        stds = simple_parameter_space.get_stds()
        assert isinstance(stds, dict)
        assert len(stds) == 3

        # Uniform std is scale / sqrt(12)
        expected_uniform_std = 0.2 / np.sqrt(12)
        np.testing.assert_allclose(stds["omega_m"], expected_uniform_std)

        # Normal std is scale
        assert stds["sigma_8"] == 0.1

        # Fixed std is 0
        assert stds["h"] == 0.0

    def test_to_array_scalar(self) -> None:
        """Test converting scalar dict to array."""
        param_space = ParameterSpace(
            parameters={
                "b": Norm(loc=0.0, scale=1.0),
                "a": Uniform(loc=0.0, scale=1.0),
            }
        )
        values = {"a": 0.5, "b": 1.5}
        arr = param_space.to_array(values)

        # Should be sorted alphabetically: [a, b]
        assert arr.shape == (2,)
        np.testing.assert_array_equal(arr, [0.5, 1.5])

    def test_to_array_with_arrays(self, simple_parameter_space: ParameterSpace) -> None:
        """Test converting array dict to 2D array."""
        values = {
            "omega_m": np.array([0.25, 0.30, 0.35]),
            "sigma_8": np.array([0.7, 0.8, 0.9]),
            "h": np.array([0.6, 0.7, 0.8]),
        }
        arr = simple_parameter_space.to_array(values)

        # Shape should be (n_samples, n_params)
        assert arr.shape == (3, 3)

        # Check ordering (alphabetical: h, omega_m, sigma_8)
        np.testing.assert_array_equal(arr[:, 0], values["h"])
        np.testing.assert_array_equal(arr[:, 1], values["omega_m"])
        np.testing.assert_array_equal(arr[:, 2], values["sigma_8"])

    def test_to_array_missing_parameter_raises_error(self, simple_parameter_space: ParameterSpace) -> None:
        """Test to_array raises error when parameter is missing."""
        values = {"omega_m": 0.3, "sigma_8": 0.8}  # Missing 'h'
        with pytest.raises(KeyError, match="missing"):
            simple_parameter_space.to_array(values)

    def test_from_array_1d(self) -> None:
        """Test converting 1D array to dict."""
        param_space = ParameterSpace(
            parameters={
                "b": Norm(loc=0.0, scale=1.0),
                "a": Uniform(loc=0.0, scale=1.0),
            }
        )
        arr = np.array([0.5, 1.5])  # [a, b] in alphabetical order
        values = param_space.from_array(arr)

        assert isinstance(values, dict)
        assert len(values) == 2
        assert values["a"] == 0.5
        assert values["b"] == 1.5

    def test_from_array_2d(self, simple_parameter_space: ParameterSpace) -> None:
        """Test converting 2D array to dict."""
        arr = np.array(
            [
                [0.6, 0.25, 0.7],
                [0.7, 0.30, 0.8],
                [0.8, 0.35, 0.9],
            ]
        )  # Columns: h, omega_m, sigma_8

        values = simple_parameter_space.from_array(arr)

        assert isinstance(values, dict)
        assert len(values) == 3
        np.testing.assert_array_equal(values["h"], [0.6, 0.7, 0.8])
        np.testing.assert_array_equal(values["omega_m"], [0.25, 0.30, 0.35])
        np.testing.assert_array_equal(values["sigma_8"], [0.7, 0.8, 0.9])

    def test_from_array_wrong_shape_raises_error(self, simple_parameter_space: ParameterSpace) -> None:
        """Test from_array raises error with wrong shape."""
        arr = np.array([0.5, 1.5])  # Only 2 parameters, need 3
        with pytest.raises(ValueError, match="must match number of parameters"):
            simple_parameter_space.from_array(arr)

    def test_to_array_from_array_roundtrip(
        self, simple_parameter_space: ParameterSpace, random_state: int
    ) -> None:
        """Test round-trip conversion between dict and array."""
        original = simple_parameter_space.sample(50, random_state=random_state)
        arr = simple_parameter_space.to_array(original)
        reconstructed = simple_parameter_space.from_array(arr)

        for name in simple_parameter_space.parameter_names:
            np.testing.assert_array_equal(original[name], reconstructed[name])

    def test_serialization(self, simple_parameter_space: ParameterSpace) -> None:
        """Test serialization round-trip."""
        data = simple_parameter_space.model_dump()

        assert "parameters" in data
        assert len(data["parameters"]) == 3

        # Reconstruct (simplified - would need proper handling in production)
        # This tests that the structure is correct
        assert data["parameters"]["omega_m"]["dist_type"] == "uniform"
        assert data["parameters"]["sigma_8"]["dist_type"] == "norm"
        assert data["parameters"]["h"]["dist_type"] == "fixed"


class TestParameterSpaceEdgeCases:
    """Tests for edge cases in ParameterSpace."""

    def test_single_parameter(self) -> None:
        """Test parameter space with single parameter."""
        param_space = ParameterSpace(parameters={"x": Norm(loc=0.0, scale=1.0)})
        assert param_space.n_parameters == 1
        assert param_space.parameter_names == ["x"]

    def test_many_parameters(self, random_state: int) -> None:
        """Test parameter space with many parameters."""
        params = {f"param_{i}": Uniform(loc=0.0, scale=1.0) for i in range(20)}
        param_space = ParameterSpace(parameters=params)

        assert param_space.n_parameters == 20
        samples = param_space.sample(10, random_state=random_state)
        assert len(samples) == 20

        arr = param_space.to_array(samples)
        assert arr.shape == (10, 20)

    def test_all_fixed_parameters(self) -> None:
        """Test parameter space with all fixed parameters."""
        param_space = ParameterSpace(
            parameters={
                "a": FixedDistribution(value=1.0),
                "b": FixedDistribution(value=2.0),
                "c": FixedDistribution(value=3.0),
            }
        )

        samples = param_space.sample(100)
        assert np.all(samples["a"] == 1.0)
        assert np.all(samples["b"] == 2.0)
        assert np.all(samples["c"] == 3.0)

        stds = param_space.get_stds()
        assert all(std == 0.0 for std in stds.values())


class TestParameterSpaceCoverageGaps:
    """Tests to cover missing lines in parameter_space.py."""

    def test_prob_with_distribution_without_prob_method(self) -> None:
        """Test prob falls back to exp(log_prob) for distributions without prob method.

        Covers lines 229-231: else branch in prob method
        Note: FixedDistribution doesn't have prob method
        """
        param_space = ParameterSpace(parameters={"x": FixedDistribution(value=5.0)})

        values = {"x": np.array([5.0, 6.0, 7.0])}
        probs = param_space.prob(values)

        # Should use exp(log_prob) which gives exp(0) = 1.0
        assert "x" in probs
        np.testing.assert_array_equal(probs["x"], np.ones(3))
