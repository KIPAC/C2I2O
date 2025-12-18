"""Tests for c2i2o.core.distribution module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.distribution import DistributionBase, FixedDistribution


class TestFixedDistribution:
    """Tests for FixedDistribution class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = FixedDistribution(value=3.14)
        assert dist.value == 3.14
        assert dist.dist_type == "fixed"

    def test_sample_single_value(self) -> None:
        """Test sampling returns fixed value."""
        dist = FixedDistribution(value=5.0)
        samples = dist.sample(100)
        assert samples.shape == (100,)
        assert np.all(samples == 5.0)

    def test_sample_reproducible(self, random_state: int) -> None:
        """Test sampling is reproducible (though deterministic)."""
        dist = FixedDistribution(value=2.5)
        samples1 = dist.sample(50, random_state=random_state)
        samples2 = dist.sample(50, random_state=random_state)
        np.testing.assert_array_equal(samples1, samples2)

    def test_log_prob_scalar(self) -> None:
        """Test log_prob returns 0 for scalar input."""
        dist = FixedDistribution(value=1.0)
        log_p = dist.log_prob(1.0)
        assert log_p == 0.0

        log_p = dist.log_prob(999.0)  # Any value returns 0
        assert log_p == 0.0

    def test_log_prob_array(self) -> None:
        """Test log_prob returns zeros for array input."""
        dist = FixedDistribution(value=1.0)
        x = np.array([0.5, 1.0, 1.5])
        log_p = dist.log_prob(x)
        assert log_p.shape == (3,)
        np.testing.assert_array_equal(log_p, np.zeros(3))

    def test_negative_value(self) -> None:
        """Test negative values are allowed."""
        dist = FixedDistribution(value=-10.0)
        assert dist.value == -10.0
        samples = dist.sample(10)
        assert np.all(samples == -10.0)

    def test_serialization(self) -> None:
        """Test serialization round-trip."""
        dist = FixedDistribution(value=42.0)
        data = dist.model_dump()
        assert data["dist_type"] == "fixed"
        assert data["value"] == 42.0

        # Reconstruct
        dist_new = FixedDistribution(**data)
        assert dist_new.value == 42.0

    def test_validation_error_missing_value(self) -> None:
        """Test validation error when value is missing."""
        with pytest.raises(ValidationError):
            FixedDistribution()  # type: ignore


class TestDistributionBase:
    """Tests for DistributionBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that DistributionBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DistributionBase(dist_type="test")  # type: ignore

    def test_subclass_must_implement_methods(self) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteDistribution(DistributionBase):
            """Dummy class missing sample() and log_prob()"""

            dist_type: str = "incomplete"

        with pytest.raises(TypeError):
            IncompleteDistribution()  # type: ignore
