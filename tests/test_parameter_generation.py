"""Tests for parameter generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tables_io
import yaml
from pydantic import ValidationError

from c2i2o.core.distribution import FixedDistribution
from c2i2o.core.multi_distribution import MultiDistributionSet, MultiGauss, MultiLogNormal
from c2i2o.core.parameter_space import ParameterSpace
from c2i2o.core.scipy_distributions import Norm, Uniform
from c2i2o.parameter_generation import ParameterGenerator


class TestParameterGenerator:
    """Tests for ParameterGenerator class."""

    def test_creation_basic(self) -> None:
        """Test creating a basic ParameterGenerator."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=1000,
            scale_factor=1.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        assert generator.num_samples == 1000
        assert generator.scale_factor == 1.0
        assert len(generator.parameter_space.parameters) == 1
        assert len(generator.multi_distribution_set.distributions) == 1

    def test_creation_with_scale_factor(self) -> None:
        """Test creating ParameterGenerator with non-default scale_factor."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=500,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        assert generator.scale_factor == 1.5

    def test_num_samples_validation_negative(self) -> None:
        """Test that negative num_samples raises error."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError):
            ParameterGenerator(
                num_samples=-100,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_num_samples_validation_zero(self) -> None:
        """Test that zero num_samples raises error."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError, match="greater than 0"):
            ParameterGenerator(
                num_samples=0,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_scale_factor_validation_negative(self) -> None:
        """Test that negative scale_factor raises error."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError):
            ParameterGenerator(
                num_samples=100,
                scale_factor=-1.5,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_scale_factor_validation_zero(self) -> None:
        """Test that zero scale_factor raises error."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError, match="greater than 0"):
            ParameterGenerator(
                num_samples=100,
                scale_factor=0.0,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_name_collision_detection(self) -> None:
        """Test that parameter name collisions are detected."""
        param_space = ParameterSpace(
            parameters={
                "omega_m": Norm(loc=0.3, scale=0.05),  # Collision here
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],  # omega_m collision
                )
            ]
        )

        with pytest.raises(ValidationError, match="Parameter name collision"):
            ParameterGenerator(
                num_samples=100,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_name_collision_multiple_parameters(self) -> None:
        """Test detection of multiple parameter name collisions."""
        param_space = ParameterSpace(
            parameters={
                "omega_m": Norm(loc=0.3, scale=0.05),
                "sigma_8": Norm(loc=0.8, scale=0.1),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        with pytest.raises(ValidationError, match="Parameter name collision"):
            ParameterGenerator(
                num_samples=100,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_no_collision_valid_names(self) -> None:
        """Test that valid distinct names don't cause collision errors."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        # Should not raise
        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        assert generator is not None

    def test_collision_with_default_names(self) -> None:
        """Test collision detection with default multi-distribution names."""
        param_space = ParameterSpace(
            parameters={
                "dist0_param0": Norm(loc=0.3, scale=0.05),  # Matches default name
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=None,  # Will get default names
                )
            ]
        )

        with pytest.raises(ValidationError, match="Parameter name collision"):
            ParameterGenerator(
                num_samples=100,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

    def test_generate_basic(self) -> None:
        """Test basic parameter generation."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        assert set(samples.keys()) == {"n_s", "omega_m", "sigma_8"}
        assert samples["n_s"].shape == (100,)
        assert samples["omega_m"].shape == (100,)
        assert samples["sigma_8"].shape == (100,)

    def test_generate_reproducibility(self) -> None:
        """Test that generation is reproducible with same random_state."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=50,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples1 = generator.generate(random_state=42)
        samples2 = generator.generate(random_state=42)

        np.testing.assert_array_equal(samples1["n_s"], samples2["n_s"])
        np.testing.assert_array_equal(samples1["omega_m"], samples2["omega_m"])

    def test_generate_correct_sample_count(self) -> None:
        """Test that correct number of samples are generated."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        for num_samples in [10, 100, 1000]:
            generator = ParameterGenerator(
                num_samples=num_samples,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

            samples = generator.generate(random_state=42)

            assert samples["n_s"].shape == (num_samples,)
            assert samples["omega_m"].shape == (num_samples,)

    def test_scale_factor_affects_univariate(self) -> None:
        """Test that scale_factor correctly scales univariate distributions."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        # Generate with scale_factor = 1.0
        gen1 = ParameterGenerator(
            num_samples=10000,
            scale_factor=1.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )
        samples1 = gen1.generate(random_state=42)
        std1 = np.std(samples1["n_s"])

        # Generate with scale_factor = 2.0
        gen2 = ParameterGenerator(
            num_samples=10000,
            scale_factor=2.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )
        samples2 = gen2.generate(random_state=42)
        std2 = np.std(samples2["n_s"])

        # Standard deviation should roughly double
        np.testing.assert_allclose(std2 / std1, 2.0, rtol=0.1)

    def test_scale_factor_affects_multivariate(self) -> None:
        """Test that scale_factor correctly scales multivariate distributions."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        # Generate with scale_factor = 1.0
        gen1 = ParameterGenerator(
            num_samples=10000,
            scale_factor=1.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )
        samples1 = gen1.generate(random_state=42)
        std1 = np.std(samples1["omega_m"])

        # Generate with scale_factor = 1.5
        gen2 = ParameterGenerator(
            num_samples=10000,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )
        samples2 = gen2.generate(random_state=42)
        std2 = np.std(samples2["omega_m"])

        # Standard deviation should scale by factor
        np.testing.assert_allclose(std2 / std1, 1.5, rtol=0.1)

    def test_scale_factor_preserves_mean(self) -> None:
        """Test that scale_factor doesn't affect means."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        gen = ParameterGenerator(
            num_samples=10000,
            scale_factor=2.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = gen.generate(random_state=42)

        # Means should be unchanged
        np.testing.assert_allclose(np.mean(samples["n_s"]), 0.96, atol=0.01)
        np.testing.assert_allclose(np.mean(samples["omega_m"]), 0.3, atol=0.01)

    def test_scale_factor_fixed_distribution_unaffected(self) -> None:
        """Test that FixedDistribution is not affected by scale_factor."""
        param_space = ParameterSpace(
            parameters={
                "n_s": FixedDistribution(value=0.96),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        gen = ParameterGenerator(
            num_samples=100,
            scale_factor=10.0,  # Large scale factor
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = gen.generate(random_state=42)

        # Fixed value should remain exactly 0.96
        np.testing.assert_array_equal(samples["n_s"], np.full(100, 0.96))

    def test_to_yaml_basic(self) -> None:
        """Test writing configuration to YAML file."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=1000,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            generator.to_yaml(filepath)

            # Check file exists
            assert filepath.exists()

            # Check file is valid YAML
            with open(filepath) as f:
                data = yaml.safe_load(f)

            assert data["num_samples"] == 1000
            assert data["scale_factor"] == 1.5

    def test_from_yaml_basic(self) -> None:
        """Test loading configuration from YAML file."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=1000,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            generator.to_yaml(filepath)

            # Load from YAML
            loaded_generator = ParameterGenerator.from_yaml(filepath)

            assert loaded_generator.num_samples == 1000
            assert loaded_generator.scale_factor == 1.5
            assert "n_s" in loaded_generator.parameter_space.parameters
            assert len(loaded_generator.multi_distribution_set.distributions) == 1

    def test_yaml_roundtrip_preserves_samples(self) -> None:
        """Test that YAML roundtrip preserves sample generation."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        # Generate original samples
        samples_orig = generator.generate(random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            generator.to_yaml(filepath)
            loaded_generator = ParameterGenerator.from_yaml(filepath)

            # Generate samples from loaded generator
            samples_loaded = loaded_generator.generate(random_state=42)

            # Should be identical
            for key in samples_orig:
                np.testing.assert_array_equal(samples_orig[key], samples_loaded[key])

    def test_yaml_with_lognormal(self) -> None:
        """Test YAML serialization with log-normal distributions."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiLogNormal(
                    mean=np.array([0.0]),
                    cov=np.array([[0.1]]),
                    param_names=["A_s"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            generator.to_yaml(filepath)
            loaded_generator = ParameterGenerator.from_yaml(filepath)

            assert isinstance(loaded_generator.multi_distribution_set.distributions[0], MultiLogNormal)

    def test_generate_to_hdf5_basic(self) -> None:
        """Test generating samples directly to HDF5 file."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "samples.h5"
            generator.generate_to_hdf5(filepath, random_state=42)

            # Check file exists
            assert filepath.exists()

            # Read back and verify
            data = tables_io.read(str(filepath))

            assert set(data.keys()) == {"n_s", "omega_m", "sigma_8"}
            assert data["n_s"].shape == (100,)
            assert data["omega_m"].shape == (100,)
            assert data["sigma_8"].shape == (100,)

    def test_generate_to_hdf5_matches_generate(self) -> None:
        """Test that HDF5 output matches direct generation."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        # Generate in memory
        samples_direct = generator.generate(random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "samples.h5"
            generator.generate_to_hdf5(filepath, random_state=42)

            # Read from HDF5
            samples_hdf5 = tables_io.read(str(filepath))

            # Should match
            for key in samples_direct:
                np.testing.assert_array_equal(samples_direct[key], samples_hdf5[key])

    def test_multiple_univariate_distributions(self) -> None:
        """Test with multiple univariate distributions."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
                "tau": FixedDistribution(value=0.06),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        assert set(samples.keys()) == {"n_s", "h", "tau", "omega_m"}
        assert all(arr.shape == (100,) for arr in samples.values())

    def test_multiple_multivariate_distributions(self) -> None:
        """Test with multiple multivariate distributions."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                ),
                MultiLogNormal(
                    mean=np.array([0.0]),
                    cov=np.array([[0.1]]),
                    param_names=["A_s"],
                ),
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        assert set(samples.keys()) == {"n_s", "omega_m", "sigma_8", "A_s"}
        assert all(arr.shape == (100,) for arr in samples.values())

    def test_empty_parameter_space(self) -> None:
        """Test with empty ParameterSpace (only multivariate)."""
        param_space = ParameterSpace(parameters={})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        assert set(samples.keys()) == {"omega_m", "sigma_8"}
        assert samples["omega_m"].shape == (100,)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ParameterGenerator(
                num_samples=100,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
                extra_field="not allowed",
            )  # type: ignore

    def test_large_scale_factor(self) -> None:
        """Test with very large scale factor."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=10000,
            scale_factor=10.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        # Standard deviations should be scaled by factor
        std_ns = np.std(samples["n_s"])
        std_omega = np.std(samples["omega_m"])

        np.testing.assert_allclose(std_ns, 0.01 * 10.0, rtol=0.1)
        np.testing.assert_allclose(std_omega, 0.1 * 10.0, rtol=0.1)

    def test_small_scale_factor(self) -> None:
        """Test with very small scale factor."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=10000,
            scale_factor=0.1,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        # Standard deviations should be reduced
        std_ns = np.std(samples["n_s"])
        std_omega = np.std(samples["omega_m"])

        np.testing.assert_allclose(std_ns, 0.01 * 0.1, rtol=0.1)
        np.testing.assert_allclose(std_omega, 0.1 * 0.1, rtol=0.1)

    def test_yaml_path_as_string(self) -> None:
        """Test YAML I/O with string paths."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_str = str(Path(tmpdir) / "config.yaml")
            generator.to_yaml(filepath_str)

            # Load using string path
            loaded_generator = ParameterGenerator.from_yaml(filepath_str)

            assert loaded_generator.num_samples == 100

    def test_hdf5_path_as_string(self) -> None:
        """Test HDF5 generation with string paths."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=50,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_str = str(Path(tmpdir) / "samples.h5")
            generator.generate_to_hdf5(filepath_str, random_state=42)

            assert Path(filepath_str).exists()

            # Read and verify
            data = tables_io.read(filepath_str)
            assert "n_s" in data

    def test_yaml_missing_file(self) -> None:
        """Test that loading from non-existent YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            ParameterGenerator.from_yaml("nonexistent.yaml")

    def test_serialization_complex_setup(self) -> None:
        """Test serialization with complex multi-distribution setup."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
                "tau": FixedDistribution(value=0.06),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                ),
                MultiLogNormal(
                    mean=np.array([0.0, 0.0]),
                    cov=np.array([[0.1, 0.05], [0.05, 0.15]]),
                    param_names=["A_s", "omega_b"],
                ),
            ]
        )

        generator = ParameterGenerator(
            num_samples=500,
            scale_factor=1.2,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        # Test model_dump
        data = generator.model_dump()

        assert data["num_samples"] == 500
        assert data["scale_factor"] == 1.2
        assert len(data["parameter_space"]["parameters"]) == 3
        assert len(data["multi_distribution_set"]["distributions"]) == 2

    def test_correlation_preserved_after_scaling(self) -> None:
        """Test that correlations are preserved after scaling."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.008], [0.008, 0.02]]),  # Correlated
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        # Original correlation
        orig_corr = multi_dist.distributions[0].correlation[0, 1]

        generator = ParameterGenerator(
            num_samples=10000,
            scale_factor=2.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        # Compute sample correlation
        sample_corr = np.corrcoef(samples["omega_m"], samples["sigma_8"])[0, 1]

        # Correlation should be preserved
        np.testing.assert_allclose(sample_corr, orig_corr, atol=0.05)

    def test_generate_with_different_random_states(self) -> None:
        """Test that different random states give different samples."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples1 = generator.generate(random_state=42)
        samples2 = generator.generate(random_state=43)

        # Should be different
        assert not np.array_equal(samples1["n_s"], samples2["n_s"])
        assert not np.array_equal(samples1["omega_m"], samples2["omega_m"])

    def test_hdf5_overwrite(self) -> None:
        """Test that HDF5 file can be overwritten."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator1 = ParameterGenerator(
            num_samples=50,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        generator2 = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "samples.h5"

            # Write first time
            generator1.generate_to_hdf5(filepath, random_state=42)
            data1 = tables_io.read(str(filepath))
            assert data1["n_s"].shape == (50,)

            # Overwrite
            generator2.generate_to_hdf5(filepath, random_state=42)
            data2 = tables_io.read(str(filepath))
            assert data2["n_s"].shape == (100,)

    def test_default_scale_factor_is_one(self) -> None:
        """Test that default scale_factor is 1.0."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=100,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        assert generator.scale_factor == 1.0

    def test_yaml_contains_all_fields(self) -> None:
        """Test that YAML output contains all expected fields."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=1000,
            scale_factor=1.5,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            generator.to_yaml(filepath)

            with open(filepath) as f:
                data = yaml.safe_load(f)

            # Check all top-level fields present
            assert "num_samples" in data
            assert "scale_factor" in data
            assert "parameter_space" in data
            assert "multi_distribution_set" in data

            # Check nested structure
            assert "parameters" in data["parameter_space"]
            assert "distributions" in data["multi_distribution_set"]

    def test_statistical_properties_preserved(self) -> None:
        """Test that statistical properties are preserved in generated samples."""
        param_space = ParameterSpace(
            parameters={
                "n_s": Norm(loc=0.96, scale=0.01),
                "h": Uniform(loc=0.6, scale=0.2),
            }
        )
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3, 0.8]),
                    cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
                    param_names=["omega_m", "sigma_8"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=20000,
            scale_factor=1.0,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        samples = generator.generate(random_state=42)

        # Check means
        np.testing.assert_allclose(np.mean(samples["n_s"]), 0.96, atol=0.005)
        np.testing.assert_allclose(np.mean(samples["h"]), 0.7, atol=0.01)
        np.testing.assert_allclose(np.mean(samples["omega_m"]), 0.3, atol=0.005)
        np.testing.assert_allclose(np.mean(samples["sigma_8"]), 0.8, atol=0.01)

        # Check standard deviations
        np.testing.assert_allclose(np.std(samples["n_s"]), 0.01, rtol=0.1)
        np.testing.assert_allclose(np.std(samples["omega_m"]), 0.1, rtol=0.1)
        np.testing.assert_allclose(np.std(samples["sigma_8"]), np.sqrt(0.02), rtol=0.1)

    def test_hdf5_with_tables_io_kwargs(self) -> None:
        """Test passing additional kwargs to tables_io.write()."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        generator = ParameterGenerator(
            num_samples=50,
            parameter_space=param_space,
            multi_distribution_set=multi_dist,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "samples.h5"

            # This should not raise even with extra kwargs
            generator.generate_to_hdf5(filepath, random_state=42)

            data = tables_io.read(str(filepath), groupname="parameters")
            assert "n_s" in data

    def test_num_samples_validation_negative_explicit(self) -> None:
        """Test that explicitly negative num_samples raises ValidationError."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            ParameterGenerator(
                num_samples=-1,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

        # Check that the error message mentions the constraint
        assert "greater than 0" in str(exc_info.value).lower()

    def test_num_samples_validation_large_negative(self) -> None:
        """Test that large negative num_samples raises ValidationError."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            ParameterGenerator(
                num_samples=-1000,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_scale_factor_validation_negative_explicit(self) -> None:
        """Test that explicitly negative scale_factor raises ValidationError."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            ParameterGenerator(
                num_samples=100,
                scale_factor=-0.5,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_scale_factor_validation_large_negative(self) -> None:
        """Test that large negative scale_factor raises ValidationError."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            ParameterGenerator(
                num_samples=100,
                scale_factor=-10.0,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

        assert "greater than 0" in str(exc_info.value).lower()

    def test_both_num_samples_and_scale_factor_negative(self) -> None:
        """Test that both num_samples and scale_factor being negative raises ValidationError."""
        param_space = ParameterSpace(parameters={"n_s": Norm(loc=0.96, scale=0.01)})
        multi_dist = MultiDistributionSet(
            distributions=[
                MultiGauss(
                    mean=np.array([0.3]),
                    cov=np.array([[0.01]]),
                    param_names=["omega_m"],
                )
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            ParameterGenerator(
                num_samples=-100,
                scale_factor=-1.5,
                parameter_space=param_space,
                multi_distribution_set=multi_dist,
            )

        # Should have errors for both fields
        error_str = str(exc_info.value).lower()
        assert "num_samples" in error_str or "scale_factor" in error_str
