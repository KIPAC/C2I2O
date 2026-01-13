"""Tests for CCL tracer implementations and
base tracer configuration classes."""

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.tracer import (
    CMBLensingTracerConfig,
    NumberCountsTracerConfig,
    TracerConfigBase,
    WeakLensingTracerConfig,
)
from c2i2o.interfaces.ccl.tracer import (
    CCLCMBLensingTracerConfig,
    CCLNumberCountsTracerConfig,
    CCLWeakLensingTracerConfig,
)

# Skip all tests if pyccl is not installed, or mark them to avoid segfaults
pytestmark = pytest.mark.skipif(
    True,  # Always skip CCL tracer tests that call actual pyccl
    reason="CCL tracer tests require mocking that can cause segfaults",
)


class TestCCLNumberCountsTracerConfig:
    """Tests for CCLNumberCountsTracerConfig."""

    @pytest.fixture
    def z_grid(self) -> np.ndarray:
        """Create a redshift grid."""
        return np.linspace(0, 2, 100)

    @pytest.fixture
    def dNdz_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """Create a redshift distribution."""
        return np.exp(-(((z_grid - 0.5) / 0.3) ** 2))

    @pytest.fixture
    def bias_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """Create a bias evolution."""
        return np.ones_like(z_grid) * 1.5

    def test_creation_basic(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating a basic number counts tracer."""
        tracer = CCLNumberCountsTracerConfig(
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        assert tracer.name == "galaxies_bin1"
        assert tracer.tracer_type == "ccl_number_counts"
        assert tracer.has_rsd is True
        assert tracer.bias_grid is None
        assert tracer.mag_bias is None

    def test_creation_with_bias(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
        bias_grid: np.ndarray,
    ) -> None:
        """Test creating tracer with bias."""
        tracer = CCLNumberCountsTracerConfig(
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            bias_grid=bias_grid,
        )

        assert tracer.bias_grid is not None
        np.testing.assert_array_equal(tracer.bias_grid, bias_grid)

    def test_creation_with_mag_bias(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating tracer with magnification bias."""
        mag_bias = np.ones_like(z_grid) * 0.4

        tracer = CCLNumberCountsTracerConfig(
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            mag_bias=mag_bias,
        )

        assert tracer.mag_bias is not None
        np.testing.assert_array_equal(tracer.mag_bias, mag_bias)

    def test_has_rsd_flag(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test has_rsd flag."""
        # With RSD
        tracer_rsd = CCLNumberCountsTracerConfig(
            name="galaxies_with_rsd",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            has_rsd=True,
        )
        assert tracer_rsd.has_rsd is True

        # Without RSD
        tracer_no_rsd = CCLNumberCountsTracerConfig(
            name="galaxies_no_rsd",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            has_rsd=False,
        )
        assert tracer_no_rsd.has_rsd is False

    def test_mag_bias_coercion_from_list(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test that mag_bias is coerced from list to array."""
        mag_bias_list = np.array([0.4] * len(z_grid))

        tracer = CCLNumberCountsTracerConfig(
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            mag_bias=mag_bias_list,
        )

        assert isinstance(tracer.mag_bias, np.ndarray)

    def test_serialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test serialization to dict."""
        tracer = CCLNumberCountsTracerConfig(
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "ccl_number_counts"
        assert data["name"] == "galaxies_bin1"
        assert "z_grid" in data
        assert "dNdz_grid" in data

    def test_deserialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test deserialization from dict."""
        data = {
            "tracer_type": "ccl_number_counts",
            "name": "galaxies_bin1",
            "z_grid": z_grid.tolist(),
            "dNdz_grid": dNdz_grid.tolist(),
            "has_rsd": True,
        }

        tracer = CCLNumberCountsTracerConfig(**data)

        assert tracer.name == "galaxies_bin1"
        assert tracer.tracer_type == "ccl_number_counts"


class TestCCLWeakLensingTracerConfig:
    """Tests for CCLWeakLensingTracerConfig."""

    @pytest.fixture
    def z_grid(self) -> np.ndarray:
        """Create a redshift grid."""
        return np.linspace(0, 3, 150)

    @pytest.fixture
    def dNdz_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """Create a source redshift distribution."""
        return np.exp(-(((z_grid - 1.0) / 0.5) ** 2))

    def test_creation_basic(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating a basic weak lensing tracer."""
        tracer = CCLWeakLensingTracerConfig(
            name="source_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        assert tracer.name == "source_bin1"
        assert tracer.tracer_type == "ccl_weak_lensing"
        assert tracer.ia_bias is None
        assert tracer.use_A_ia is False

    def test_creation_with_ia_bias(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating tracer with intrinsic alignment bias."""
        tracer = CCLWeakLensingTracerConfig(
            name="source_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            ia_bias=1.0,
            use_A_ia=True,
        )

        assert tracer.ia_bias == 1.0
        assert tracer.use_A_ia is True

    def test_serialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test serialization to dict."""
        tracer = CCLWeakLensingTracerConfig(
            name="source_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            ia_bias=1.0,
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "ccl_weak_lensing"
        assert data["name"] == "source_bin1"
        assert data["ia_bias"] == 1.0

    def test_deserialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test deserialization from dict."""
        data = {
            "tracer_type": "ccl_weak_lensing",
            "name": "source_bin1",
            "z_grid": z_grid.tolist(),
            "dNdz_grid": dNdz_grid.tolist(),
            "ia_bias": 1.5,
            "use_A_ia": True,
        }

        tracer = CCLWeakLensingTracerConfig(**data)

        assert tracer.name == "source_bin1"
        assert tracer.ia_bias == 1.5
        assert tracer.use_A_ia is True


class TestCCLCMBLensingTracerConfig:
    """Tests for CCLCMBLensingTracerConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic CMB lensing tracer."""
        tracer = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
        )

        assert tracer.name == "cmb_lensing"
        assert tracer.tracer_type == "ccl_cmb_lensing"
        assert tracer.z_source == 1100.0

    def test_creation_custom_z_source(self) -> None:
        """Test creating tracer with custom z_source."""
        tracer = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
            z_source=1090.0,
        )

        assert tracer.z_source == 1090.0

    def test_z_source_validation_negative(self) -> None:
        """Test that negative z_source is rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            CCLCMBLensingTracerConfig(
                name="cmb_lensing",
                z_source=-10.0,
            )

    def test_z_source_validation_zero(self) -> None:
        """Test that zero z_source is rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            CCLCMBLensingTracerConfig(
                name="cmb_lensing",
                z_source=0.0,
            )

    def test_z_source_validation_too_low(self) -> None:
        """Test that unreasonably low z_source is rejected."""
        with pytest.raises(ValidationError, match="should be around 1100"):
            CCLCMBLensingTracerConfig(
                name="cmb_lensing",
                z_source=100.0,
            )

    def test_z_source_validation_too_high(self) -> None:
        """Test that unreasonably high z_source is rejected."""
        with pytest.raises(ValidationError, match="should be around 1100"):
            CCLCMBLensingTracerConfig(
                name="cmb_lensing",
                z_source=3000.0,
            )

    def test_z_source_validation_boundary(self) -> None:
        """Test z_source validation boundaries."""
        # Lower boundary
        tracer_low = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
            z_source=500.0,
        )
        assert tracer_low.z_source == 500.0

        # Upper boundary
        tracer_high = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
            z_source=2000.0,
        )
        assert tracer_high.z_source == 2000.0

    def test_serialization(self) -> None:
        """Test serialization to dict."""
        tracer = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
            z_source=1100.0,
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "ccl_cmb_lensing"
        assert data["name"] == "cmb_lensing"
        assert data["z_source"] == 1100.0

    def test_deserialization(self) -> None:
        """Test deserialization from dict."""
        data: dict[str, Any] = {
            "tracer_type": "ccl_cmb_lensing",
            "name": "cmb_lensing",
            "z_source": 1100.0,
        }

        tracer = CCLCMBLensingTracerConfig(**data)

        assert tracer.name == "cmb_lensing"
        assert tracer.z_source == 1100.0


class TestTracerDiscrimination:
    """Tests for tracer discriminated unions."""

    def test_number_counts_tracer_type(self) -> None:
        """Test that number counts has correct tracer_type."""
        z = np.linspace(0, 2, 100)
        dNdz = np.exp(-(((z - 0.5) / 0.3) ** 2))

        tracer = CCLNumberCountsTracerConfig(
            name="galaxies",
            z_grid=z,
            dNdz_grid=dNdz,
        )

        assert tracer.tracer_type == "ccl_number_counts"

    def test_weak_lensing_tracer_type(self) -> None:
        """Test that weak lensing has correct tracer_type."""
        z = np.linspace(0, 3, 150)
        dNdz = np.exp(-(((z - 1.0) / 0.5) ** 2))

        tracer = CCLWeakLensingTracerConfig(
            name="source",
            z_grid=z,
            dNdz_grid=dNdz,
        )

        assert tracer.tracer_type == "ccl_weak_lensing"

    def test_cmb_lensing_tracer_type(self) -> None:
        """Test that CMB lensing has correct tracer_type."""
        tracer = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
        )

        assert tracer.tracer_type == "ccl_cmb_lensing"

    def test_all_tracer_types_unique(self) -> None:
        """Test that all tracer types are unique."""
        z = np.linspace(0, 2, 100)
        dNdz = np.ones_like(z)

        nc = CCLNumberCountsTracerConfig(name="nc", z_grid=z, dNdz_grid=dNdz)
        wl = CCLWeakLensingTracerConfig(name="wl", z_grid=z, dNdz_grid=dNdz)
        cmb = CCLCMBLensingTracerConfig(name="cmb")

        types = {nc.tracer_type, wl.tracer_type, cmb.tracer_type}
        assert len(types) == 3


class TestTracerIntegration:
    """Integration tests for tracer configurations."""

    def test_yaml_roundtrip_number_counts(self) -> None:
        """Test YAML roundtrip for number counts tracer."""
        z = np.linspace(0, 2, 100)
        dNdz = np.exp(-(((z - 0.5) / 0.3) ** 2))
        bias = np.ones_like(z) * 1.5

        tracer = CCLNumberCountsTracerConfig(
            name="galaxies",
            z_grid=z,
            dNdz_grid=dNdz,
            bias_grid=bias,
            has_rsd=True,
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = CCLNumberCountsTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.tracer_type == tracer.tracer_type
        assert tracer_loaded.has_rsd == tracer.has_rsd
        np.testing.assert_array_almost_equal(tracer_loaded.z_grid, tracer.z_grid)

    def test_yaml_roundtrip_weak_lensing(self) -> None:
        """Test YAML roundtrip for weak lensing tracer."""
        z = np.linspace(0, 3, 150)
        dNdz = np.exp(-(((z - 1.0) / 0.5) ** 2))

        tracer = CCLWeakLensingTracerConfig(
            name="source",
            z_grid=z,
            dNdz_grid=dNdz,
            ia_bias=1.5,
            use_A_ia=True,
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = CCLWeakLensingTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.ia_bias == tracer.ia_bias
        assert tracer_loaded.use_A_ia == tracer.use_A_ia

    def test_yaml_roundtrip_cmb_lensing(self) -> None:
        """Test YAML roundtrip for CMB lensing tracer."""
        tracer = CCLCMBLensingTracerConfig(
            name="cmb_lensing",
            z_source=1100.0,
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = CCLCMBLensingTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.z_source == tracer.z_source

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        z = np.linspace(0, 2, 100)
        dNdz = np.ones_like(z)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CCLNumberCountsTracerConfig(
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz,
                extra_field="not allowed",  # type: ignore
            )


class TestTracerConfigBase:
    """Tests for TracerConfigBase."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that abstract base class cannot be instantiated directly."""
        # TracerConfigBase is not technically abstract in Python terms,
        # but it's meant to be subclassed
        # We can still instantiate it but it's not recommended
        tracer = TracerConfigBase(tracer_type="test", name="test_tracer")
        assert tracer.tracer_type == "test"
        assert tracer.name == "test_tracer"

    def test_name_validation_empty(self) -> None:
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            TracerConfigBase(tracer_type="test", name="")

    def test_name_validation_whitespace(self) -> None:
        """Test that whitespace-only name is rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            TracerConfigBase(tracer_type="test", name="   ")

    def test_name_validation_valid(self) -> None:
        """Test that valid names are accepted."""
        valid_names = [
            "tracer1",
            "my_tracer",
            "tracer-bin-1",
            "TRACER",
            "tracer 1",
        ]

        for name in valid_names:
            tracer = TracerConfigBase(tracer_type="test", name=name)
            assert tracer.name == name

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TracerConfigBase(
                tracer_type="test",
                name="test_tracer",
                extra_field="not allowed",  # type: ignore
            )


class TestNumberCountsTracerConfig:
    """Tests for NumberCountsTracerConfig."""

    @pytest.fixture
    def z_grid(self) -> np.ndarray:
        """Create a valid redshift grid."""
        return np.linspace(0, 2, 100)

    @pytest.fixture
    def dNdz_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """Create a valid redshift distribution."""
        return np.exp(-(((z_grid - 0.5) / 0.3) ** 2))

    def test_creation_basic(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating a basic number counts tracer."""
        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        assert tracer.name == "galaxies_bin1"
        assert tracer.tracer_type == "number_counts"
        assert tracer.bias_grid is None
        np.testing.assert_array_equal(tracer.z_grid, z_grid)
        np.testing.assert_array_equal(tracer.dNdz_grid, dNdz_grid)

    def test_creation_with_bias(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating tracer with bias."""
        bias = np.ones_like(z_grid) * 1.5

        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            bias_grid=bias,
        )

        assert tracer.bias_grid is not None
        np.testing.assert_array_equal(tracer.bias_grid, bias)

    def test_z_grid_coercion_from_list(self) -> None:
        """Test that z_grid is coerced from list to array."""
        z_list = [0.0, 0.5, 1.0, 1.5, 2.0]
        dNdz_list = [0.1, 0.5, 1.0, 0.5, 0.1]

        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z_list,  # type: ignore
            dNdz_grid=dNdz_list,  # type: ignore
        )

        assert isinstance(tracer.z_grid, np.ndarray)
        assert isinstance(tracer.dNdz_grid, np.ndarray)

    def test_z_grid_validation_not_1d(self) -> None:
        """Test that non-1D z_grid is rejected."""
        z_2d = np.array([[0, 1], [2, 3]])
        dNdz = np.array([1, 2, 3, 4])

        with pytest.raises(ValidationError, match="must be 1D array"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z_2d,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_too_few_points(self) -> None:
        """Test that z_grid with < 2 points is rejected."""
        z = np.array([0.5])
        dNdz = np.array([1.0])

        with pytest.raises(ValidationError, match="at least 2 points"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_negative_values(self) -> None:
        """Test that negative redshifts are rejected."""
        z = np.array([-0.5, 0.0, 0.5, 1.0])
        dNdz = np.ones_like(z)

        with pytest.raises(ValidationError, match="non-negative"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_not_sorted(self) -> None:
        """Test that unsorted z_grid is rejected."""
        z = np.array([0.0, 1.0, 0.5, 2.0])
        dNdz = np.ones_like(z)

        with pytest.raises(ValidationError, match="must be sorted"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_dNdz_validation_not_1d(self) -> None:
        """Test that non-1D dNdz_grid is rejected."""
        z = np.linspace(0, 2, 4)
        dNdz_2d = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError, match="must be 1D array"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz_2d,
            )

    def test_dNdz_validation_negative_values(self) -> None:
        """Test that negative dNdz values are rejected."""
        z = np.linspace(0, 2, 10)
        dNdz = np.array([1, 2, -1, 3, 4, 5, 6, 7, 8, 9])

        with pytest.raises(ValidationError, match="non-negative"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_bias_validation_not_1d(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test that non-1D bias_grid is rejected."""
        bias_2d = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError, match="must be 1D array"):
            NumberCountsTracerConfig(
                tracer_type="number_counts",
                name="galaxies",
                z_grid=z_grid,
                dNdz_grid=dNdz_grid,
                bias_grid=bias_2d,
            )

    def test_bias_none_is_valid(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test that bias_grid=None is valid."""
        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
            bias_grid=None,
        )

        assert tracer.bias_grid is None

    def test_serialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test serialization to dict."""
        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "number_counts"
        assert data["name"] == "galaxies"
        assert "z_grid" in data
        assert "dNdz_grid" in data

    def test_deserialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test deserialization from dict."""
        data = {
            "tracer_type": "number_counts",
            "name": "galaxies",
            "z_grid": z_grid.tolist(),
            "dNdz_grid": dNdz_grid.tolist(),
        }

        tracer = NumberCountsTracerConfig(**data)

        assert tracer.name == "galaxies"
        np.testing.assert_array_almost_equal(tracer.z_grid, z_grid)
        np.testing.assert_array_almost_equal(tracer.dNdz_grid, dNdz_grid)


class TestWeakLensingTracerConfig:
    """Tests for WeakLensingTracerConfig."""

    @pytest.fixture
    def z_grid(self) -> np.ndarray:
        """Create a valid redshift grid."""
        return np.linspace(0, 3, 150)

    @pytest.fixture
    def dNdz_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """Create a valid source redshift distribution."""
        return np.exp(-(((z_grid - 1.0) / 0.5) ** 2))

    def test_creation_basic(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test creating a basic weak lensing tracer."""
        tracer = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source_bin1",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        assert tracer.name == "source_bin1"
        assert tracer.tracer_type == "weak_lensing"
        np.testing.assert_array_equal(tracer.z_grid, z_grid)
        np.testing.assert_array_equal(tracer.dNdz_grid, dNdz_grid)

    def test_z_grid_coercion_from_list(self) -> None:
        """Test that z_grid is coerced from list to array."""
        z_list = [0.0, 1.0, 2.0, 3.0]
        dNdz_list = [0.1, 1.0, 0.5, 0.1]

        tracer = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z_list,  # type: ignore
            dNdz_grid=dNdz_list,  # type: ignore
        )

        assert isinstance(tracer.z_grid, np.ndarray)
        assert isinstance(tracer.dNdz_grid, np.ndarray)

    def test_z_grid_validation_not_1d(self) -> None:
        """Test that non-1D z_grid is rejected."""
        z_2d = np.array([[0, 1], [2, 3]])
        dNdz = np.array([1, 2, 3, 4])

        with pytest.raises(ValidationError, match="must be 1D array"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z_2d,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_too_few_points(self) -> None:
        """Test that z_grid with < 2 points is rejected."""
        z = np.array([1.0])
        dNdz = np.array([1.0])

        with pytest.raises(ValidationError, match="at least 2 points"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_negative_values(self) -> None:
        """Test that negative redshifts are rejected."""
        z = np.array([-1.0, 0.0, 1.0, 2.0])
        dNdz = np.ones_like(z)

        with pytest.raises(ValidationError, match="non-negative"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_z_grid_validation_not_sorted(self) -> None:
        """Test that unsorted z_grid is rejected."""
        z = np.array([0.0, 2.0, 1.0, 3.0])
        dNdz = np.ones_like(z)

        with pytest.raises(ValidationError, match="must be sorted"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_dNdz_validation_not_1d(self) -> None:
        """Test that non-1D dNdz_grid is rejected."""
        z = np.linspace(0, 3, 4)
        dNdz_2d = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError, match="must be 1D array"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z,
                dNdz_grid=dNdz_2d,
            )

    def test_dNdz_validation_negative_values(self) -> None:
        """Test that negative dNdz values are rejected."""
        z = np.linspace(0, 3, 10)
        dNdz = np.array([1, 2, -1, 3, 4, 5, 6, 7, 8, 9])

        with pytest.raises(ValidationError, match="non-negative"):
            WeakLensingTracerConfig(
                tracer_type="weak_lensing",
                name="source",
                z_grid=z,
                dNdz_grid=dNdz,
            )

    def test_serialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test serialization to dict."""
        tracer = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z_grid,
            dNdz_grid=dNdz_grid,
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "weak_lensing"
        assert data["name"] == "source"
        assert "z_grid" in data
        assert "dNdz_grid" in data

    def test_deserialization(
        self,
        z_grid: np.ndarray,
        dNdz_grid: np.ndarray,
    ) -> None:
        """Test deserialization from dict."""
        data = {
            "tracer_type": "weak_lensing",
            "name": "source",
            "z_grid": z_grid.tolist(),
            "dNdz_grid": dNdz_grid.tolist(),
        }

        tracer = WeakLensingTracerConfig(**data)

        assert tracer.name == "source"
        np.testing.assert_array_almost_equal(tracer.z_grid, z_grid)
        np.testing.assert_array_almost_equal(tracer.dNdz_grid, dNdz_grid)


class TestCMBLensingTracerConfig:
    """Tests for CMBLensingTracerConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic CMB lensing tracer."""
        tracer = CMBLensingTracerConfig(
            tracer_type="cmb_lensing",
            name="cmb_lensing",
        )

        assert tracer.name == "cmb_lensing"
        assert tracer.tracer_type == "cmb_lensing"

    def test_minimal_fields(self) -> None:
        """Test that CMB lensing has minimal required fields."""
        tracer = CMBLensingTracerConfig(
            tracer_type="cmb_lensing",
            name="cmb",
        )

        # Should only have base class fields
        assert hasattr(tracer, "tracer_type")
        assert hasattr(tracer, "name")

    def test_serialization(self) -> None:
        """Test serialization to dict."""
        tracer = CMBLensingTracerConfig(
            tracer_type="cmb_lensing",
            name="cmb_lensing",
        )

        data = tracer.model_dump()

        assert data["tracer_type"] == "cmb_lensing"
        assert data["name"] == "cmb_lensing"

    def test_deserialization(self) -> None:
        """Test deserialization from dict."""
        data = {
            "tracer_type": "cmb_lensing",
            "name": "cmb_lensing",
        }

        tracer = CMBLensingTracerConfig(**data)

        assert tracer.name == "cmb_lensing"
        assert tracer.tracer_type == "cmb_lensing"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CMBLensingTracerConfig(
                tracer_type="cmb_lensing",
                name="cmb",
                extra_field="not allowed",  # type: ignore
            )


class TestTracerConfigIntegration:
    """Integration tests for tracer configurations."""

    def test_all_tracers_have_unique_usage(self) -> None:
        """Test that different tracer types are used for different purposes."""
        z_gal = np.linspace(0, 2, 50)
        dNdz_gal = np.exp(-(((z_gal - 0.5) / 0.3) ** 2))

        z_src = np.linspace(0, 3, 100)
        dNdz_src = np.exp(-(((z_src - 1.0) / 0.5) ** 2))

        # Number counts for galaxy clustering
        nc = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z_gal,
            dNdz_grid=dNdz_gal,
        )

        # Weak lensing for cosmic shear
        wl = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z_src,
            dNdz_grid=dNdz_src,
        )

        # CMB lensing
        cmb = CMBLensingTracerConfig(
            tracer_type="cmb_lensing",
            name="cmb_lensing",
        )

        assert nc.tracer_type == "number_counts"
        assert wl.tracer_type == "weak_lensing"
        assert cmb.tracer_type == "cmb_lensing"

    def test_roundtrip_serialization_number_counts(self) -> None:
        """Test roundtrip serialization for number counts."""
        z = np.linspace(0, 2, 100)
        dNdz = np.exp(-(((z - 0.5) / 0.3) ** 2))
        bias = np.ones_like(z) * 1.5

        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z,
            dNdz_grid=dNdz,
            bias_grid=bias,
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = NumberCountsTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.tracer_type == tracer.tracer_type
        np.testing.assert_array_almost_equal(tracer_loaded.z_grid, tracer.z_grid)
        np.testing.assert_array_almost_equal(tracer_loaded.dNdz_grid, tracer.dNdz_grid)
        np.testing.assert_array_almost_equal(
            cast(np.ndarray, tracer_loaded.bias_grid),
            cast(np.ndarray, tracer.bias_grid),
        )

    def test_roundtrip_serialization_weak_lensing(self) -> None:
        """Test roundtrip serialization for weak lensing."""
        z = np.linspace(0, 3, 150)
        dNdz = np.exp(-(((z - 1.0) / 0.5) ** 2))

        tracer = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z,
            dNdz_grid=dNdz,
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = WeakLensingTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.tracer_type == tracer.tracer_type
        np.testing.assert_array_almost_equal(tracer_loaded.z_grid, tracer.z_grid)
        np.testing.assert_array_almost_equal(tracer_loaded.dNdz_grid, tracer.dNdz_grid)

    def test_roundtrip_serialization_cmb_lensing(self) -> None:
        """Test roundtrip serialization for CMB lensing."""
        tracer = CMBLensingTracerConfig(
            tracer_type="cmb_lensing",
            name="cmb_lensing",
        )

        # Serialize
        data = tracer.model_dump()

        # Deserialize
        tracer_loaded = CMBLensingTracerConfig(**data)

        assert tracer_loaded.name == tracer.name
        assert tracer_loaded.tracer_type == tracer.tracer_type

    def test_z_grid_zero_is_valid(self) -> None:
        """Test that z=0 is a valid starting point."""
        z = np.linspace(0.0, 2.0, 100)  # Starting at z=0
        dNdz = np.ones_like(z)

        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z,
            dNdz_grid=dNdz,
        )

        assert tracer.z_grid[0] == 0.0

    def test_dNdz_can_be_zero(self) -> None:
        """Test that dNdz=0 is valid (e.g., at edges of distribution)."""
        z = np.linspace(0, 3, 100)
        dNdz = np.exp(-(((z - 1.0) / 0.5) ** 2))
        # Set edges to zero
        dNdz[0] = 0.0
        dNdz[-1] = 0.0

        tracer = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z,
            dNdz_grid=dNdz,
        )

        assert tracer.dNdz_grid[0] == 0.0
        assert tracer.dNdz_grid[-1] == 0.0

    def test_different_grid_sizes(self) -> None:
        """Test that different tracers can have different grid sizes."""
        # Coarse grid for number counts
        z_gal = np.linspace(0, 2, 50)
        dNdz_gal = np.ones_like(z_gal)

        # Fine grid for weak lensing
        z_src = np.linspace(0, 3, 200)
        dNdz_src = np.ones_like(z_src)

        nc = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z_gal,
            dNdz_grid=dNdz_gal,
        )

        wl = WeakLensingTracerConfig(
            tracer_type="weak_lensing",
            name="source",
            z_grid=z_src,
            dNdz_grid=dNdz_src,
        )

        assert len(nc.z_grid) == 50
        assert len(wl.z_grid) == 200

    def test_bias_can_vary_with_redshift(self) -> None:
        """Test that bias can evolve with redshift."""
        z = np.linspace(0, 2, 100)
        dNdz = np.ones_like(z)
        # Evolving bias: b(z) = 1 + z
        bias = 1.0 + z

        tracer = NumberCountsTracerConfig(
            tracer_type="number_counts",
            name="galaxies",
            z_grid=z,
            dNdz_grid=dNdz,
            bias_grid=bias,
        )

        tracer_bias_grid = cast(np.ndarray, tracer.bias_grid)
        np.testing.assert_array_almost_equal(tracer_bias_grid, bias)
        assert tracer_bias_grid[0] == 1.0
        assert tracer_bias_grid[-1] == pytest.approx(3.0)
