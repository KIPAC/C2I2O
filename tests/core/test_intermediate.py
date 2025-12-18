"""Tests for c2i2o.core.intermediate module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.core.tensor import NumpyTensor


class TestIntermediateBase:
    """Tests for IntermediateBase class."""

    def test_initialization(self, simple_intermediate: IntermediateBase) -> None:
        """Test basic initialization."""
        assert simple_intermediate.name == "test_intermediate"
        assert simple_intermediate.units == "Mpc"
        assert simple_intermediate.description == "Test intermediate product"
        assert simple_intermediate.tensor is not None

    def test_initialization_minimal(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test initialization with minimal parameters."""
        intermediate = IntermediateBase(
            name="minimal",
            tensor=simple_numpy_tensor_1d,
        )
        assert intermediate.name == "minimal"
        assert intermediate.units is None
        assert intermediate.description is None

    def test_evaluate(self, simple_intermediate: IntermediateBase) -> None:
        """Test evaluate method."""
        result = simple_intermediate.evaluate(np.array([2.5, 5.0]))
        expected = np.array([6.5, 25.0])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_dict(self, simple_intermediate: IntermediateBase) -> None:
        """Test evaluate with dict input."""
        result = simple_intermediate.evaluate({"x": np.array([3.0])})
        np.testing.assert_allclose(result, [9.0])

    def test_get_values(self, simple_intermediate: IntermediateBase) -> None:
        """Test get_values returns tensor values."""
        values = simple_intermediate.get_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (11,)

    def test_set_values(self, simple_intermediate: IntermediateBase) -> None:
        """Test set_values updates tensor values."""
        new_values = np.ones(11) * 10.0
        simple_intermediate.set_values(new_values)
        np.testing.assert_array_equal(simple_intermediate.get_values(), new_values)

    def test_shape_property(self, simple_intermediate: IntermediateBase) -> None:
        """Test shape property."""
        assert simple_intermediate.shape == (11,)

    def test_ndim_property(self, simple_intermediate: IntermediateBase) -> None:
        """Test ndim property."""
        assert simple_intermediate.ndim == 1

    def test_grid_property(self, simple_intermediate: IntermediateBase, simple_grid_1d: Grid1D) -> None:
        """Test grid property."""
        assert simple_intermediate.grid == simple_grid_1d

    def test_repr(self, simple_intermediate: IntermediateBase) -> None:
        """Test string representation."""
        repr_str = repr(simple_intermediate)
        assert "IntermediateBase" in repr_str
        assert "test_intermediate" in repr_str
        assert "shape=(11,)" in repr_str
        assert "units=Mpc" in repr_str


class TestIntermediateSet:
    """Tests for IntermediateSet class."""

    def test_initialization(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test basic initialization."""
        assert len(simple_intermediate_set.intermediates) == 2
        assert "intermediate1" in simple_intermediate_set.intermediates
        assert "intermediate2" in simple_intermediate_set.intermediates

    def test_initialization_with_description(self, simple_intermediate: IntermediateBase) -> None:
        """Test initialization with description."""
        intermediate_set = IntermediateSet(
            intermediates={"test_intermediate": simple_intermediate},
            description="Test set",
        )
        assert intermediate_set.description == "Test set"

    def test_validation_empty_raises_error(self) -> None:
        """Test validation error with empty intermediates dict."""
        with pytest.raises(ValidationError, match="at least one intermediate"):
            IntermediateSet(intermediates={})

    def test_validation_names_match_keys(self, simple_intermediate: IntermediateBase) -> None:
        """Test validation that intermediate names match dictionary keys."""
        with pytest.raises(ValidationError, match="does not match dictionary key"):
            IntermediateSet(intermediates={"wrong_key": simple_intermediate})

    def test_names_property(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test names property returns sorted list."""
        names = simple_intermediate_set.names
        assert names == ["intermediate1", "intermediate2"]

    def test_get(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get method."""
        intermediate = simple_intermediate_set.get("intermediate1")
        assert intermediate.name == "intermediate1"

    def test_get_raises_keyerror(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get raises KeyError for missing intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.get("nonexistent")

    def test_getitem(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __getitem__ bracket notation."""
        intermediate = simple_intermediate_set["intermediate1"]
        assert intermediate.name == "intermediate1"

    def test_contains(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __contains__ membership test."""
        assert "intermediate1" in simple_intermediate_set
        assert "intermediate2" in simple_intermediate_set
        assert "nonexistent" not in simple_intermediate_set

    def test_len(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __len__ returns number of intermediates."""
        assert len(simple_intermediate_set) == 2

    def test_evaluate(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate single intermediate."""
        result = simple_intermediate_set.evaluate("intermediate1", np.array([5.0]))
        np.testing.assert_allclose(result, [25.0])

    def test_evaluate_raises_keyerror(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate raises KeyError for missing intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.evaluate("nonexistent", np.array([1.0]))

    def test_evaluate_all(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate_all method."""
        points = {
            "intermediate1": np.array([2.0, 3.0]),
            "intermediate2": np.array([1.0, 2.0]),
        }
        results = simple_intermediate_set.evaluate_all(points)

        assert isinstance(results, dict)
        assert len(results) == 2
        np.testing.assert_allclose(results["intermediate1"], [4.0, 9.0])
        np.testing.assert_allclose(results["intermediate2"], [1.0, 1.0])

    def test_evaluate_all_missing_points_raises_error(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate_all raises error when points are missing."""
        points = {"intermediate1": np.array([1.0])}  # Missing intermediate2
        with pytest.raises(KeyError, match="missing"):
            simple_intermediate_set.evaluate_all(points)

    def test_get_values_dict(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get_values_dict returns all values."""
        values_dict = simple_intermediate_set.get_values_dict()

        assert isinstance(values_dict, dict)
        assert len(values_dict) == 2
        assert "intermediate1" in values_dict
        assert "intermediate2" in values_dict
        assert values_dict["intermediate1"].shape == (11,)
        assert values_dict["intermediate2"].shape == (11,)

    def test_set_values_dict(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test set_values_dict updates multiple intermediates."""
        new_values = {
            "intermediate1": np.ones(11) * 5.0,
            "intermediate2": np.ones(11) * 10.0,
        }
        simple_intermediate_set.set_values_dict(new_values)

        values_dict = simple_intermediate_set.get_values_dict()
        np.testing.assert_array_equal(values_dict["intermediate1"], np.ones(11) * 5.0)
        np.testing.assert_array_equal(values_dict["intermediate2"], np.ones(11) * 10.0)

    def test_set_values_dict_missing_intermediate_raises_error(
        self, simple_intermediate_set: IntermediateSet
    ) -> None:
        """Test set_values_dict raises error for missing intermediate."""
        new_values = {"nonexistent": np.ones(11)}
        with pytest.raises(KeyError, match="not found"):
            simple_intermediate_set.set_values_dict(new_values)

    def test_add(self, simple_intermediate_set: IntermediateSet, simple_grid_1d: Grid1D) -> None:
        """Test add method."""
        new_intermediate = IntermediateBase(
            name="intermediate3",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 2.0),
        )
        simple_intermediate_set.add(new_intermediate)

        assert len(simple_intermediate_set) == 3
        assert "intermediate3" in simple_intermediate_set

    def test_add_duplicate_raises_error(
        self, simple_intermediate_set: IntermediateSet, simple_grid_1d: Grid1D
    ) -> None:
        """Test add raises error for duplicate name."""
        duplicate = IntermediateBase(
            name="intermediate1",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        )
        with pytest.raises(ValueError, match="already exists"):
            simple_intermediate_set.add(duplicate)

    def test_remove(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test remove method."""
        removed = simple_intermediate_set.remove("intermediate1")

        assert removed.name == "intermediate1"
        assert len(simple_intermediate_set) == 1
        assert "intermediate1" not in simple_intermediate_set

    def test_remove_nonexistent_raises_error(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test remove raises error for nonexistent intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.remove("nonexistent")

    def test_repr(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test string representation."""
        repr_str = repr(simple_intermediate_set)
        assert "IntermediateSet" in repr_str
        assert "n_intermediates=2" in repr_str
        assert "intermediate1" in repr_str
        assert "intermediate2" in repr_str


class TestIntermediateSetOperations:
    """Tests for complex operations on IntermediateSet."""

    def test_add_and_remove_sequence(self, simple_grid_1d: Grid1D) -> None:
        """Test sequence of add and remove operations."""
        # Start with one intermediate
        intermediate1 = IntermediateBase(
            name="int1",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        )
        intermediate_set = IntermediateSet(intermediates={"int1": intermediate1})

        # Add two more
        intermediate2 = IntermediateBase(
            name="int2",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 2),
        )
        intermediate3 = IntermediateBase(
            name="int3",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 3),
        )
        intermediate_set.add(intermediate2)
        intermediate_set.add(intermediate3)
        assert len(intermediate_set) == 3

        # Remove one
        intermediate_set.remove("int2")
        assert len(intermediate_set) == 2
        assert "int2" not in intermediate_set
        assert "int1" in intermediate_set
        assert "int3" in intermediate_set

    def test_batch_update_values(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test batch updating values."""
        # Get current values
        old_values = simple_intermediate_set.get_values_dict()

        # Update all values
        new_values = {name: vals * 2.0 for name, vals in old_values.items()}
        simple_intermediate_set.set_values_dict(new_values)

        # Verify update
        updated_values = simple_intermediate_set.get_values_dict()
        for name in simple_intermediate_set.names:
            np.testing.assert_array_equal(updated_values[name], old_values[name] * 2.0)

    def test_iteration_over_names(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test iterating over intermediate names."""
        count = 0
        for name in simple_intermediate_set.names:
            intermediate = simple_intermediate_set[name]
            assert intermediate.name == name
            count += 1
        assert count == 2
