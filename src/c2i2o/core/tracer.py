"""Tracer definitions for cosmological observables in c2i2o.

This module provides classes for representing tracers of cosmological observables.
Tracers combine multiple components (radial kernels, transfer functions, prefactors)
that are summed to compute the final observable.

This module also provides abstract base classes for tracer configurations used
in cosmological observations like galaxy clustering, weak lensing, and
CMB lensing.
"""

from abc import ABC

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from c2i2o.core.tensor import TensorBase


class TracerConfigBase(BaseModel, ABC):
    """Abstract base class for tracer configurations.

    This class provides common functionality for all tracer types,
    including parameter validation and serialization.

    Attributes
    ----------
    tracer_type
        String identifier for the tracer type.
    name
        Unique identifier for this tracer instance.

    Notes
    -----
    Subclasses must implement specific tracer functionality and
    define their tracer_type as a Literal for discriminated unions.
    """

    tracer_type: str = Field(..., description="Type identifier for the tracer")
    name: str = Field(..., description="Unique name for this tracer")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty.

        Parameters
        ----------
        v
            Name to validate.

        Returns
        -------
            Validated name.

        Raises
        ------
        ValueError
            If name is empty.
        """
        if not v or not v.strip():
            raise ValueError("Tracer name cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class NumberCountsTracerConfig(TracerConfigBase):
    """Base class for galaxy number counts tracer configuration.

    This class configures a tracer for galaxy number counts (clustering)
    observations. It requires a redshift distribution n(z) and optional
    galaxy bias evolution b(z).

    Attributes
    ----------
    tracer_type
        Type identifier for the tracer.
    name
        Unique identifier for this tracer.
    z_grid
        Redshift grid for n(z) and b(z) evaluation.
    dNdz_grid
        Galaxy redshift distribution dN/dz values on z_grid.
    bias_grid
        Optional galaxy bias b(z) values on z_grid.
        If None, assumes b(z) = 1.

    Examples
    --------
    >>> z = np.linspace(0, 2, 100)
    >>> dNdz = np.exp(-((z - 0.5) / 0.3)**2)
    >>> bias = np.ones_like(z) * 1.5
    >>> tracer = NumberCountsTracerConfig(
    ...     tracer_type="number_counts",
    ...     name="galaxies_bin1",
    ...     z_grid=z,
    ...     dNdz_grid=dNdz,
    ...     bias_grid=bias,
    ... )
    """

    z_grid: np.ndarray = Field(..., description="Redshift grid")
    dNdz_grid: np.ndarray = Field(..., description="Redshift distribution dN/dz")
    bias_grid: np.ndarray | None = Field(
        default=None,
        description="Galaxy bias b(z) (optional, default=1)",
    )

    @field_validator("z_grid", mode="before")
    @classmethod
    def coerce_z_grid_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce z_grid to NumPy array if needed.

        Parameters
        ----------
        v
            Redshift grid values.

        Returns
        -------
            NumPy array.
        """
        return np.asarray(v)

    @field_validator("dNdz_grid", mode="before")
    @classmethod
    def coerce_dndz_grid_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce dNdz_grid to NumPy array if needed.

        Parameters
        ----------
        v
            Redshift distribution values.

        Returns
        -------
            NumPy array.
        """
        return np.asarray(v)

    @field_validator("bias_grid", mode="before")
    @classmethod
    def coerce_bias_grid_to_array(cls, v: np.ndarray | list | None) -> np.ndarray | None:
        """Coerce bias_grid to NumPy array if needed.

        Parameters
        ----------
        v
            Bias values.

        Returns
        -------
            NumPy array or None.
        """
        if v is None:
            return None
        return np.asarray(v)

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_1d(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid is 1D.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If not 1D array.
        """
        if v.ndim != 1:
            raise ValueError(f"z_grid must be 1D array, got shape {v.shape}")
        if len(v) < 2:
            raise ValueError(f"z_grid must have at least 2 points, got {len(v)}")
        return v

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_positive(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid values are non-negative.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If any redshift is negative.
        """
        if np.any(v < 0):
            raise ValueError("All redshift values must be non-negative")
        return v

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_sorted(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid is sorted.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If not sorted.
        """
        if not np.all(v[1:] >= v[:-1]):
            raise ValueError("z_grid must be sorted in ascending order")
        return v

    @field_validator("dNdz_grid")
    @classmethod
    def validate_dndz_grid_shape(cls, v: np.ndarray, _info: ValidationInfo) -> np.ndarray:
        """Validate that dNdz_grid has same shape as z_grid.

        Parameters
        ----------
        v
            Redshift distribution.
        info
            Validation context.

        Returns
        -------
            Validated distribution.

        Raises
        ------
        ValueError
            If shapes don't match.
        """
        if v.ndim != 1:
            raise ValueError(f"dNdz_grid must be 1D array, got shape {v.shape}")
        return v

    @field_validator("dNdz_grid")
    @classmethod
    def validate_dndz_grid_non_negative(cls, v: np.ndarray) -> np.ndarray:
        """Validate that dNdz_grid is non-negative.

        Parameters
        ----------
        v
            Redshift distribution.

        Returns
        -------
            Validated distribution.

        Raises
        ------
        ValueError
            If any values are negative.
        """
        if np.any(v < 0):
            raise ValueError("dNdz_grid values must be non-negative")
        return v

    @field_validator("bias_grid")
    @classmethod
    def validate_bias_grid_shape(cls, v: np.ndarray | None, _info: ValidationInfo) -> np.ndarray | None:
        """Validate that bias_grid has same shape as z_grid.

        Parameters
        ----------
        v
            Bias values.
        info
            Validation context.

        Returns
        -------
            Validated bias.

        Raises
        ------
        ValueError
            If shapes don't match.
        """
        if v is None:
            return None

        if v.ndim != 1:
            raise ValueError(f"bias_grid must be 1D array, got shape {v.shape}")

        return v


class WeakLensingTracerConfig(TracerConfigBase):
    """Base class for weak gravitational lensing tracer configuration.

    This class configures a tracer for weak lensing (cosmic shear)
    observations. It requires a source galaxy redshift distribution n(z).

    Attributes
    ----------
    tracer_type
        Type identifier for the tracer.
    name
        Unique identifier for this tracer.
    z_grid
        Redshift grid for n(z) evaluation.
    dNdz_grid
        Source galaxy redshift distribution dN/dz values on z_grid.

    Examples
    --------
    >>> z = np.linspace(0, 3, 150)
    >>> dNdz = np.exp(-((z - 1.0) / 0.5)**2)
    >>> tracer = WeakLensingTracerConfig(
    ...     tracer_type="weak_lensing",
    ...     name="source_bin1",
    ...     z_grid=z,
    ...     dNdz_grid=dNdz,
    ... )
    """

    z_grid: np.ndarray = Field(..., description="Redshift grid")
    dNdz_grid: np.ndarray = Field(..., description="Source redshift distribution dN/dz")

    @field_validator("z_grid", mode="before")
    @classmethod
    def coerce_z_grid_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce z_grid to NumPy array if needed.

        Parameters
        ----------
        v
            Redshift grid values.

        Returns
        -------
            NumPy array.
        """
        return np.asarray(v)

    @field_validator("dNdz_grid", mode="before")
    @classmethod
    def coerce_dndz_grid_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce dNdz_grid to NumPy array if needed.

        Parameters
        ----------
        v
            Redshift distribution values.

        Returns
        -------
            NumPy array.
        """
        return np.asarray(v)

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_1d(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid is 1D.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If not 1D array.
        """
        if v.ndim != 1:
            raise ValueError(f"z_grid must be 1D array, got shape {v.shape}")
        if len(v) < 2:
            raise ValueError(f"z_grid must have at least 2 points, got {len(v)}")
        return v

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_positive(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid values are non-negative.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If any redshift is negative.
        """
        if np.any(v < 0):
            raise ValueError("All redshift values must be non-negative")
        return v

    @field_validator("z_grid")
    @classmethod
    def validate_z_grid_sorted(cls, v: np.ndarray) -> np.ndarray:
        """Validate that z_grid is sorted.

        Parameters
        ----------
        v
            Redshift grid.

        Returns
        -------
            Validated redshift grid.

        Raises
        ------
        ValueError
            If not sorted.
        """
        if not np.all(v[1:] >= v[:-1]):
            raise ValueError("z_grid must be sorted in ascending order")
        return v

    @field_validator("dNdz_grid")
    @classmethod
    def validate_dndz_grid_shape(cls, v: np.ndarray, _info: ValidationInfo) -> np.ndarray:
        """Validate that dNdz_grid has same shape as z_grid.

        Parameters
        ----------
        v
            Redshift distribution.
        info
            Validation context.

        Returns
        -------
            Validated distribution.

        Raises
        ------
        ValueError
            If shapes don't match.
        """
        if v.ndim != 1:
            raise ValueError(f"dNdz_grid must be 1D array, got shape {v.shape}")
        return v

    @field_validator("dNdz_grid")
    @classmethod
    def validate_dndz_grid_non_negative(cls, v: np.ndarray) -> np.ndarray:
        """Validate that dNdz_grid is non-negative.

        Parameters
        ----------
        v
            Redshift distribution.

        Returns
        -------
            Validated distribution.

        Raises
        ------
        ValueError
            If any values are negative.
        """
        if np.any(v < 0):
            raise ValueError("dNdz_grid values must be non-negative")
        return v


class CMBLensingTracerConfig(TracerConfigBase):
    """Base class for CMB lensing tracer configuration.

    This class configures a tracer for CMB lensing convergence observations.
    The source is at the last scattering surface (z ~ 1100).

    Attributes
    ----------
    tracer_type
        Type identifier for the tracer.
    name
        Unique identifier for this tracer (typically "cmb_lensing").

    Examples
    --------
    >>> tracer = CMBLensingTracerConfig(
    ...     tracer_type="cmb_lensing",
    ...     name="cmb_lensing",
    ... )

    Notes
    -----
    CMB lensing does not require a redshift distribution n(z) since
    the source is at a fixed redshift (last scattering surface).
    """


__all__ = [
    "TracerConfigBase",
    "NumberCountsTracerConfig",
    "WeakLensingTracerConfig",
    "CMBLensingTracerConfig",
]


class TracerElement(BaseModel):
    """Single element of a cosmological tracer.

    A tracer element represents one component that contributes to a cosmological
    observable. It contains tensors for the radial kernel, transfer function,
    and prefactor, along with derivative orders for Bessel functions and angular
    terms.

    Attributes
    ----------
    radial_kernel
        Radial kernel function as a function of redshift/distance (optional).
    transfer_function
        Transfer function as a function of wavenumber (optional).
    prefactor
        Multiplicative prefactor (optional).
    bessel_derivative
        Order of derivative for Bessel function (default: 0).
    angles_derivative
        Order of derivative for angular terms (default: 0).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=50)
    >>> kernel_values = np.ones(50)
    >>> kernel_tensor = NumpyTensor(grid=z_grid, values=kernel_values)
    >>>
    >>> element = TracerElement(
    ...     radial_kernel=kernel_tensor,
    ...     bessel_derivative=0,
    ... )
    """

    radial_kernel: TensorBase | None = Field(
        default=None, description="Radial kernel as function of redshift/distance"
    )
    transfer_function: TensorBase | None = Field(
        default=None, description="Transfer function as function of wavenumber"
    )
    prefactor: TensorBase | None = Field(default=None, description="Multiplicative prefactor")
    bessel_derivative: int = Field(default=0, ge=0, description="Order of Bessel function derivative")
    angles_derivative: int = Field(default=0, ge=0, description="Order of angular derivative")

    def __repr__(self) -> str:
        """Return string representation of the tracer element.

        Returns
        -------
            String representation.
        """
        components = []
        if self.radial_kernel is not None:
            components.append("radial_kernel")
        if self.transfer_function is not None:
            components.append("transfer_function")
        if self.prefactor is not None:
            components.append("prefactor")

        comp_str = ", ".join(components) if components else "empty"
        return (
            f"TracerElement({comp_str}, "
            f"bessel_der={self.bessel_derivative}, angles_der={self.angles_derivative})"
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class Tracer(BaseModel):
    """Collection of tracer elements for a cosmological observable.

    A tracer combines multiple tracer elements that are summed to compute
    the final observable. It provides methods to extract and sum the
    individual components across all elements.

    Attributes
    ----------
    elements
        List of TracerElement objects that comprise this tracer.
    name
        Optional name for the tracer.
    description
        Optional description of the tracer.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=50)
    >>>
    >>> # Create first element
    >>> kernel1 = NumpyTensor(grid=z_grid, values=np.ones(50))
    >>> element1 = TracerElement(radial_kernel=kernel1)
    >>>
    >>> # Create second element
    >>> kernel2 = NumpyTensor(grid=z_grid, values=np.ones(50) * 2.0)
    >>> element2 = TracerElement(radial_kernel=kernel2)
    >>>
    >>> # Create tracer
    >>> tracer = Tracer(elements=[element1, element2], name="density")
    >>> len(tracer)
    2
    >>>
    >>> # Sum radial kernels
    >>> summed = tracer.sum_radial_kernels()
    >>> # Result is on z_grid with values = 1 + 2 = 3
    """

    elements: list[TracerElement] = Field(..., description="List of tracer elements")
    name: str | None = Field(default=None, description="Name of the tracer")
    description: str | None = Field(default=None, description="Description of the tracer")

    @field_validator("elements")
    @classmethod
    def validate_non_empty(cls, v: list[TracerElement]) -> list[TracerElement]:
        """Validate that elements list is not empty."""
        if not v:
            raise ValueError("Tracer must contain at least one element")
        return v

    def get_radial_kernels(self) -> list[TensorBase | None]:
        """Get list of radial kernels from all elements.

        Returns
        -------
            List of radial kernel tensors (may contain None).

        Examples
        --------
        >>> kernels = tracer.get_radial_kernels()
        >>> len(kernels) == len(tracer)
        True
        """
        return [element.radial_kernel for element in self.elements]

    def get_transfer_functions(self) -> list[TensorBase | None]:
        """Get list of transfer functions from all elements.

        Returns
        -------
            List of transfer function tensors (may contain None).

        Examples
        --------
        >>> transfers = tracer.get_transfer_functions()
        """
        return [element.transfer_function for element in self.elements]

    def get_prefactors(self) -> list[TensorBase | None]:
        """Get list of prefactors from all elements.

        Returns
        -------
            List of prefactor tensors (may contain None).

        Examples
        --------
        >>> prefactors = tracer.get_prefactors()
        """
        return [element.prefactor for element in self.elements]

    def get_bessel_derivatives(self) -> list[int]:
        """Get list of Bessel derivative orders from all elements.

        Returns
        -------
            List of Bessel derivative orders.

        Examples
        --------
        >>> bessel_ders = tracer.get_bessel_derivatives()
        """
        return [element.bessel_derivative for element in self.elements]

    def get_angles_derivatives(self) -> list[int]:
        """Get list of angular derivative orders from all elements.

        Returns
        -------
            List of angular derivative orders.

        Examples
        --------
        >>> angle_ders = tracer.get_angles_derivatives()
        """
        return [element.angles_derivative for element in self.elements]

    def sum_radial_kernels(self) -> TensorBase:
        """Sum all radial kernels on their common grid.

        Returns
        -------
            Tensor containing the sum of all radial kernels.

        Raises
        ------
        ValueError
            If no radial kernels are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_kernel = tracer.sum_radial_kernels()

        Notes
        -----
        All radial kernels must be defined on the same grid. The sum is
        computed by adding the underlying tensor values.
        """
        kernels = [k for k in self.get_radial_kernels() if k is not None]

        if not kernels:
            raise ValueError("No radial kernels to sum")

        # Check that all grids are compatible (same grid object or equivalent)
        first_grid = kernels[0].grid
        for kernel in kernels[1:]:
            if kernel.grid != first_grid:
                raise ValueError("All radial kernels must be defined on the same grid for summation")

        # Sum the values
        summed_values = kernels[0].get_values().copy()
        for kernel in kernels[1:]:
            summed_values = summed_values + kernel.get_values()

        # Create new tensor with summed values
        # Use the same class as the first kernel
        result = kernels[0].__class__(grid=first_grid, values=summed_values)  # type: ignore
        return result

    def sum_transfer_functions(self) -> TensorBase:
        """Sum all transfer functions on their common grid.

        Returns
        -------
            Tensor containing the sum of all transfer functions.

        Raises
        ------
        ValueError
            If no transfer functions are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_transfer = tracer.sum_transfer_functions()

        Notes
        -----
        All transfer functions must be defined on the same grid.
        """
        transfers = [t for t in self.get_transfer_functions() if t is not None]

        if not transfers:
            raise ValueError("No transfer functions to sum")

        # Check grid compatibility
        first_grid = transfers[0].grid
        for transfer in transfers[1:]:
            if transfer.grid != first_grid:  # pragma: no cover
                raise ValueError("All transfer functions must be defined on the same grid for summation")

        # Sum the values
        summed_values = transfers[0].get_values().copy()
        for transfer in transfers[1:]:
            summed_values = summed_values + transfer.get_values()

        # Create new tensor with summed values
        result = transfers[0].__class__(grid=first_grid, values=summed_values)  # type: ignore
        return result

    def sum_prefactors(self) -> TensorBase:
        """Sum all prefactors on their common grid.

        Returns
        -------
            Tensor containing the sum of all prefactors.

        Raises
        ------
        ValueError
            If no prefactors are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_prefactor = tracer.sum_prefactors()

        Notes
        -----
        All prefactors must be defined on the same grid.
        """
        prefactors = [p for p in self.get_prefactors() if p is not None]

        if not prefactors:
            raise ValueError("No prefactors to sum")

        # Check grid compatibility
        first_grid = prefactors[0].grid
        for prefactor in prefactors[1:]:
            if prefactor.grid != first_grid:  # pragma: no cover
                raise ValueError("All prefactors must be defined on the same grid for summation")

        # Sum the values
        summed_values = prefactors[0].get_values().copy()
        for prefactor in prefactors[1:]:
            summed_values = summed_values + prefactor.get_values()

        # Create new tensor with summed values
        result = prefactors[0].__class__(grid=first_grid, values=summed_values)  # type: ignore
        return result

    def add_element(self, element: TracerElement) -> None:
        """Add a tracer element to the tracer.

        Parameters
        ----------
        element
            TracerElement to add.

        Examples
        --------
        >>> new_element = TracerElement(radial_kernel=kernel)
        >>> tracer.add_element(new_element)
        """
        self.elements.append(element)

    def remove_element(self, index: int) -> TracerElement:
        """Remove and return a tracer element by index.

        Parameters
        ----------
        index
            Index of the element to remove.

        Returns
        -------
            The removed TracerElement.

        Raises
        ------
        IndexError
            If index is out of range.

        Examples
        --------
        >>> removed = tracer.remove_element(0)
        """
        return self.elements.pop(index)

    def __len__(self) -> int:
        """Return the number of tracer elements.

        Returns
        -------
            Number of elements.
        """
        return len(self.elements)

    def __getitem__(self, index: int) -> TracerElement:
        """Get a tracer element by index.

        Parameters
        ----------
        index
            Index of the element.

        Returns
        -------
            The TracerElement at the given index.
        """
        return self.elements[index]

    def __repr__(self) -> str:
        """Return string representation of the tracer.

        Returns
        -------
            String representation.
        """
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return f"Tracer({name_str}, n_elements={len(self)})"

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "TracerElement",
    "Tracer",
    "TracerConfigBase",
    "NumberCountsTracerConfig",
    "WeakLensingTracerConfig",
    "CMBLensingTracerConfig",
]
