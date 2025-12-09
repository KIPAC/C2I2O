from __future__ import annotations

from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, Field


class GridParams(BaseModel):
    """Base class for parameters for a grid"""

    n_values: int = Field(101, description="Number of values for grid")

    def build_grid(self) -> np.ndarray:
        """Construct the grid"""
        raise NotImplementedError()


class LinearGridParams(GridParams):
    """Parameters for a linear grid

    Notes
    -----
    This will produce a grid using

    code
        np.linspace(min_value, max_value, n_vals)
    """

    grid_type: Literal["linear"] = Field(default="", description="Grid type.")

    min_value: float = Field(..., description="Minimum value for grid")
    max_value: float = Field(..., description="Maximum value for grid")

    def build_grid(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, self.n_values)


class LogGridParams(GridParams):
    """Parameters for a logarithmic grid

    Notes
    -----
    min_value and max_value are the values themselves, not to the log10 of
    the values

    I.e., this will produce a grid using

    code
        np.logspace(np.log10(z_min), np.log10(z_max), n_zvals)
    """

    grid_type: Literal["log"] = Field(default="", description="Grid type.")

    min_value: float = Field(..., description="Minimum value for grid", ge=0.0)
    max_value: float = Field(..., description="Maximum value for grid", ge=0.0)

    def build_grid(self) -> np.ndarray:
        return np.logspace(
            np.log10(self.min_value), np.log10(self.max_value), self.n_values
        )


GridParamsUnion = Union[
    LinearGridParams,
    LogGridParams,
]


class LinearZGridParams(LinearGridParams):
    """Parameters for a linear grid in redshift"""

    grid_type: Literal["linear"] = Field(default="linear", description="Grid type.")

    # Constrain values to be positive
    min_value: float = Field(0.0, description="Minimum value for grid", ge=0.0)
    max_value: float = Field(..., description="Maximum value for grid")


class LogZGridParams(LogGridParams):
    """Parameters for a logarithmic grid in redshift"""

    grid_type: Literal["log"] = Field(default="log", description="Grid type.")


class LinearAGridParams(LinearGridParams):
    """Parameters for a linear grid in scale factor"""

    grid_type: Literal["linear"] = Field(default="linear", description="Grid type.")

    min_value: float = Field(..., description="Minimum value for grid", gt=0.0, le=1.0)
    max_value: float = Field(..., description="Maximum value for grid", gt=0.0, le=1.0)


class LogAGridParams(LogGridParams):
    """Parameters for a logarithmic grid in scale factor"""

    grid_type: Literal["log"] = Field(default="log", description="Grid type.")

    min_value: float = Field(..., description="Minimum value for grid", gt=0.0, le=1.0)
    max_value: float = Field(..., description="Maximum value for grid", gt=0.0, le=1.0)


class LogKGridParams(LogGridParams):
    """Parameters for a logarithmic grid in wavenumber

    Notes
    -----
    This sets up a grid for the wavenumber, in Mpc^1, which
    is used for evaluation of various 3D power spectra
    """

    grid_type: Literal["log"] = Field(default="log", description="Grid type.")

    min_value: float = Field(
        1e-4, description="Minimum value for wavenumber grid [Mpc-1]", gt=0.0
    )
    max_value: float = Field(
        1e1, description="Maximum value for wavenumber grid [Mpc-1]", gt=0.0
    )
