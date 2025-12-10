from __future__ import annotations

from enum import Enum
from typing import TypeAlias, Union

from .enums import CosmologyCalculatorType

try:
    from pyccl.cosmology import Cosmology as CCLCosmology
except ImportError:
    print("Warning, could not import pyccl.cosmology")
    CCLCosmology = None

try:
    from astropy.cosmology import Cosmology as AstropyCosmology
except ImportError:
    print("Warning, could not import astropy.cosmology")
    AstropyCosmology = None


Cosmology: TypeAlias = Union[CCLCosmology, AstropyCosmology]


COSMOLOGY_CLASS_DICT: dict[int, type] = {
    CosmologyCalculatorType.CCL.value: CCLCosmology,
    CosmologyCalculatorType.ASTROPY.value: AstropyCosmology,
}

BaseType = Union[int, float, str, Enum]
