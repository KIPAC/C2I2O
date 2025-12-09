from __future__ import annotations

from enum import Enum
from typing import Any, Type, TypeVar, Union

from .enums import CosmologyCalculatorType

COSMOLOGY_LIST = []

try:
    from pyccl.cosmology import Cosmology as CCLCosmology
    COSMOLOGY_LIST.append(CCLCosmology)
except ImportError:
    print("Warning, could not import pyccl.cosmology")
    CCLCosmology = None

try:
    from astropy.cosmology import Cosmology as AstropyCosmology
    COSMOLOGY_LIST.append(AstropyCosmology)
except ImportError:
    print("Warning, could not import astropy.cosmology")
    AstropyCosmology = None


Cosmology = Union[*COSMOLOGY_LIST]

COSMOLOGY_CLASS_DICT: dict[int, Cosmology] = {
    CosmologyCalculatorType.ccl.value:CCLCosmology,
    CosmologyCalculatorType.astropy.value:AstropyCosmology,
}
                               

BaseType = Union[int, float, str,  Enum]


    
    
