from __future__ import annotations

from enum import Enum
from typing import Any, Type, TypeVar, Union

COSMOLOGY_LIST = []

try:
    from astropy.cosmology import Cosmology as AstropyCosmology
    COSMOLOGY_LIST.append(AstropyCosmology)
except ImportError:
    print("Warning, could not import astropy.cosmology")
    AstropyCosmology = None

try:
    from pyccl.cosmology import Cosmology as CCLCosmology
    COSMOLOGY_LIST.append(CCLCosmology)
except ImportError:
    print("Warning, could not import pyccl.cosmology")
    CCLCosmology = None


Cosmology = Union[*COSMOLOGY_LIST]

BaseType = Union[int, float, str,  Enum]




