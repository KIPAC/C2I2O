from __future__ import annotations

from enum import Enum
from types import UnionType
from typing import Any, Union

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


def make_union(class_list: list[Any]) -> UnionType:
    """Build a Union from a list of classes

    Parameters
    ----------
    class_list:
        Classes we are joining

    Returns
    -------
    Union of the classes
    """
    the_union = class_list[0] | class_list[1]
    for a_class_ in class_list[2:]:
        the_union = the_union | a_class_
    return the_union


Cosmology = make_union(COSMOLOGY_LIST)

COSMOLOGY_CLASS_DICT: dict[int, type] = {
    CosmologyCalculatorType.ccl.value: CCLCosmology,
    CosmologyCalculatorType.astropy.value: AstropyCosmology,
}

BaseType = Union[int, float, str, Enum]
