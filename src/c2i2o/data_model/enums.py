from __future__ import annotations

from enum import Enum


class CosmologyCalculatorType(Enum):
    """Enum for the types of cosmology libraries"""

    CCL = 0
    ASTROPY = 1
