from typing import Any, Literal, TypeAlias, Union

import pyccl as ccl
from pydantic import BaseModel, Field

from c2i2o.data_model.base_classes import Cosmology

Baryons: TypeAlias = dict
ModifiedGravity: TypeAlias = dict


class CosmologyParams(BaseModel):
    """Base class pydantic models for cosmology parameter sets"""

    cosmology_type: str = Field(..., description="Type of cosmology parameter set")

    def __init__(self, **kwargs: Any):
        BaseModel.__init__(self, **kwargs)
        self._cosomlogy: Cosmology | None = None

    def build_cosmology(self) -> Cosmology:
        """Build a Cosmology calculator"""
        raise NotImplementedError()

    def build_baryonic_effects(self, baryons: Baryons) -> Any:
        """Build a Baryon effects calculator"""
        raise NotImplementedError()

    def build_mg_parametrization(self, mg: ModifiedGravity) -> Any:
        """Build a ModifiedGravity calculator"""
        raise NotImplementedError()


class CCLCosmologyVanillaLCDMParams(CosmologyParams):
    """CCL VanillaLCDM Specific Parameter set"""

    cosmology_type: Literal["ccl.vanillaLCDM"] = Field(
        default="", description="Type of cosmology parameter set"
    )

    Omega_k: float = Field(
        default=0.0,
        description="Curvature density fraction density fraction",
        ge=-1.0,
        le=1.0,
    )
    Omega_g: float | None = Field(
        default=None,
        description="Density in relativistic species except massless neutrinos",
        ge=0.0,
        le=1.0,
    )
    Neff: float = Field(
        default=3.044,
        description="Effective number of massless neutrinos present.",
        ge=3.0,
        le=3.2,
    )
    m_nu: float = Field(
        default=0.0, description="Mass in eV of the massive neutrinos present."
    )
    w0: float = Field(
        default=-1.0,
        description="First order term of dark energy equation of state.",
        ge=-2.0,
        le=1.0,
    )
    wa: float = Field(
        default=0.0,
        description="Second order term of dark energy equation of state.",
        ge=-2.0,
        le=2.0,
    )
    T_CMB: float = Field(default=2.7255, description="The CMB temperature today.")
    T_ncdm: float = Field(
        default=0.71611,
        description="Non-CDM temperature in units of photon temperature.",
    )

    mass_split: str = Field(default="normal", description="Type of massive neutrinos.")
    transfer_function: str = Field(
        default="boltzmann_camb", description="The transfer function to use."
    )
    matter_power_spectrum: str = Field(
        default="halofit", description="The matter power spectrum to use."
    )

    baryonic_effects: Baryons | None = Field(
        default=None, description="The baryonic effects model to use."
    )
    mg_parametrization: ModifiedGravity | None = Field(
        default=None, description="The modified gravity parametrization to use."
    )

    extra_parameters: dict = {}

    def build_cosmology(self) -> ccl.Cosmology:

        kwargs_dict = self.dict()

        baryonic_effects_ = kwargs_dict.pop("baryonic_effects")
        if baryonic_effects_ is not None:
            kwargs_dict["baryonic_effects"] = self.build_baryonic_effects(
                baryonic_effects_
            )

        mg_parametrization_ = kwargs_dict.pop("mg_parametrization")
        if mg_parametrization_ is not None:
            kwargs_dict["mg_parametrization"] = self.build_mg_parametrization(
                mg_parametrization_
            )

        self._cosomlogy = ccl.CosmologyVanillaLCDM(**kwargs_dict)
        return self._cosomlogy


class CCLCosmologyParams(CosmologyParams):
    """General CCL Parameter set"""

    cosmology_type: Literal["ccl"] = Field(
        default="", description="Type of cosmology parameter set"
    )

    Omega_c: float = Field(
        ..., description="Cold dark matter density fraction", ge=0.0, le=1.0
    )
    Omega_b: float = Field(
        ..., description="Baryonic matter density fraction", ge=0.0, le=1.0
    )
    Omega_k: float = Field(
        default=0.0,
        description="Curvature density fraction density fraction",
        ge=-1.0,
        le=1.0,
    )
    Omega_g: float | None = Field(
        default=None,
        description="Density in relativistic species except massless neutrinos",
        ge=0.0,
        le=1.0,
    )
    h: float = Field(
        default=0.0,
        description="Hubble constant divided by 100 km/s/Mpc",
        ge=0.0,
        le=2.0,
    )
    n_s: float = Field(
        default=0.96,
        description="Primordial scalar perturbation spectral index",
        ge=0.0,
        le=2.0,
    )
    sigma8: float | None = Field(
        default=0.96,
        description="Variance of matter density perturbations at an 8 Mpc/h scale.",
        ge=0.0,
        le=2.0,
    )
    A_s: float | None = Field(
        default=0.96, description="Power spectrum normalization.", ge=0.0, le=2.0
    )
    Neff: float = Field(
        default=3.044,
        description="Effective number of massless neutrinos present.",
        ge=3.0,
        le=3.2,
    )
    m_nu: float = Field(
        default=0.0, description="Mass in eV of the massive neutrinos present."
    )
    w0: float = Field(
        default=-1.0,
        description="First order term of dark energy equation of state.",
        ge=-2.0,
        le=1.0,
    )
    wa: float = Field(
        default=0.0,
        description="Second order term of dark energy equation of state.",
        ge=-2.0,
        le=2.0,
    )
    T_CMB: float = Field(default=2.7255, description="The CMB temperature today.")
    T_ncdm: float = Field(
        default=0.71611,
        description="Non-CDM temperature in units of photon temperature.",
    )

    mass_split: str = Field(default="normal", description="Type of massive neutrinos.")
    transfer_function: str = Field(
        default="boltzmann_camb", description="The transfer function to use."
    )
    matter_power_spectrum: str = Field(
        default="halofit", description="The matter power spectrum to use."
    )

    baryonic_effects: Baryons | None = Field(
        default=None, description="The baryonic effects model to use."
    )
    mg_parametrization: ModifiedGravity | None = Field(
        default=None, description="The modified gravity parametrization to use."
    )

    extra_parameters: dict = {}

    def build_cosmology(self) -> ccl.Cosmology:

        kwargs_dict = self.dict()

        baryonic_effects_ = kwargs_dict.pop("baryonic_effects")
        if baryonic_effects_ is not None:
            kwargs_dict["baryonic_effects"] = self.build_baryonic_effects(
                baryonic_effects_
            )

        mg_parametrization_ = kwargs_dict.pop("mg_parametrization")
        if mg_parametrization_ is not None:
            kwargs_dict["mg_parametrization"] = self.build_mg_parametrization(
                mg_parametrization_
            )

        self._cosomlogy = ccl.Cosmology(**kwargs_dict)
        return self._cosomlogy


CosmologyParamsUnion = Union[
    CCLCosmologyVanillaLCDMParams,
    CCLCosmologyParams,
]
