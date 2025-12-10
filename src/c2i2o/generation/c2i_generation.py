from __future__ import annotations

import sys
from typing import Type

import numpy as np
from pydantic import BaseModel, Field

from c2i2o.data_model.base_classes import COSMOLOGY_CLASS_DICT, Cosmology
from c2i2o.data_model.cosmology_ccl import CosmologyParamsUnion
from c2i2o.data_model.enums import CosmologyCalculatorType
from c2i2o.data_model.intermediates import IntermediateCalculationParamsUnion
from c2i2o.data_model.prior_set import PriorSet, convert_table_to_list_of_dicts


class C2IGenerationParams(BaseModel):
    """Class to wrap generation of cosmological paramters
    and computation of intermediate data products
    """

    base_cosmology: CosmologyParamsUnion = Field(
        ..., description="Reference Cosmological Parameters"
    )
    calculator_type: CosmologyCalculatorType = Field(
        default=CosmologyCalculatorType.CCL, description="Cosmology caclulator type"
    )

    prior_set: PriorSet = Field(..., description="Priors to sample for generation")
    n_samples: int = Field(..., description="Number of samples to generator")

    computation_parameters: dict[str, IntermediateCalculationParamsUnion] = Field(
        ..., description="Parameters for computations of intermediate data products"
    )

    def generate_cosmology_parameters(self) -> dict[str, np.ndarray]:
        """Generate a set of cosmological parameters

        Notes
        -----
        This will create `self.n_samples` samples drawn from the priors and
        fixed parameters defined in `self.prior_set`
        """
        samples = self.prior_set.generate_data(self.n_samples)
        return samples

    def build_cosmology_class(self) -> type[Cosmology]:
        """Return the Cosmology calculator class specified

        Notes
        -----
        This will return a cosmology class as selected by
        `self.calculator_type` from the static dict
        `c2i2o.data_model.base_classes.COSMOLOGY_CLASS_DICT`
        """
        return COSMOLOGY_CLASS_DICT[self.calculator_type.value]

    def build_cosmology(
        self, cosmology_class: Type[Cosmology], override_parameters: dict[str, float]
    ) -> Cosmology:
        """Build a Cosmology calculator object

        Parameters
        ----------
        cosmology_class:
            Cosmology calculator class to instantiate.
            This is provided to avoid having to look it up repeatedly
            when computing intermediates

        override_parameters:
            Override values with respect to the `self.base_cosmology`

        Returns
        -------
        Newly instantiated Cosmology calculator object

        Notes
        -----
        This will return a cosmology class as selected by
        `self.calculator_type` from the static dict
        `c2i2o.data_model.base_classes.COSMOLOGY_CLASS_DICT`
        and updated the parameters in `self.base_cosmology`
        with `override_parameters`
        """

        cosmo_params = self.base_cosmology.dict().copy()
        cosmo_params.update(**override_parameters)
        cosmo_params.pop("cosmology_type")
        cosmology = cosmology_class(**cosmo_params)
        return cosmology

    def compute_intermediates(
        self, parameters: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Compute a set of intermediate data products from input parameters

        Parameters
        ----------
        parameters:
            Cosmological parameters to use in calculations.

        Returns
        -------
        Computed intermediate data products

        Notes
        -----
        For each set of `paramters` this will compute
        all of the intermediates defined in `self.computation_parameters`,
        on the specified evaluation grids
        """

        cosmology_inputs = convert_table_to_list_of_dicts(parameters)
        n_cosmologies = len(cosmology_inputs)
        cosmology_class = self.build_cosmology_class()

        output_dict: dict[str, np.ndarray] = {}
        for key, computation in self.computation_parameters.items():
            output_dict[key] = computation.allocate_arrays(n_cosmologies)

        for i, cosmo_params_override in enumerate(cosmology_inputs):
            if i == 0:
                pass
            elif i % 100 == 0:
                sys.stdout.write("x")
                sys.stdout.flush()
            elif i % 20 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            cosmology = self.build_cosmology(cosmology_class, cosmo_params_override)
            for key, computation in self.computation_parameters.items():
                output_dict[key][i] = computation.evalute_function(cosmology).T
        sys.stdout.write("!\n")
        sys.stdout.flush()
        return output_dict

    def generate_intermediates(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Generate a set of cosmological parameters and compute intermediates

        Returns
        -------
        Generated cosmological parameters
        Computed intermediate data products

        Notes
        -----
        This will create `self.n_samples` samples drawn from the priors and
        fixed parameters defined in `self.prior_set`.  It will then use those
        to compute intermediate the data products defined in
        `self.computation_parameters`.
        """
        parameter_samples = self.generate_cosmology_parameters()
        intermediates = self.compute_intermediates(parameter_samples)

        return (parameter_samples, intermediates)
