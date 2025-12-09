
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field

from c2i2o.data_model.base_classes import Cosmology
from c2i2o.data_model.prior_set import PriorSet
from c2i2o.data_model.intermediates import IntermediateCalculationParamsUnion



class C2IGenerationParams(BaseModel):

    base_comosology_paramters: dict = Field(..., description="Reference Cosmological Parameters")    
    calculator_type: str = Field(default="pyccl", description="Cosmology caclulator type")

    prior_set: PriorSet = Field(..., description="Priors to sample for generation")
    n_samples: int = Field(..., description="Number of samples to generator")

    computation_parameters: dict[str, IntermediateCalculationParamsUnion] = Field(..., description="Parameters for computations of intermediate data products")
    
    def generate_cosmology_parameters(self) -> dict[str, np.ndarray]:
        
        samples = self.prior_set.generate_data(self.n_samples)
        return samples

        
    def build_cosmology(self, cosmology_class: Type[Cosmology], override_parameters: dict[str, float]) -> Cosmology:
        
        cosmo_params = self.base_comosology_paramters.dict().copy()        
        cosmo_params.udpate(**override_parameters)
        cosmology = cosmo_class(**comos_params)
        return cosmology
        
        
    def compute_intermediates(self, parameters: dict[str, np.ndarray]) -> dict[str, IntermediateProductVector]:

        base_parameters = self.base_comosology_paramters.dict()
        cosmology_inputs = convert_table_to_list_of_dicts(t_dict)
        n_cosmologies = len(cosmology_inputs)
        cosmology_class = self.build_cosmology_class()

        output_dict: dict[str, IntermediateProductVector] = {}
        for key, computation in self.computation_parameters.items():
            output_dict[key] = computation.allocate_arrays(n_cosmologies)        

        for i, cosmo_params_override in enumerate(cosmology_inputs):
            cosmology = self.build_cosmology(cosmology_class, cosmo_params_override)                           
            for key, computation in self.computation_parameters.items():
                output_dict[key][i] = computation.evalute_function(cosmology)
                
        return output_dict
        

    def generate_intermediates(self) -> tuple[dict[str, np.ndarray], dict[str, IntermediateProductVector]]:

        parameter_samples = self.generate_cosmology_parameters()
        intermediates = self.compute_intermediates(parameter_samples)
        
        return (parameter_samples, intermediates)
