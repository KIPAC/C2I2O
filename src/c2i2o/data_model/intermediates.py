from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field

from c2i2o.data_model.parameter_grid import LinearAGridParams, LogKGridParams



class IntermediateProductCalculationParams(BaseModel):

    cosmology_type: str = Field(..., description="Cosmology type")
    function_name: str = Field(..., description="Function to call")

    eval_kwargs: dict = Field({}, description="Function evaluate kwarg parameters")
    
    def get_function_grid_args(self) -> list[np.ndarry]:
        return [grid_parms.build_grid() for grid_parms in self.eval_grid]

    def get_function_kwargs(self) -> dict[str, Any]:
        return eval_kwargs
    
    def evalute_function(self):
        the_function = getattr(self.cosmology_class, function_name)
        grid_args = self.get_function_grid_args()
        func_kwargs = self.get_function_kwargs()
        return the_function(*grid_args, **func_kwargs)


class CCLLinearMatterPowerSpectrumCalculationParams(IntermediateProductCalculationParams):

    calculation_type: Literal["pyccl.linear_matter_power"] = Field(..., description="Calculation type")

    cosmology_type: Literal["pyccl"] = Field(default="pyccl", description="Cosmology type")
    function_name: Literal["linear_matter_power"] = Field(default="linear_matter_power", description="Function to call")

    a_grid: LinearAGridParams = Field("a_grid", description="Scale factor grid parameters")
    k_grid: LogKGridParams = Field("k_grid", description="Logarithmic wavenumber grid parameters")
    
    def get_function_grid_args(self) -> list[np.ndarry]:
        return [a_grid.build_grid(), k_grid.build_grid()]

        
class CCLComovingRadialDistanceCalculationParams(IntermediateProductCalculationParams):

    calculation_type: Literal["pyccl.comoving_radial_distance"] = Field(..., description="Calculation type")

    cosmology_type: Literal["pyccl"] = Field(default="pyccl", description="Cosmology type")
    function_name: Literal["comoving_radial_distance"] = Field(default="comoving_radial_distance", description="Function to call")

    a_grid: LinearAGridParams = Field("a_grid", description="Scale factor grid parameters")
    
    def get_function_grid_args(self) -> list[np.ndarry]:
        return [a_grid.build_grid()]

        
class CCLHOverH0CalculationParams(IntermediateProductCalculationParams):

    calculation_type: Literal["pyccl.h_over_h0"] = Field(..., description="Calculation type")

    cosmology_type: Literal["pyccl"] = Field(default="pyccl", description="Cosmology type")
    function_name: Literal["h_over_h0"] = Field(default="h_over_h0", description="Function to call")

    a_grid: LinearAGridParams = Field("a_grid", description="Scale factor grid parameters")
    
    def get_function_grid_args(self) -> list[np.ndarry]:
        return [a_grid.build_grid()]


IntermediateCalculationParamsUnion = Union[
    CCLLinearMatterPowerSpectrumCalculationParams,
    CCLComovingRadialDistanceCalculationParams,
    CCLHOverH0CalculationParams,
]
        
