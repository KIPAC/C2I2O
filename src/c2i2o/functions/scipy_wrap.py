import inspect
import re
from typing import Dict, Type, Any, Optional

# Import SciPy and Pydantic components
import scipy.stats as sps
from pydantic import BaseModel, Field, create_model



class ScipyWrapped(BaseModel):

    def get_dist(self):

        dd = self.dict().copy()
        scipy_class = dd.pop('scipy_type')        
        return scipy_class(**dd)



def get_scipy_distributions() -> Dict[str, Any]:
    """
    Scans the scipy.stats module for continuous distribution instances.
    """
    distributions = {}    
    
    for name, obj in inspect.getmembers(sps):
        if isinstance(obj, sps.rv_continuous):
            distributions[name] = obj
            
    return distributions


def create_pydantic_models_for_scipy_stats() -> Dict[str, Type[BaseModel]]:
    """
    Dynamically creates Pydantic models for the parameters of SciPy distributions.
    
    For example, 'norm' (Normal distribution) will get a model with 'loc' and 'scale'.
    'gamma' distribution will get 'a', 'loc', and 'scale'.
    """
    distributions = get_scipy_distributions()
    dynamic_models: Dict[str, Type[BaseModel]] = {}
    
    # Base parameters common to almost all distributions
    base_params = {
        'loc': (float, Field(default=0.0, description="Location parameter.")),
        'scale': (float, Field(default=1.0, description="Scale parameter."))
    }

    print(f"Found {len(distributions)} distributions in scipy.stats to model...")

    for dist_name, dist_obj in distributions.items():
        # Start with base parameters
        fields = base_params.copy()

        # Add the type of the underlying distribution
        fields['scipy_type'] = (type, Field(default=dist_obj, description="Scipy type."))
        
        # Extract shape parameters from the 'shapes' attribute (comma-separated string)
        shape_params_str = getattr(dist_obj, 'shapes', None)

        # _updated_ctor_param has the default values
        ctor_param = dist_obj._updated_ctor_param()
        
        if shape_params_str:
            # Clean and split the string into individual shape names
            shape_names = [s.strip() for s in shape_params_str.split(',') if s.strip()]
            
            for shape_name in shape_names:

                # look for the default in the 
                param_default = ctor_param.get(shape_name)
                # use float as the safest type if we can't get the type from the default
                if param_default is None:
                    param_type = float
                else:
                    param_type = type(param_default)                

                # Use float as the safest general type for SciPy shape parameters.
                fields[shape_name] = (
                    float, 
                    Field(ctor_param.get(shape_name), description=f"Shape parameter '{shape_name}'.")
                )

        # Create the dynamic Pydantic model
        ModelName = f"{dist_name.capitalize()}"
        
        # Use a docstring to clearly identify the model
        docstring = (
            f"Pydantic model for validating input parameters of the "
            f"scipy.stats.{dist_name} distribution."
        )
        
        try:
            model = create_model(
                ModelName,
                __module__=__name__,
                __doc__=docstring,
                __base__=ScipyWrapped,
                **fields
            )
            dynamic_models[dist_name] = model
        except Exception as e:
            # Catch errors during model creation (e.g., invalid field names)
            print(f"Skipping {dist_name} due to model creation error: {e}")

    return dynamic_models
