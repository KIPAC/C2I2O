import numpy as np
from pydantic import BaseModel, Field

from c2i2o.functions.scipy_wrap import ScipyWrapped


def generate_samples_from_priors(
    priors: dict[str, ScipyWrapped],
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Generate a set of parameters from a dict of priors

    Parameters
    ----------
    priors:
        Dict of priors we use to generate the data

    n_samples:
        Number of samples to generate

    Returns
    -------
    A dict, key by parameter name, for numpy arrays of values
    """

    return {key: val.build_dist().rvs(n_samples) for key, val in priors.items()}


class PriorSet(BaseModel):
    """Class that wraps a set of priors on parameters

    yaml.code

      fixed:
        w0 = -1.
        wa = 0.
      priors:
        Omega_b:
          scipy_type: 'norm'
          loc: 0.05
          scale: 0.01
    """

    fixed: dict[str, float] = Field({}, description="Fixed Parameter and values")
    priors: dict[str, ScipyWrapped] = Field(
        {}, description="Parameters and associted priors"
    )

    def generate_data(
        self,
        n_samples: int,
    ) -> dict[str, np.ndarray]:
        """Generate a set of parameters from a dict of priors

        Parameters
        ----------
        n_samples:
           Number of samples to generate

        Returns
        -------
        A dict, key by parameter name, for numpy arrays of values
        """
        ret_values: dict[str, np.ndarray] = {
            key: np.full(n_samples, val) for key, val in self.fixed.items()
        }
        ret_values.update(**generate_samples_from_priors(self.priors, n_samples))
        return ret_values
