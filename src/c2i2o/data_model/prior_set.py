import numpy as np
from pydantic import BaseModel, Field

from c2i2o.functions.scipy_wrap import ScipyWrapped


def convert_dict_to_2d_array(
    input_dict: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Convert a dict of numpy arrays to a 2d array

    Parameters
    ----------
    input_dict:
        Dict of numpy arrays

    Returns
    -------
    List of the names of the columns, and a 2d array with the data
    """
    var_list: list[str] = list(input_dict.keys())
    out_array = np.vstack(list(input_dict.values())).T
    return (var_list, out_array)


def convert_table_to_list_of_dicts(
    input_dict: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    """Convert a dict of numpy arrays to a list of dicts of floats

    Parameters
    ----------
    input_dict:
        Dict of numpy arrays

    Returns
    -------
    List of dictionaries, one per item in the numpy arrays
    """
    return [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]


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
