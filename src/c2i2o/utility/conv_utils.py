"""Utility functions to convert between various data structures"""

import numpy as np


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
