from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as


# Define a TypeVar. This represents a generic type.
T = TypeVar('T', bound=BaseModel)


def read_yaml_file_as(the_type: Type[T], filepath: str) -> T:
    """Read a file into an Pydantic object

    Parameters
    ----------
    theType:
        Pydantic type we are reading

    filepath:
        File we are reading

    Returns
    -------
    Object of the requested type
    """
    with open(filepath, "r", encoding="utf-8") as fin:
        yaml_data = fin.read()

    return parse_yaml_raw_as(the_type, yaml_data)
