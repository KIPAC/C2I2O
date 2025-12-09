from __future__ import annotations

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as



def read_yaml_file_as(theType: Type[BaseModel], filepath: str) -> theType:

    with open(filepath, 'r') as fin:
        yaml_data = fin.read()

    return parse_yaml_raw_as(theType, yaml_data)
