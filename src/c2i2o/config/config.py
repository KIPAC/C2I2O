"""A small module with functionality to handle configuration"""

from typing import Any, Iterator

from .parameter import Parameter, cast_to_streamable


class Config(dict[str, Any]):
    """A small class to manage a dictionary of configuration parameters with basic type checking"""

    def __init__(self, **kwargs: Any):
        """Build from keywords

        Note
        ----
        The keywords are used as keys for the configuration parameters

        The values are used to define the allowed data type and default values

        For each key-value pair:
        If the value is a type then it will define the data type and the default will be `None`
        If the value is a value then it will set the default value define the data type as type(value)
        """
        dict.__init__(self)
        for key, val in kwargs.items():
            if not isinstance(val, Parameter):
                raise TypeError("Config can only take Parameter objects as values")
            self[key] = val.copy()

    def __str__(self) -> str:
        """Override __str__ casting to deal with `Parameter` object in the map"""
        s = "{"
        for key, attr in dict.items(self):
            assert isinstance(attr, Parameter)
            val = attr.value
            s += f"{key}:{val},"
        s += "}"
        return s

    def __repr__(self) -> str:
        """A custom representation"""
        s = "Config"
        s += self.__str__()
        return s

    def to_dict(self) -> dict[str, Any]:
        """Forcibly return a dict where the values have been cast from Parameter"""
        return {key: cast_to_streamable(value) for key, value in dict.items(self)}

    def __iter__(self) -> Iterator[str]:
        """Override the __iter__ to work with `Parameter`"""
        d = self.to_dict()
        return iter(d)

    def __getitem__(self, key: str) -> Any:
        """Override the __getitem__ to work with `Parameter`"""
        attr = dict.__getitem__(self, key)
        return attr.value

    def __setitem__(self, key: str, value: Any) -> Any:
        """Override the __setitem__ to work with `Parameter`"""
        attr = dict.__getitem__(self, key)
        return attr.set(value)

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-like parameter access"""
        return self.__getitem__(key)

    def __setattr__(self, key: str, value: Any) -> Any:
        """Allow attribute-like parameter setting"""
        return self.__setitem__(key, value)

    def to_streamable_dict(self) -> dict[str, Any]:
        """Override return a dict where all the values have been cast to streamable classes"""
        return {key: cast_to_streamable(value) for key, value in dict.items(self)}

    def reset(self) -> None:
        """Reset values to their defaults"""
        for _, val in dict.items(self):
            val.set_to_default()

    def get_type(self, key: str) -> type:
        """Get the type associated to a particular configuration parameter"""
        attr = dict.__getitem__(self, key)
        return attr.dtype

    def numpy_style_help_text(self) -> str:
        """Create a docstring followwing numpy style guidelines"""
        if not self:
            return "<This class has no configuration options>"
        s = []
        for key, val in dict.items(self):
            assert isinstance(val, Parameter)
            s.append(f"{key}: {val.numpy_style_help_text()}")
        return "\n\n".join(s)
