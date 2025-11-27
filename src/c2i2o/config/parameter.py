"""A small module with functionality to handle define configuration parameters"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar

import Enum
import numpy as np

T = TypeVar("T")


def cast_value(dtype: type[T] | None, value: Any) -> T | None:
    """Casts an input value to a particular type

    Parameters
    ----------
    dtype:
        The type we are casting to

    value:
        The value being cast

    Returns
    -------
    The value cast to type T

    Raises
    ------
    TypeError if neither value nor dtype are None and the casting fails

    Notes
    -----
    This will proceed in the following order
        1.  If dtype is None it will simply return value
        2.  If value is None it will return None
        3.  If value is an instance of dtype it will return value
        4.  If value is a Mapping it will use it as a keyword dictionary
            to the constructor of dtype, i.e., return dtype(**value)
        5.  If value is an Iterable it will use it as args to the constructor
            of dtype, i.e., return dtype(*args)
        6.  It will try to pass value to the constructor of dtype, i.e.,
            return dtype(value)
        7.  If all of these fail, it will raise a TypeError
    """
    # dtype is None means all values are legal
    if dtype is None:
        return value
    # value is None is always allowed
    if value is None:
        return None
    # if value is an instance of self.dtype, then return it
    if isinstance(value, dtype):
        return value
    # if value is a Mapping it will use it as  keywords
    if isinstance(value, Mapping):
        return dtype(**value)
    # if value is an Iterable it will use it as args
    if isinstance(value, Iterable):
        return dtype(*value)
    # try the constructor of dtype
    try:
        return dtype(value)
    except (TypeError, ValueError):
        pass

    msg = f"Value of type {type(value)}, when {str(dtype)} was expected and casting failed"
    raise TypeError(msg)


def cast_to_streamable(value: Any) -> Any:
    """Cast a value to something that yaml can easily stream"""
    if isinstance(value, Parameter):
        return value.value
    return value


class Parameter[T]:
    """A small class to manage a single parameter with basic type checking

    Attributes
    ----------
    _dtype: type[T] | None
        The data type for this parameter

    _msg: str
        A help or docstring

    _default: T | None
        The default value

    _fmt: str
        A formatting string for printout and representation

    _required: bool
        Is the Parameter required

    _value: T | None
        Current value of the parameter
    """

    def __init__(
        self,
        dtype: type[T] | None,
        msg: str,
        default: T | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        self._dtype = dtype
        self._msg = msg
        self._default = default
        self._fmt = fmt
        self._required = required
        self._value = cast_value(self._dtype, self._default)

    def __repr__(self) -> str:
        req_str = "required" if self.required else "optional"
        return f"Parameter({self.msg}, type: {self.dtype}, default: {self.default} [{req_str}])"

    @property
    def value(self) -> T | None:
        """Return the value"""
        return self._value

    @property
    def dtype(self) -> type[T] | None:
        """Return the data type"""
        return self._dtype

    @property
    def default(self) -> T | None:
        """Return the default value"""
        return self._default

    @property
    def required(self) -> bool:
        """Return the required flag"""
        return self._required

    @property
    def msg(self) -> str:
        """Return the help or docstring"""
        return self._msg

    def copy(self) -> Parameter[T]:
        """Return a copy of self"""
        copy_par = Parameter(
            dtype=self._dtype,
            msg=self._msg,
            default=self._default,
            fmt=self._fmt,
            required=self._required,
        )
        copy_par.set(self._value)
        return copy_par

    def set(self, value: T | None) -> T | None:
        """Set the value, raising a TypeError if the value is the wrong type"""
        self._value = cast_value(self._dtype, value)
        return self._value

    def validate(self, value: Any) -> None:
        """Test if a value is legal

        Raises
        ------
        ValueError if the value is not legal
        """
        if value is None:
            return
        if self._dtype is None:
            return
        assert isinstance(value, self._dtype)

    def set_to_default(self) -> T | None:
        """Set the value to the default"""
        self._value = cast_value(self._dtype, self._default)
        return self._value

    def set_default(self, default: T | None) -> T | None:
        """Set the default value"""
        self._default = default
        return self.set_to_default()

    def numpy_style_help_text(self) -> str:
        """Create a docstring followwing numpy style guidelines"""
        s = ""
        if self._dtype is None:
            s += "[type not specified] "
        else:
            s += f"[{self._dtype.__name__}] "
        if self._required:
            s += "(required)"
        else:
            # truncate long dicts
            if isinstance(self._default, dict) and len(self._default) > 10:
                s += "(default={...})"
            # truncate long lists
            elif isinstance(self._default, list) and len(self._default) > 10:
                s += "(default=[...])"
            else:
                s += f"default={self._default}"
        s += f"\n    {self._msg}"
        return s


#
# Simple parameter types: str, int, float, choice
#


class StrParameter(Parameter[str]):
    """Specialization for string parameters"""

    def __init__(
        self,
        msg: str,
        default: str | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter[str].__init__(self, str, msg, default, fmt, required=required)


class FloatParameter(Parameter[float]):
    """Specialization for float parameters

    This includes bounds checking.

    Trying to set either the value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: float | None = None,
        fmt: str = "%0.4f",
        min_value: float = -np.inf,
        max_value: float = np.inf,
        *,
        required: bool = False,
    ):
        Parameter[float].__init__(self, float, msg, default, fmt, required=required)
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: Any) -> None:
        """Test if a value is legal

        Raises
        ------
        ValueError if the value is not legal
        """
        Parameter[float].validate(self, value)
        if value < self._min_value or value > self._min_value:
            raise ValueError(
                f"{self._msg}: value: {value} not in range {self._min_value}:{self._min_value}"
            )


class IntParameter(Parameter[int]):
    """Specialization for int parameters

    This includes bounds checking.

    Trying to set either the value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: int | None = None,
        fmt: str = "%i",
        *,
        required: bool = False,
    ):
        Parameter[int].__init__(self, int, msg, default, fmt, required=required)


class ChoiceParameter(Parameter[Enum]):
    """Specialization for choice parameters

    This includes checking.

    Trying to set either the value or the default to a non-mapped value
    will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: str | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter.__init__(self, int, msg, default, fmt, required=required)


#
# list parameter types: str, int, float, choice
#


class StrListParameter(Parameter[list[str]]):
    """Specialization for list of string parameters"""

    def __init__(
        self,
        msg: str,
        default: list[str] | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter[list[str]].__init__(
            self, list[str], msg, default, fmt, required=required
        )


class FloatListParameter(Parameter[list[float]]):
    """Specialization for list of int parameters

    This includes bounds checking.

    Trying to set either any value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: list[float] | None = None,
        fmt: str = "%s",
        min_value: float = -np.inf,
        max_value: float = np.inf,
        *,
        required: bool = False,
    ):
        Parameter[list[float]].__init__(
            self, list[float], msg, default, fmt, required=required
        )
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: Any) -> None:
        """Test if a value is legal

        Raises
        ------
        ValueError if the value is not legal
        """
        Parameter[list[float]].validate(self, value)
        for i, val in value:
            if val < self._min_value or val > self._min_value:
                raise ValueError(
                    f"{self._msg}: value[{i}]: {val} not in range {self._min_value}:{self._min_value}"
                )


class IntListParameter(Parameter[list[int]]):
    """Specialization for list of int parameters

    This includes bounds checking.

    Trying to set either any value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: list[int] | None = None,
        fmt: str = "%i",
        *,
        required: bool = False,
    ):
        Parameter[list[int]].__init__(
            self, list[int], msg, default, fmt, required=required
        )


class ChoiceListParameter(Parameter[list[int]]):
    """Specialization for list of choice parameters

    This includes checking.

    Trying to set either any value or the default to a non-mapped value
    will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: list[int] | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter[list[int]].__init__(
            self, list[int], msg, default, fmt, required=required
        )


#
# dict parameter types: str, int, float, choice
#


class StrToStrDictParameter(Parameter[dict[str, str]]):
    """Specialization for dict mapping str to str parameters"""

    def __init__(
        self,
        msg: str,
        default: dict[str, str] | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter[dict[str, str]].__init__(
            self, dict[str, str], msg, default, fmt, required=required
        )


class StrToFloatDictParameter(Parameter[dict[str, float]]):
    """Specialization for dict mapping str to float parameters

    This includes bounds checking.

    Trying to set either any value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: dict[str, float] | None = None,
        fmt: str = "%s",
        min_value: float = -np.inf,
        max_value: float = np.inf,
        *,
        required: bool = False,
    ):
        Parameter[dict[str, float]].__init__(
            self, dict[str, float], msg, default, fmt, required=required
        )
        self._min_value = min_value
        self._max_value = max_value

    def validate(self, value: Any) -> None:
        """Test if a value is legal

        Raises
        ------
        ValueError if the value is not legal
        """
        Parameter[dict[str, float]].validate(self, value)
        for key, val in value.items():
            if val < self._min_value or val > self._min_value:
                raise ValueError(
                    f"{self._msg}: value[{key}]: {val} not in range {self._min_value}:{self._min_value}"
                )


class StrToIntDictParameter(Parameter[dict[str, int]]):
    """Specialization for dict mapping str to int parameters

    This includes bounds checking.

    Trying to set either any value or the default outside
    the bound will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: dict[str, int] | None = None,
        fmt: str = "%i",
        *,
        required: bool = False,
    ):
        Parameter[dict[str, int]].__init__(
            self, dict[str, int], msg, default, fmt, required=required
        )


class StrToChoiceDictParameter(Parameter[dict[str, int]]):
    """Specialization for dict mapping str to choice parameters

    This includes checking.

    Trying to set either any value or the default to a non-mapped value
    will raise a ValueError
    """

    def __init__(
        self,
        msg: str,
        default: dict[str, int] | None = None,
        fmt: str = "%s",
        *,
        required: bool = False,
    ):
        Parameter[dict[str, int]].__init__(
            self, dict[str, int], msg, default, fmt, required=required
        )
