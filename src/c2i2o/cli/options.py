import enum
from functools import partial
from typing import Any, Type, TypeVar

import click

EnumType_co = TypeVar("EnumType_co", bound=Type[enum.Enum], covariant=True)


class EnumChoice(click.Choice):
    """A version of click.Choice specialized for enum types"""

    def __init__(self, the_enum: EnumType_co, case_sensitive: bool = True) -> None:
        self._enum = the_enum
        super().__init__(
            list(the_enum.__members__.keys()), case_sensitive=case_sensitive
        )

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> enum.Enum:  # pragma: no cover
        converted_str = super().convert(value, param, ctx)
        return self._enum.__members__[converted_str]


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: str, **kwargs: Any) -> None:
        self._partial = partial(click.option, *param_decls, cls=click.Option, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=click.Argument, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)


config_file = PartialArgument(
    "config_file",
    type=str,
    nargs=1,
)


cosmo_parameter_file = PartialOption(
    "--cosmo-parameter-file",
    type=click.Path(),
    default=None,
    help="Path for cosmological parameters file",
)


intermediates_file = PartialOption(
    "--intermediates-file",
    type=click.Path(),
    default=None,
    help="Path for intermediates file",
)
