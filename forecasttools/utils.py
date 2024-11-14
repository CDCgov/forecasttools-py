"""
A collection of utility functions used
across other forecasttools code.
"""

from collections.abc import MutableSequence


def check_input_variable_type(value: any, expected_type: any, param_name: str):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{param_name}' must be of type '{expected_type.__name__}'; got {type(value)}"
        )


def ensure_listlike(x):
    """
    Ensure that an object either behaves like a
    :class:`MutableSequence` and if not return a
    one-item :class:`list` containing the object.
    Useful for handling list-of-strings inputs
    alongside single strings.
    Based on this _`StackOverflow approach
    <https://stackoverflow.com/a/66485952>`.
    Parameters
    ----------
    x
        The item to ensure is :class:`list`-like.
    Returns
    -------
    MutableSequence
        ``x`` if ``x`` is a :class:`MutableSequence`
        otherwise ``[x]`` (i.e. a one-item list containing
        ``x``.
    """
    return x if isinstance(x, MutableSequence) else [x]
