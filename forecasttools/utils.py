"""
A collection of utility functions used
across other forecasttools code.
"""

from collections.abc import Iterable, MutableSequence

import arviz as az
import xarray as xr


def validate_input_type(value: any, expected_type: type | tuple[type], param_name: str):
    """Checks the type of a variable and
    raises a TypeError if it does not match
    the expected type."""
    if isinstance(expected_type, tuple):
        if not any(isinstance(value, t) for t in expected_type):
            raise TypeError(
                f"Parameter '{param_name}' must be one of the types {expected_type}; got {type(value)}"
            )
    else:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter '{param_name}' must be of type '{expected_type.__name__}'; got {type(value)}"
            )


def validate_and_get_idata_group(idata: az.InferenceData, group: str):
    """Retrieves the group from the
    InferenceData object and validates its
    existence."""
    if not hasattr(idata, group):
        raise ValueError(f"Group '{group}' not found in idata object.")
    return getattr(idata, group)


def validate_and_get_idata_group_var(
    idata_group: xr.Dataset,
    group: str,
    variable: str,
):
    """Retrieves the variable from the group
    and validates its existence."""
    if variable not in idata_group.data_vars:
        raise ValueError(f"Variable '{variable}' not found in group '{group}'.")
    return idata_group[variable]


def validate_idata_group_var_dim(variable_data: xr.DataArray, dimension: str):
    """Validates if the given dimension
    exists in the variable."""
    if dimension not in variable_data.dims:
        raise ValueError(
            f"Dimension '{dimension}' not found in variable dimensions: '{variable_data.dims}'."
        )


def validate_iter_has_expected_types(
    iterable: Iterable, expected_type: type, param_name: str
):
    """
    Check if all elements in the iterable are
    instances of expected_type. If not, raise
     a TypeError.
    """
    if not all(isinstance(item, expected_type) for item in iterable):
        raise TypeError(
            f"All items in '{param_name}' must be of type '{expected_type.__name__}'."
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
