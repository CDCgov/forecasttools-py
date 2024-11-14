"""
A collection of utility functions used
across other forecasttools code.
"""

from collections.abc import Iterable, MutableSequence
from datetime import datetime, timedelta

import arviz as az
import polars as pl
import xarray as xr


def validate_input_type(
    value: any, expected_type: type | tuple[type], param_name: str
):
    """Checks the type of a variable and
    raises a TypeError if it does not match
    the expected type."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{param_name}' must be of type '{expected_type.__name__}'; got {type(value)}"
        )


def validate_and_get_start_date(start_date_iso: any):
    """Handles the start_date_iso input,
    converting it to a datetime object."""
    if isinstance(start_date_iso, str):
        try:
            return datetime.strptime(start_date_iso, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Parameter 'start_date_iso' must be in the format 'YYYY-MM-DD' if provided as a string; got {start_date_iso}."
            )
    elif isinstance(start_date_iso, datetime):
        return start_date_iso
    else:
        raise TypeError(
            f"Parameter 'start_date_iso' must be of type 'str' or 'datetime'; got {type(start_date_iso)}."
        )


def validate_and_get_idata_group(idata: az.InferenceData, group: str):
    """Retrieves the group from the
    InferenceData object and validates its
    existence."""
    idata_group = getattr(idata, group, None)
    if idata_group is None:
        raise ValueError(f"Group '{group}' not found in idata object.")
    return idata_group


def validate_and_get_idata_group_var(
    idata_group: xr.Dataset,
    group: str,
    variable: str,
):
    """Retrieves the variable from the group
    and validates its existence."""
    if variable not in idata_group.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in group '{group}'."
        )
    return idata_group[variable]


def validate_idata_group_var_dim(variable_data: xr.DataArray, dimension: str):
    """Validates if the given dimension
    exists in the variable."""
    if dimension not in variable_data.dims:
        raise ValueError(
            f"Dimension '{dimension}' not found in variable dimensions: '{variable_data.dims}'."
        )


def validate_group_var_dim_instances(
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


def generate_date_range_for_dim(
    start_date_as_dt: datetime,
    variable_data: xr.DataArray,
    dimension: str,
    time_step: timedelta,
):
    """Generates a range of dates based on
    the start date, time step, and variable's
    dimension size."""
    interval_size = variable_data.sizes[dimension]
    return (
        pl.date_range(
            start=start_date_as_dt,
            end=start_date_as_dt + (interval_size - 1) * time_step,
            interval=f"{time_step.days}d" if time_step.days else "1d",
            closed="both",
            eager=True,
        )
        .to_numpy()
        .astype("datetime64[ns]")
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
