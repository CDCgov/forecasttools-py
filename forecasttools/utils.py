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


def generate_time_range_for_dim(
    start_date_as_dt: datetime,
    variable_data: xr.DataArray,
    dimension: str,
    time_step: timedelta,
):
    """
    Generates a range of times based on the
    start date, time step, and variable's
    dimension size. A range of dates is
    generated if the start date is a date and
    the time step is in days. A range of times
    is generated if the start date is a time
    and/or the time step is a time.
    """
    # number of seconds in a day
    SECONDS_IN_DAY = timedelta(days=1).total_seconds()

    # total number of seconds in the time_step
    total_seconds = time_step.total_seconds()

    # get the size of the dimension
    interval_size = variable_data.sizes[dimension]

    # determine the interval string for Polars
    if (
        total_seconds % SECONDS_IN_DAY == 0
    ):  # check if time_step is in full days
        interval_str = f"{int(total_seconds // SECONDS_IN_DAY)}d"
    else:
        interval_str = (
            f"{total_seconds}s"  # handle finer-grained time intervals
        )

    # check if the start_date_as_dt is a datetime or date
    if isinstance(start_date_as_dt, datetime):
        # use datetime_range for times
        return (
            pl.datetime_range(
                start=start_date_as_dt,
                end=start_date_as_dt + (interval_size - 1) * time_step,
                interval=interval_str,  # use the calculated interval
                closed="both",
                eager=True,  # return a Polars Series
            )
            .to_numpy()
            .astype("datetime64[ns]")  # Ensure consistent datetime format
        )
    elif isinstance(start_date_as_dt, datetime.date):
        # use date_range for dates
        return (
            pl.date_range(
                start=start_date_as_dt,
                end=start_date_as_dt + (interval_size - 1) * time_step,
                interval=interval_str,  # use the calculated interval
                closed="both",
                eager=True,  # return a Polars Series
            )
            .to_numpy()
            .astype("datetime64[D]")  # date format
        )
    else:
        raise ValueError(
            f"Unsupported start_date type: {type(start_date_as_dt)}. Must be datetime or date."
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
