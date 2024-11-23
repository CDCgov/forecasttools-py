"""
Contains functions for converting
Arviz InferenceData objects to Polars
dataframes with dates and draws and
for working with conversions of
Polars dataframes to hubverse ready and
scoringutils ready dataframes
"""

from datetime import date, datetime, timedelta

import arviz as az
import numpy as np
import polars as pl
import xarray as xr

import forecasttools


def is_timedelta_in_days_only(td: timedelta) -> bool:
    """
    Checks if a timedelta object can be
    represented exclusively in days,
    meaning it has no hours, minutes,
    seconds, or microseconds.
    """
    return td.seconds == 0 and td.microseconds == 0


def convert_date_or_datetime_to_np(time_object: any) -> np.datetime64:
    """
    Converts a date or datetime object to
    numpy.datetime64: date -> datetime64[D]
    (day precision), datetime ->
    datetime64[ns] (nanosecond precision).
    """
    if isinstance(time_object, date) and not isinstance(time_object, datetime):
        return np.datetime64(time_object, "D")
    elif isinstance(time_object, datetime):
        return np.datetime64(time_object, "ns")
    else:
        raise TypeError(
            f"Input must be a date or datetime object; got {type(time_object)}"
        )


def convert_timedelta_to_np(td: timedelta) -> np.timedelta64:
    """
    Converts a Python timedelta to:
    numpy.timedelta64[D] if it is
    representable in days only. Otherwise,
    convert to numpy.timedelta64[ns].
    """
    if is_timedelta_in_days_only(td):
        return np.timedelta64(td.days, "D")
    else:
        return np.timedelta64(td).astype("timedelta64[ns]")


def generate_time_range_for_dim(
    start_time_as_dt: datetime | date | np.datetime64,
    variable_data: xr.DataArray,
    dimension: str,
    time_step: timedelta | np.timedelta64,
) -> np.ndarray
    """
    Generates a range of times based on the
    start date, time step, and variable's
    dimension size. 
    """

    # get the size of the dimension
    interval_size = variable_data.sizes[dimension]

    # convert the start time to correct np
    start_time_as_np = convert_date_or_datetime_to_np(start_time_as_dt)
    time_step_as_np = convert_timedelta_to_np(time_step)
    end_step_as_np = start_time_as_np + interval_size * time_step_as_np
    return np.arange(
        start=start_time_as_np,
        stop=end_step_as_np,
        step=time_step_as_np,
    )


def add_time_coords_to_idata_dimension(
    idata: az.InferenceData,
    group: str,
    variable: str,
    dimension: str,
    start_date_iso: datetime | date | np.datetime,
    time_step: timedelta,
) -> az.InferenceData:
    """
    Adds time coordinates to a specified
    variable within a group in an ArviZ
    InferenceData object. This function
    assigns a range of time coordinates
    to a specified dimension in a variable
    within a group in an InferenceData object.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object containing
        the group and variable to modify.
    group : str
        The name of the group within the
        InferenceData object (e.g.,
        "posterior_predictive").
    variable : str
        The name of the variable within the
        specified group to assign time
        coordinates to.
    dimension : str
        The dimension name to which time
        coordinates should be assigned.
    start_date_iso : date | datetime
        The start date for the time
        coordinates as a str in ISO format
        (e.g., "2022-08-20") or as a
        datetime object.
    time_step : timedelta
        The time interval between each
        coordinate (e.g., `timedelta(days=1)`
        for daily intervals).


    Returns
    -------
    az.InferenceData
        The InferenceData object with updated
        time coordinates for the specified
        group, variable, and dimension.
    """
    inputs = [
        (idata, az.InferenceData, "idata"),
        (group, str, "group"),
        (variable, str, "variable"),
        (dimension, str, "dimension"),
        (time_step, timedelta, "time_step"),
        (start_date_iso, (date, datetime), "start_date_iso"),
    ]
    for value, expected_type, param_name in inputs:
        forecasttools.validate_input_type(
            value=value, expected_type=expected_type, param_name=param_name
        )
    idata_group = forecasttools.validate_and_get_idata_group(
        idata=idata, group=group
    )
    variable_data = forecasttools.validate_and_get_idata_group_var(
        idata_group=idata_group, group=group, variable=variable
    )
    forecasttools.validate_idata_group_var_dim(
        variable_data=variable_data, dimension=dimension
    )
    interval_dates = generate_time_range_for_dim(
        start_time_as_dt=start_date_iso,
        variable_data=variable_data,
        dimension=dimension,
        time_step=time_step,
    )
    idata_group = idata_group.assign_coords({dimension: interval_dates})
    setattr(idata, group, idata_group)
    return idata


def add_time_coords_to_idata_dimensions(
    idata: az.InferenceData,
    groups: str | list[str],
    variables: str | list[str],
    dimensions: str | list[str],
    start_date_iso: str | datetime,
    time_step: timedelta,
) -> az.InferenceData:
    """
    Modifies the time-based coordinates across
    groups, variables, and dimensions. The
    function checks if the group, variable,
    and dimension exist, and then applies the
    specified time-based coordinates.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object to modify.
    groups : str | list[str]
        A group or a list of groups to modify
        (e.g., "posterior_predictive").
    variables : str | list[str]
        A variable or a list of variables
        within the specified groups to modify.
    dimensions : str | list[str]
        A dimension or a list of dimensions
        to modify for the variables.
    start_date_iso : date | datetime
        The start date for the time
        coordinates as a datetime date or as
        a datetime datetime.
    time_step : timedelta
        The time interval between each
        coordinate (e.g., `timedelta(days=1)`).

    Returns
    -------
    az.InferenceData
        The modified InferenceData object with
        updated time coordinates for the
        specified groups, variables, and
        dimensions.
    """
    inputs = [
        (idata, az.InferenceData, "idata"),
        (groups, (str, list), "groups"),
        (variables, (str, list), "variables"),
        (dimensions, (str, list), "dimensions"),
    ]
    for value, expected_type, param_name in inputs:
        forecasttools.validate_input_type(
            value=value, expected_type=expected_type, param_name=param_name
        )
    # if str, convert to list
    groups = forecasttools.ensure_listlike(groups)
    variables = forecasttools.ensure_listlike(variables)
    dimensions = forecasttools.ensure_listlike(dimensions)
    # check groups, variables, and dimensions
    # all contain str vars
    forecasttools.validate_iter_has_expected_types(groups, str, "groups")
    forecasttools.validate_iter_has_expected_types(variables, str, "variables")
    forecasttools.validate_iter_has_expected_types(
        dimensions, str, "dimensions"
    )
    # iterate over (group, variable, dimension) triples
    for group, variable, dimension in zip(groups, variables, dimensions):
        try:
            idata = add_time_coords_to_idata_dimension(
                idata=idata,
                group=group,
                variable=variable,
                dimension=dimension,
                start_date_iso=start_date_iso,
                time_step=time_step,
            )
        except ValueError as e:
            raise ValueError(
                f"Error for (group={group}, variable={variable}, dimension={dimension})."
            ) from e

    return idata


def idata_forecast_w_dates_to_df(
    idata_w_dates: az.InferenceData,
    location: str,
    postp_val_name: str,
    postp_dim_name: str,
    timepoint_col_name: str = "date",
    value_col_name: str = "hosp",
) -> pl.DataFrame:
    """
    Converts an Arviz InferenceData object
    into a Polars dataframe contains dates
    and draws for a given location. The
    number of rows in the dataframe is equal
    to the number of posterior predictive
    samples times the duration of the time
    series (end_date - start_date).
    """
    # get dates from InferenceData's posterior_predictive group
    dates = idata_w_dates.posterior_predictive.coords[postp_dim_name].values
    # convert the dates to ISO8601 strings
    iso8601_dates = np.datetime_as_string(dates, unit="D")
    # stack posterior predictive samples by chain and draw
    stacked_post_pred_samples = idata_w_dates.posterior_predictive.stack(
        sample=("chain", "draw")
    )[postp_val_name].to_pandas()
    # forecast dateframe wide (chain, draws) as cols
    forecast_df_wide = pl.from_pandas(stacked_post_pred_samples)
    forecast_df_wide = forecast_df_wide.with_columns(
        pl.Series(timepoint_col_name, iso8601_dates).cast(pl.Utf8)
    )
    forecast_df_unpivoted = (
        forecast_df_wide.unpivot(index=timepoint_col_name)
        .with_columns(
            draw=pl.col("variable").rank("dense").cast(pl.Int64),
            location=pl.lit(location),
        )
        .rename({"value": value_col_name})
    )
    return forecast_df_unpivoted
