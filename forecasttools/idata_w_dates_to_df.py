"""
Contains functions for converting
Arviz InferenceData objects to Polars
dataframes with dates and draws and
for working with conversions of
Polars dataframes to hubverse ready and
scoringutils ready dataframes
"""

from datetime import datetime, timedelta

import arviz as az
import numpy as np
import polars as pl


def add_time_coords_to_idata_dimension(
    idata: az.InferenceData,
    group: str,
    variable: str,
    dimension: str,
    start_date_iso: str | datetime,
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
    start_date_iso : str | datetime
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
    # checking of inputted variables
    if not isinstance(idata, az.InferenceData):
        raise TypeError(
            f"Parameter 'idata' must be of type 'az.InferenceData'; got {type(idata)}"
        )
    if not isinstance(group, str):
        raise TypeError(
            f"Parameter 'group' must be of type 'str'; got {type(group)}"
        )
    if not isinstance(variable, str):
        raise TypeError(
            f"Parameter 'variable' must be of type 'str'; got {type(variable)}"
        )
    if not isinstance(dimension, str):
        raise TypeError(
            f"Parameter 'dimension' must be of type 'str'; got {dimension}."
        )
    if isinstance(start_date_iso, str):
        try:
            start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Parameter 'start_date_iso' must be in the format 'YYYY-MM-DD' if provided as a string; got {start_date_iso}"
            )
    elif isinstance(start_date_iso, datetime):
        start_date_as_dt = start_date_iso
    else:
        raise TypeError(
            f"Parameter 'start_date_iso' must be of type 'str' or 'datetime'; got {type(start_date_iso)}"
        )
    if not isinstance(time_step, timedelta):
        raise TypeError(
            f"Parameter 'time_step' must be of type 'datetime.timedelta'; got {type(time_step)}"
        )
    # retrieve the specified group from the
    # idata object
    idata_group = getattr(idata, group, None)
    # check if the group is not present
    if idata_group is None:
        raise ValueError(f"Group '{group}' not found in idata object.")
    # check if the specified variable exists
    # in the idata's group
    if variable not in idata_group.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in group '{group}'."
        )
    # retrieve the variable's data array
    variable_data = idata_group[variable]
    # check and apply time coordinates only
    # to the specified dimensions that exist
    # in the variable
    if dimension not in variable_data.dims:
        raise ValueError(
            f"Dimension '{dimension}' not found in variable dimensions: '{variable_data.dims}'."
        )
    # determine the interval size for the
    # selected dimension
    interval_size = variable_data.sizes[dimension]
    # generate date range using the specified time_step and interval size
    interval_dates = (
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
    # update the dimension's coordinates
    # (corresponding to the passed variable)
    idata_group = idata_group.assign_coords({dimension: interval_dates})
    setattr(idata, group, idata_group)
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
