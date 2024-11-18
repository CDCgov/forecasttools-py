# %% LIBRARY IMPORTS

from datetime import datetime, timedelta

import polars as pl
import xarray as xr

import forecasttools

# %% FUNCTION TO CHECK


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

    # get the size of the dimension
    interval_size = variable_data.sizes[dimension]
    # number of seconds in a day
    SECONDS_IN_DAY = timedelta(days=1).total_seconds()

    # total number of seconds in the time_step
    total_seconds = time_step.total_seconds()

    # determine the interval string for Polars
    if (
        total_seconds % SECONDS_IN_DAY == 0
    ):  # check if time_step is in full days
        # use date_range for dates
        return (
            pl.date_range(
                start=start_date_as_dt,
                end=start_date_as_dt + (interval_size - 1) * time_step,
                interval=time_step,  # use the calculated interval
                closed="both",
                eager=True,  # return a Polars Series
            )
            .to_numpy()
            .astype("datetime64[D]")  # date format
        )
    else:
        # use datetime_range for times
        return (
            pl.datetime_range(
                start=start_date_as_dt,
                end=start_date_as_dt + (interval_size - 1) * time_step,
                interval=time_step,  # use the calculated interval
                closed="both",
                eager=True,  # return a Polars Series
            )
            .to_numpy()
            .astype("datetime64[ns]")  # Ensure consistent datetime format
        )


# %% TEST ABOVE FUNCTION

IDATA_WO_DATES = forecasttools.nhsn_flu_forecast_wo_dates

variable_data = IDATA_WO_DATES["posterior_predictive"]["obs"]

start_date_iso = "2022-08-01"

start_date_as_dt = forecasttools.validate_and_get_start_date(start_date_iso)

as_dates_idata = generate_time_range_for_dim(
    start_date_as_dt=start_date_as_dt,
    variable_data=variable_data,
    dimension="obs_dim_0",
    time_step=timedelta(days=1),
)
print(as_dates_idata[:5], type(as_dates_idata[0]))

as_time_idata = generate_time_range_for_dim(
    start_date_as_dt=start_date_as_dt,
    variable_data=variable_data,
    dimension="obs_dim_0",
    time_step=timedelta(days=1.5),
)
print(as_time_idata[:5], type(as_time_idata[0]))
# %% ANOTHER CHECK

as_dates_idata = generate_time_range_for_dim(
    start_date_as_dt=start_date_as_dt,
    variable_data=variable_data,
    dimension="obs_dim_0",
    time_step=timedelta(weeks=1),
)
print(as_dates_idata[:5], type(as_dates_idata[0]))

as_time_idata = generate_time_range_for_dim(
    start_date_as_dt=start_date_as_dt,
    variable_data=variable_data,
    dimension="obs_dim_0",
    time_step=timedelta(weeks=1.5),
)
print(as_time_idata[:5], type(as_time_idata[0]))

# %%
