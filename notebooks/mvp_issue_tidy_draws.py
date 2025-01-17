"""
Script demonstrating an issue with assigning
coordinates to an InferenceData object can
produce.
"""

# %% LIBRARIES

from datetime import date, datetime, timedelta

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

# %% DEMONSTRATION OF EXPECTED BEHAVIOR

obs_data = np.random.normal(loc=0, scale=1, size=(1,))
obs_dim_name = "obs_dim_0"
start_date = "2023-01-01"
interval_dates = pd.date_range(start=start_date, periods=1, freq="D")
obs_group = xr.Dataset(
    {"obs": ([obs_dim_name], obs_data)}, coords={obs_dim_name: np.arange(1)}
)
idata = az.from_dict(observed_data={"obs": obs_group["obs"].values})
idata.observed_data = idata.observed_data.assign_coords(
    {obs_dim_name: interval_dates}
)
df = idata.observed_data.to_dataframe()
print(df)

# %% FAILED EXAMPLE


def convert_date_or_datetime_to_np(
    dt: datetime | date | np.datetime64,
) -> np.datetime64:
    """Converts a Python date/datetime or numpy.datetime64 to numpy.datetime64."""
    if isinstance(dt, (datetime, date)):
        return np.datetime64(dt)
    elif isinstance(dt, np.datetime64):
        return dt
    else:
        raise TypeError(
            f"Input must be a date, datetime, or np.datetime64; got {type(dt)}"
        )


def is_timedelta_in_days_only(td: timedelta) -> bool:
    """Checks if a timedelta is representable in days only."""
    return td.seconds == 0 and td.microseconds == 0


def convert_timedelta_to_np(td: timedelta | np.timedelta64) -> np.timedelta64:
    """
    Converts a Python timedelta to:
    numpy.timedelta64[D] if it is
    representable in days only. Otherwise,
    convert to numpy.timedelta64[ns]. If
    already np.timedelta64, return as is.
    """
    if isinstance(td, np.timedelta64):
        return td
    elif isinstance(td, timedelta):
        return (
            np.timedelta64(td.days, "D")
            if is_timedelta_in_days_only(td)
            else np.timedelta64(td).astype("timedelta64[ns]")
        )
    else:
        raise TypeError(f"Input must be a timedelta object; got {type(td)}")


def generate_time_range_for_dim(
    start_time_as_dt: datetime | date | np.datetime64,
    variable_data: xr.DataArray,
    dimension: str,
    time_step: timedelta | np.timedelta64,
) -> np.ndarray:
    """
    Generates a range of times based on the
    start date, time step, and variable's
    dimension size.
    """
    interval_size = variable_data.sizes[dimension]
    start_time_as_np = convert_date_or_datetime_to_np(start_time_as_dt)
    time_step_as_np = convert_timedelta_to_np(time_step)
    end_step_as_np = start_time_as_np + interval_size * time_step_as_np
    return np.arange(
        start=start_time_as_np,
        stop=end_step_as_np,
        step=time_step_as_np,
    )


num_days = 11
obs_data = np.random.normal(loc=0, scale=1, size=(num_days,))
obs_dim_name = "obs_dim_0"


start_date = datetime(2023, 1, 1)
time_step = timedelta(days=2)
interval_dates = generate_time_range_for_dim(
    start_time_as_dt=start_date,
    variable_data=xr.DataArray(obs_data, dims=[obs_dim_name]),
    dimension=obs_dim_name,
    time_step=time_step,
)
obs_group = xr.Dataset(
    {"obs": ([obs_dim_name], obs_data)},
    coords={obs_dim_name: np.arange(num_days)},
)
idata = az.from_dict(observed_data={"obs": obs_group["obs"].values})
print(idata["observed_data"])
idata.observed_data = idata.observed_data.assign_coords(
    {obs_dim_name: interval_dates}
)
df = idata.observed_data.to_dataframe()
print(df)

# %%
