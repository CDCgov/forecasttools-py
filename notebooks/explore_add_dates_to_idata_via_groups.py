"""
Experimentation file for adding dates
to InferenceData objects via adding
groups.
"""

# %% LOADING LIBRARIES
import datetime as dt

import arviz as az
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr

import forecasttools

xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
)


# %% SETTING UP START DATE AND IDATA

# received by function is a str ISO8601 start date
start_date_iso = "2022-08-01"
start_date_as_dt = dt.datetime.strptime(
    start_date_iso, "%Y-%m-%d"
)

# and an inference data object (built az.from_numpyro())
idata = forecasttools.nhsn_flu_forecast

# examine groups available with idata
print(idata)

# %% DATE CALCULATIONS FOR OBSERVED DATA

# the length of the observed data becomes its date array
obs_dim_name = "obs_dim_0"
obs_length = idata.observed_data.sizes[
    obs_dim_name
]

# creates dates array
obs_dates = (
    pl.DataFrame()
    .select(
        pl.date_range(
            start=start_date_as_dt,
            end=start_date_as_dt
            + pl.duration(days=obs_length - 1),
            interval="1d",
            closed="both",
        )
    )
    .to_series()
    .to_numpy()
    .astype("datetime64[ns]")
)

# %% DATE CALCULATIONS FOR POSTERIOR PREDICTIVE

# same as with observed_data group but not for
# posterior predictive group
postp_dim_name = "obs_dim_0"
postp_length = idata.posterior_predictive.sizes[
    postp_dim_name
]
postp_dates = (
    pl.DataFrame()
    .select(
        pl.date_range(
            start=start_date_as_dt,
            end=start_date_as_dt
            + dt.timedelta(days=postp_length - 1),
            interval="1d",
            closed="both",
        )
    )
    .to_series()
    .to_numpy()
    .astype("datetime64[ns]")
)


# %% ADDING DATES GROUP TO IDATA

idata_w_dates = idata.copy()


obs_dates_da = xr.DataArray(
    obs_dates,
    dims=[
        obs_dim_name
    ],  # name might need to change
    coords={obs_dim_name: obs_dates},
    name="obs_dates",
)

postp_dates_da = xr.DataArray(
    postp_dates,
    dims=[
        postp_dim_name
    ],  # name might need to change
    coords={
        postp_dim_name: postp_dates
    },  # name might need to change
    name="postp_dates",
)

# WARNINGS
# UserWarning: Converting non-nanosecond precision datetime values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.
# --> fixed via: .astype("datetime64[ns]")
# UserWarning: The group postp_dates is not defined in the InferenceData scheme
# UserWarning: The group obs_dates is not defined in the InferenceData scheme


dates_dataset = xr.Dataset(
    {
        "obs_dates": obs_dates_da,
        "postp_dates": postp_dates_da,
    }
)

idata_w_dates.add_groups(
    {
        "obs_dates": obs_dates_da,
        "postp_dates": postp_dates_da,
    }
)

# %% EXAMINING IDATA WITH DATES GROUP

print(idata_w_dates)

print(idata_w_dates.obs_dates)

print(idata_w_dates.postp_dates)

# %% PLOT USING PLOT TS WITH DATES

az.plot_ts(idata_w_dates, y="obs")
plt.show()

# %%
