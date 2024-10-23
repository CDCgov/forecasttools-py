"""
Experimentation file for adding dates
to InferenceData objects via assigning
coordinates.
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
            + pl.duration(days=postp_length - 1),
            interval="1d",
            closed="both",
        )
    )
    .to_series()
    .to_numpy()
    .astype("datetime64[ns]")
)


# %% ADDING DATES TO IDATA

# modify the idata group observed data with new dates
idata.observed_data = (
    idata.observed_data.assign_coords(
        obs_dim_0=obs_dates
    )
)

# modify the idata group posterior predictive with new dates
idata.posterior_predictive = (
    idata.posterior_predictive.assign_coords(
        obs_dim_0=postp_dates
    )
)

# WARNING:
# <ipython-input-5-4b6b6e9c7c9c>:4: UserWarning: Converting non-nanosecond precision datetime values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.
# fixed via: --> .astype("datetime64[ns]")
#   idata.observed_data = idata.observed_data.assign_coords(obs_dim_0=obs_dates)

# %% EXAMINE NEW DATED IDATA GROUPS

# examine observed_data dates
obs_dates_out = idata.observed_data.coords[
    "obs_dim_0"
].values
print(
    f"Obs start should be: {start_date_iso}\n{obs_dates_out[-10:]}"
)

# examine posterior_predictive dates
postp_dates_out = (
    idata.posterior_predictive.coords[
        "obs_dim_0"
    ].values
)
print(
    f"PostP start should be: {start_date_iso}\n{postp_dates_out[-10:]}"
)
print(
    f"PostP size should be greater than obs group:\nPostP: {len(postp_dates_out)} v. Obs: {len(obs_dates_out)}, Diff: {len(postp_dates_out) - len(obs_dates_out)}"
)

# %% PLOT USING PLOT TS WITH DATES

az.plot_ts(idata, y="obs")
plt.show()
