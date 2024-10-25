"""
Experimentation file for
going from an idata without
dates to scores via ScoringUtils.
"""

# %% IMPORT LIBRARIES


import xarray as xr

import forecasttools

xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
)

# %% GET IDATA WITHOUT DATES

idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

# %% ADD DATES TO IDATA OBJECT GROUP(S)

start_date_iso = "2022-08-08"
idata_w_dates = forecasttools.add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates,
    group_dim_dict={
        "observed_data": "obs_dim_0",
        "posterior_predictive": "obs_dim_0",
        "prior_predictive": "obs_dim_0",
    },
    start_date_iso=start_date_iso,
)

# %% GET OBSERVATIONS FOR FORECAST PERIOD

# get dates from posterior predictive and observed data
postp_dim_name = "obs_dim_0"
postp_dates = idata_w_dates.posterior_predictive[postp_dim_name].values
obs_dim_name = "obs_dim_0"
obs_dates = idata_w_dates.observed_data[obs_dim_name].values

# get influenza data that

# load influenza data
nhsn_flu_data = forecasttools.nhsn_hosp_flu
