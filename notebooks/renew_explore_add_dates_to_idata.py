# %% LOADING IDATA
import datetime as dt

import numpy as np
import xarray as xr

import forecasttools

xr.set_options(display_expand_data=False, display_expand_attrs=False)


# %% CREATING DATES ARRAY

# retrieve inference data object from forecasttools
idata = forecasttools.nhsn_flu_forecast

# create fictitious starting date for fitting
start_date_iso = "2022-08-01"
start_date_as_dt = dt.datetime.strptime(start_date_iso, "%Y-%m-%d")

# get the size of the fitting data
obs_length = idata.observed_data["obs_dim_0"].size

# create an array of dates corresponding to the fitting data
obs_dates = np.array(
    [
        (start_date_as_dt + dt.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(obs_length)
    ]
)

# %% ADDING DATES TO IDATA

# modify the inference data observed data with new dates
idata.observed_data = idata.observed_data.assign_coords(obs_dim_0=obs_dates)

# retrieve observed dates
obs_dates = idata.observed_data.coords["obs_dim_0"].values
print(obs_dates)


# examine other properties
obs_dates = idata.posterior_predictive.coords["obs_dim_0"].values
print(obs_dates)


# %%
