# %% LOADING IDATA
import datetime as dt

import numpy as np
import xarray as xr

import forecasttools

xr.set_options(display_expand_data=False, display_expand_attrs=False)

# accept either str or datetime
# polars date col as datetime
# change .draw to draw
# generic reference wo/ string construction (.dim)
# obs_dim_0 --> day_index

# %% SETTING UP START DATE AND IDATA

# received by function is a str ISO8601 start date
start_date_iso = "2022-08-01"
start_date_as_dt = dt.datetime.strptime(start_date_iso, "%Y-%m-%d")

# and an inference data object (built az.from_numpyro())
idata = forecasttools.nhsn_flu_forecast

# examine groups available with idata
print(idata)

# %% DATE CALCULATIONS FOR OBSERVED DATA

# the length of the observed data becomes its date array
obs_length = idata.observed_data["obs_dim_0"].size
obs_dates = np.array(
    [
        (start_date_as_dt + dt.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(obs_length)
    ]
)

# %% DATE CALCULATIONS FOR POSTERIOR PREDICTIVE

# the length of the posterior predictive data becomes its date array
postp_length = idata.posterior_predictive["obs_dim_0"].size
postp_dates = np.array(
    [
        (start_date_as_dt + dt.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(postp_length)
    ]
)


# %% ADDING DATES TO IDATA

# modify the idata group observed data with new dates
idata.observed_data = idata.observed_data.assign_coords(obs_dim_0=obs_dates)

# modify the idata group posterior predictive with new dates
idata.posterior_predictive = idata.posterior_predictive.assign_coords(
    obs_dim_0=postp_dates
)

# %% EXAMINE NEW DATED IDATA GROUPS

# examine observed_data dates
obs_dates_out = idata.observed_data.coords["obs_dim_0"].values
print(f"Obs start should be: {start_date_iso}\n{obs_dates_out[:10]}")

# examine posterior_predictive dates
postp_dates_out = idata.posterior_predictive.coords["obs_dim_0"].values
print(f"PostP start should be: {start_date_iso}\n{postp_dates_out[:10]}")
print(
    f"PostP size should be greater than obs group:\nPostP: {len(postp_dates_out)} v. Obs: {len(obs_dates_out)}, Diff: {len(postp_dates_out) - len(obs_dates_out)}"
)


# %%
