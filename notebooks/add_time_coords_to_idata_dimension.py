"""
Experiment file for examining adding
time coordinates to an idata's dimension.
"""

# %% LIBRARY IMPORTS

from datetime import timedelta

import forecasttools

# %% LOAD IDATA WO DATES

idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

print(idata_wo_dates)

# # group level
# print(idata_wo_dates["posterior_predictive"])
# # variable level
# print(idata_wo_dates["posterior_predictive"]["obs"])
# # dimension level
# print(idata_wo_dates["posterior_predictive"]["obs"]["obs_dim_0"])

# %% MAKE POSTERIOR PREDICTIVE's OBS DATED

idata_w_dates = forecasttools.add_time_coords_to_idata_dimension(
    idata=idata_wo_dates,
    group="posterior_predictive",
    variable="obs",
    dimension="obs_dim_0",
    start_date_iso="2024-08-01",
    time_step=timedelta(days=1),
)

print(idata_w_dates["posterior_predictive"]["obs"]["obs_dim_0"])
# %%
