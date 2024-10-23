"""
Experimentation file for
examining how idata with
dates operate.
"""

# %% IMPORT LIBRARIES


import xarray as xr

import forecasttools

xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
)


# %% GET IDATA WITHOUT DATES

idata_wo_dates = (
    forecasttools.nhsn_flu_forecast_wo_dates
)

print(idata_wo_dates)

print(
    idata_wo_dates.observed_data[
        "obs_dim_0"
    ].values[:10]
)

# %% ADD DATES TO IDATA OBJECT GROUP(S)


start_date_iso = "2022-08-08"
idata_w_dates = (
    forecasttools.add_dates_as_coords_to_idata(
        idata_wo_dates=idata_wo_dates,
        group_dim_dict={
            "observed_data": "obs_dim_0",
            "posterior_predictive": "obs_dim_0",
            "prior_predictive": "obs_dim_0",
        },
        start_date_iso=start_date_iso,
    )
)

print(idata_w_dates)

print(
    idata_w_dates.observed_data[
        "obs_dim_0"
    ].values[:10]
)


# np.datetime is the type rather than just datetime
# start_date_iso = np.datetime_as_string(
#     idata_wo_dates.observed_data["obs_dim_0"].values[0], unit="D")

# %% CONVERT DATES IDATA TO DATAFRAME

df_from_idata_w_dates = (
    forecasttools.idata_forecast_w_dates_to_df(
        idata_w_dates=idata_w_dates,
        location="TX",
        postp_val_name="obs",
        postp_dim_name="obs_dim_0",
        timepoint_col_name="date",
        value_col_name="hosp",
    )
)
df_from_idata_w_dates

# %%
