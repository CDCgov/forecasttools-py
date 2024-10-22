# %% LOADING IDATA
import datetime as dt

import polars as pl
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
dim_name = "obs_dim_0"
obs_length = idata.observed_data.sizes[dim_name]

# suggested idata.observed_data.sizes["obs_dim_0"]
# over idata.observed_data["obs_dim_0"].size

# idata.observed_data.sizes
# Frozen({'obs_dim_0': 488})

# idata.observed_data.dims gets one:
# FrozenMappingWarningOnValuesAccess({'obs_dim_0': 488})

# ONE OPTION
# obs_dates = (
#     pl.DataFrame()
#     .select(
#         pl.date_range(
#             start=start_date_as_dt,
#             end=start_date_as_dt + dt.timedelta(days=obs_length - 1),
#             interval="1d",
#             closed="both",
#         )
#     )
#     .to_series()
#     .to_numpy()
# )

# ANOTHER OPTION
obs_dates = (
    pl.DataFrame()
    .select(
        pl.date_range(
            start=start_date_as_dt,
            end=start_date_as_dt + pl.duration(days=obs_length - 1),
            interval="1d",
            closed="both",
        )
    )
    .to_series()
    .to_numpy()
)

# %% DATE CALCULATIONS FOR POSTERIOR PREDICTIVE

# the length of the posterior predictive data becomes its date array
dim_name = "obs_dim_0"
postp_length = idata.posterior_predictive.sizes[dim_name]
postp_dates = (
    pl.DataFrame()
    .select(
        pl.date_range(
            start=start_date_as_dt,
            end=start_date_as_dt + dt.timedelta(days=postp_length - 1),
            interval="1d",
            closed="both",
        )
    )
    .to_series()
    .to_numpy()
)


# %% ADDING DATES TO IDATA

# modify the idata group observed data with new dates
idata.observed_data = idata.observed_data.assign_coords(obs_dim_0=obs_dates)

# as mapping for coord for individual variables OR coords
# DB concerned groups afford little value

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

# Absolutely the case that something that is handling dates that have already been handed to idata objects should not assume intervals, ordered, as agnostic as possible (only knowledge: "of type date")
# A function for daily data.
#
# Task: creating idata obj w/ dates --> demo func in forecastools
# TPM make case for dates as coords or as group
# show w/ plot_ts()
# Task: more general add dates to idata wo data

# %% ADDING DATES GROUP TO IDATA

idata_w_dates = idata.copy()

obs_dates_da = xr.DataArray(
    obs_dates,
    dims=["obs_dim_0"],  # name might need to change
    coords={"obs_dim_0": obs_dates},
    name="obs_dates",
)

postp_dates_da = xr.DataArray(
    postp_dates,
    dims=["obs_dim_0"],  # name might need to change
    coords={"obs_dim_0": postp_dates},  # name might need to change
    name="postp_dates",
)

idata_w_dates.add_groups(
    date_info={"obs_dates": obs_dates_da, "postp_dates": postp_dates_da}
)

# what happens when you get rid of a date?

# %% EXAMINING IDATA WITH DATES GROUP

print(idata_w_dates)

print(idata_w_dates.date_info)

print(idata_w_dates.date_info.obs_dates.to_numpy())

# %%
