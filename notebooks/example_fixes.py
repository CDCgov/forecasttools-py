"""
This file serves as a place for UPX3 to
explore suggestions provided here:
https://github.com/CDCgov/forecasttools-py/pull/10

This file will be deleted before the
PR is merged.
"""

# %% IMPORTS


import datetime as dt

import arviz as az
import epiweeks
import polars as pl
import xarray as xr

import forecasttools

# %% SIMPLE EPI COLUMNS


# create artificial date dataframe
df_dict = {
    "date": ["2024-01-01", "2024-01-06", "2024-01-08"],
    "value": [1, 2, 2],
}
df = pl.DataFrame(df_dict)
print(df)

# method 01: use map elements
new_df = df.with_columns(
    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
).with_columns(
    pl.col("date")
    .map_elements(
        lambda d: epiweeks.Week.fromdate(d).week, return_dtype=pl.Int64
    )
    .alias("epiweek"),
    pl.col("date")
    .map_elements(
        lambda d: epiweeks.Week.fromdate(d).year, return_dtype=pl.Int64
    )
    .alias("epiyear"),
)


# method 02: use unnest w/ a function
def calculate_epidate(date: str):
    week_obj = epiweeks.Week.fromdate(dt.datetime.strptime(date, "%Y-%m-%d"))
    return {"epiweek": week_obj.week, "epiyear": week_obj.year}


newer_df = df.with_columns(
    pl.struct(["date"])
    .map_elements(
        lambda x: calculate_epidate(x["date"]), return_dtype=pl.Struct
    )
    .alias("epi_struct")
).unnest("epi_struct")
print(newer_df)


# %% USE OF PREDICTIONS IN ARVIZ

# load idata object from forecasttools
idata = forecasttools.nhsn_flu_forecast

# has predictions by default
print(idata.predictions["obs_dim_0"])
print(idata.observed_data["obs_dim_0"])
print(idata.posterior_predictive["obs_dim_0"])

# get posterior samples
postp_samps = idata.posterior_predictive["obs"]

# break into forecast and fit component
forecast = postp_samps.isel(obs_dim_0=slice(-28, None))
fitted = postp_samps.isel(obs_dim_0=slice(None, -28))

# create predictions
predictions_dict = {"obs": forecast}
predictions_idata = az.InferenceData(predictions=xr.Dataset(predictions_dict))


# (attempt) edit original idata object
idata.posterior_predictive["obs"] = fitted
idata = idata.extend(predictions_idata)


# %% PRINT MARKDOWN VERSION OF DATAFRAME

loc_table = forecasttools.location_table
flu_fit = forecasttools.nhsn_hosp_flu
cov_fit = forecasttools.nhsn_hosp_COVID
sub = forecasttools.example_flusight_submission
idata = forecasttools.nhsn_flu_forecast
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
with pl.Config(tbl_rows=15):
    # print(f"Location Table:\n{loc_table}\n")
    # print(f"Flu Fitting:\n{flu_fit}\n")
    # print(f"COVID Fitting:\n{cov_fit}\n")
    # print(f"Submission:\n{sub}\n")
    print(f"Forecast:\n{idata}\n")

# %% EXAMINING FILE PATH BEING RETURNED

# example dataframe and location table
example_dict = {
    "date": ["2024-10-08", "2024-10-08", "2024-10-08"],
    "location": ["AL", "AK", "US"],
}
example_df = pl.from_dict(example_dict)

# call loc_abbr_to_flusight_code()
recoded_df = forecasttools.loc_abbr_to_flusight_code(
    df=example_df, location_col="location"
)

# received warning (original implementation)
# MapWithoutReturnDtypeWarning: Calling `map_elements` without specifying `return_dtype` can lead to unpredictable results. Specify `return_dtype` to silence this warning.

# no longer receives warning after using expr.replace()


# %% USING POLARS EXPR.REPLACE()

# example dataframe and location table
example_dict = {
    "date": ["2024-10-08", "2024-10-08", "2024-10-08"],
    "location": ["AL", "AK", "US"],
}
example_df = pl.from_dict(example_dict)
loc_table = forecasttools.location_table

# replace "location" values with codes from loc_table
new_df = example_df.with_columns(
    location=pl.col("location").replace(
        old=loc_table["short_name"], new=loc_table["location_code"]
    )
)

# %%
