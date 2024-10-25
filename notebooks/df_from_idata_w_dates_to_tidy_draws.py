"""
Experimentation file for
examining how an idata with
dates can be converted to
tidy_draws format. Made for
Damon.
"""

# %% IMPORT LIBRARIES


import os
import subprocess
import tempfile

import xarray as xr

import forecasttools

xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
)

# %% ADD DATES TO IDATA OBJECT GROUP(S)

# load inference data wo/ dates
idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

# convert to inference data with dates
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
idata_w_dates

# %% CONVERT TO DATAFRAME


df_from_idata_w_dates = forecasttools.idata_forecast_w_dates_to_df(
    idata_w_dates=idata_w_dates,
    location="TX",
    postp_val_name="obs",
    postp_dim_name="obs_dim_0",
    timepoint_col_name="date",
    value_col_name="hosp",
).drop("variable")
df_from_idata_w_dates


# %% SAVE OUTPUT TO PARQUET

# save data as parquet file
df_from_idata_w_dates.write_parquet("output_tidy_draws.parquet")


# %% INCLUDE R CODE

r_code = """
# read save parquet file
tidy_draws_df <- arrow::read_parquet("output_tidy_draws.parquet")

# checks
is(tidy_draws_df, "data.frame")
dplyr::glimpse(tidy_draws_df)

# verify tidy_draw structure
dplyr::count(tidy_draws_df, draw, location, date, hosp)

# missing values?
print("Missing values?")
print(summary(tidy_draws_df))
"""

# %% RUN R CODE

with tempfile.NamedTemporaryFile(suffix=".R", delete=False) as temp_r_file:
    temp_r_file.write(r_code.encode("utf-8"))
    temp_r_file_path = temp_r_file.name

try:
    subprocess.run(["Rscript", temp_r_file_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"R script failed with error: {e}")
finally:
    os.remove(temp_r_file_path)

# %% REMOVE FILES

if os.path.exists("output_tidy_draws.parquet"):
    subprocess.run(["rm", "output_tidy_draws.parquet"])


# %%
