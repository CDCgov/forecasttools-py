"""
Testing the historical recode locations
utilities and possibly experimenting with
additional utilities.
"""

# %% LIBRARY IMPORTS

import xarray as xr

import forecasttools

xr.set_options(display_expand_data=False, display_expand_attrs=False)


# %% LOAD hubverse LOCATION TABLE

location_table = forecasttools.location_table
print(location_table["location_code"])


# %% LOAD hubverse SUBMISSION

hubverse_submission = forecasttools.example_hubverse_submission
print(hubverse_submission)


# %% SHOW LOCATION COLUMN OF hubverse SUBMISSION

loc_col = hubverse_submission["location"]
print(loc_col)

# %% RECODE LOCATION COLUMN TO US ABBREVIATION

df_code_to_abbr = forecasttools.loc_hubverse_code_to_abbr(
    df=hubverse_submission, location_col="location"
)
new_loc_col = df_code_to_abbr["location"]
print(new_loc_col)
# %% LOCATION LOOKUP (ITER 1)

matching_loc_cols_df_01 = forecasttools.location_lookup(
    location_vector=["AL", "AK"], location_format="abbr"
)
print(matching_loc_cols_df_01)

# %% LOCATION LOOKUP (ITER 2)

matching_loc_cols_df_02 = forecasttools.location_lookup(
    location_vector=["US", "01", "65"], location_format="hubverse"
)
print(matching_loc_cols_df_02)

# %% LOCATION LOOKUP (ITER 3)

matching_loc_cols_df_03 = forecasttools.location_lookup(
    location_vector=["US", "Alabama", "Mississippi", "New Hampshire"],
    location_format="long_name",
)
print(matching_loc_cols_df_03)

# %% LOCATION LOOKUP (ITER 4), FAILURE CHECK

matching_loc_cols_df_03 = forecasttools.location_lookup(
    location_vector=["US", "Alabama", "Mississippi", "New Hampshire"],
    location_format="failed_name",
)
print(matching_loc_cols_df_03)

# %%
