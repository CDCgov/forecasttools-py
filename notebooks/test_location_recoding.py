"""
Testing the historical recode locations
utilities and possibly experimenting with
additional utilities.
"""

# %% LIBRARY IMPORTS

import xarray as xr

import forecasttools

xr.set_options(display_expand_data=False, display_expand_attrs=False)


# %% LOAD FLUSIGHT LOCATION TABLE

location_table = forecasttools.location_table
print(location_table["location_code"])


# %% LOAD FLUSIGHT SUBMISSION

flusight_submission = forecasttools.example_flusight_submission
print(flusight_submission)


# %% SHOW LOCATION COLUMN OF FLUSIGHT SUBMISSION

loc_col = flusight_submission["location"]
print(loc_col)

# %% RECODE LOCATION COLUMN TO US ABBREVIATION

df_code_to_abbr = forecasttools.loc_flusight_code_to_abbr(
    df=flusight_submission, location_col="location"
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
    location_vector=["US", "01", "65"], location_format="flusight"
)
print(matching_loc_cols_df_02)

# %% LOCATION LOOKUP (ITER 3)

matching_loc_cols_df_03 = forecasttools.location_lookup(
    location_vector=["US", "Alabama", "Mississippi", "New Hampshire"],
    location_format="long_name",
)
print(matching_loc_cols_df_03)

# %%
