"""
This file serves as a place for UPX3 to
explore suggestions provided here:
https://github.com/CDCgov/forecasttools-py/pull/10

This file will be deleted before the
PR is merged.
"""

# %% IMPORTS

import polars as pl

import forecasttools

# %% USING POLARS EXPR.REPLACE()

# example dataframe and location table
example_dict = {
    "date": ["2024-10-08", "2024-10-08", "2024-10-08"],
    "location": ["AL", "AK", "US"],
}
example_df = pl.from_dict(example_dict)
loc_table = forecasttools.location_table
print(example_df)
print(loc_table)

# replace "location" values with codes from loc_table
new_df = example_df.with_columns(
    location=pl.col("location").replace(
        old=loc_table["short_name"], new=loc_table["location_code"]
    )
)
print(new_df)

# %%
