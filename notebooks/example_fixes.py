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