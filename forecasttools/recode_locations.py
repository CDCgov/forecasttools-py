import polars as pl

import forecasttools


def loc_abbr_to_flusight_code(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    # get location table
    loc_table = forecasttools.location_table
    # recode and replaced existing loc abbrs with loc codes
    loc_recoded_df = df.with_columns(
        location=pl.col("location").replace(
            old=loc_table["short_name"], new=loc_table["location_code"]
        )
    )
    return loc_recoded_df
