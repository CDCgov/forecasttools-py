"""
Functions to work with recoding columns
containing US jurisdiction location codes
and abbreviations.
"""

import polars as pl

import forecasttools


def loc_abbr_to_flusight_code(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    """
    Takes the location columns of a Polars
    dataframe and recodes it to FluSight
    location codes.


    Parameters
    ----------
    df
        A Polars dataframe with a location
        column.
    location_col
        The name of the dataframe's location
        column.

    Returns
    -------
    pl.DataFrame
        A recoded locations dataframe.
    """
    # get location table
    loc_table = forecasttools.location_table
    # recode and replaced existing loc abbrs with loc codes
    loc_recoded_df = df.with_columns(
        location=pl.col("location").replace(
            old=loc_table["short_name"], new=loc_table["location_code"]
        )
    )
    return loc_recoded_df
