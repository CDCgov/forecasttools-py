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

    Returns
    -------
    pl.DataFrame
        Recorded locations dataframe.
    """
    # get location table
    loc_table = forecasttools.location_table
    # recode and replaced existing loc abbrs with loc codes
    loc_recoded_df = df.with_columns(
        location=pl.col("location").replace(
            old=loc_table["short_name"],
            new=loc_table["location_code"],
        )
    )
    return loc_recoded_df


def loc_flusight_code_to_abbr(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    """
    Convert a FluSight location code to a two-letter
    state/territory abbreviation.

    Parameters
    ----------
    df
        A Polars dataframe with a location column.
    location_col
        The name of the dataframe's location column.

    Returns
    -------
    pl.DataFrame
        Recoded FluSight locations table.
    """
    # get forecasttools location table
    loc_table = forecasttools.location_table

    # recode location codes to location abbreviations
    loc_recoded_df = df.with_columns(
        pl.col(location_col).replace(
            old=loc_table["location_code"], new=loc_table["short_name"]
        )
    )

    return loc_recoded_df


def to_location_table_column(location_format: str) -> str:
    """
    Map a location format string to the
    corresponding column name in the hubserve
    location table.

    Parameters
    ----------
    location_format
        The format string ("abbr",
        "flusight", or "long_name").

    Returns
    -------
    str
        Returns the corresponding column name from
        the location table.
    """
    col_keys = {
        "abbr": "short_name",
        "flusight": "location_code",
        "long_name": "long_name",
    }

    col_key = col_keys.get(location_format)

    if col_key is None:
        raise ValueError(
            f"Unknown location format {location_format}. Expected 'abbr', 'flusight', or 'long_name'."
        )

    return col_key
