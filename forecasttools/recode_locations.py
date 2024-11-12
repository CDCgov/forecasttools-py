"""
Functions to work with recoding location
columns containing US jurisdiction location
codes or two-letter abbreviations.
"""

import polars as pl

import forecasttools


def loc_abbr_to_hubverse_code(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    """
    Takes the location column of a Polars
    dataframe (formatted as US two-letter
    jurisdictional abbreviations) and recodes
    it to hubverse location codes using
    location_table, which is a Polars
    dataframe contained in forecasttools.

    Parameters
    ----------
    df
        A Polars dataframe with a location
        column consisting of US
        jurisdictional abbreviations.
    location_col
        The name of the dataframe's location
        column.

    Returns
    -------
    pl.DataFrame
        A Polars dataframe with the location
        column formatted as hubverse location
        codes.
    """
    # check inputted variable types
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected a Polars DataFrame; got {type(df)}.")
    if not isinstance(location_col, str):
        raise TypeError(
            f"Expected a string for location_col; got {type(location_col)}."
        )
    # check if dataframe entered is empty
    if df.is_empty():
        raise ValueError(f"The dataframe {df} is empty.")
    # check if the location column exists
    # in the inputted dataframe
    if location_col not in df.columns:
        raise ValueError(
            f"Column '{location_col}' not found in the dataframe; got {df.columns}."
        )
    # get location table from forecasttools
    loc_table = forecasttools.location_table
    # check if values in location_col are a
    # subset of short_name in location table
    location_values = set(df[location_col].to_list())
    valid_values = set(loc_table["short_name"].to_list())
    difference = location_values.difference(valid_values)
    if difference:
        raise ValueError(
            f"The following values in '{location_col}') are not valid jurisdictional codes: {difference}."
        )
    # recode existing location abbreviations
    # with location codes
    loc_recoded_df = df.with_columns(
        pl.col(location_col).replace(
            old=loc_table["short_name"],
            new=loc_table["location_code"],
        )
    )
    return loc_recoded_df


def loc_hubverse_code_to_abbr(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    """
    Takes the location columns of a Polars
    dataframe (formatted as hubverse codes for
    US two-letter jurisdictions) and recodes
    it to US jurisdictional abbreviations,
    using location_table, which is a Polars
    dataframe contained in forecasttools.

    Parameters
    ----------
    df
        A Polars dataframe with a location
        column consisting of US
        jurisdictional hubverse codes.
    location_col
        The name of the dataframe's location
        column.

    Returns
    -------
    pl.DataFrame
        A Polars dataframe with the location
        column formatted as US two-letter
        jurisdictional abbreviations.
    """
    # check inputted variable types
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected a Polars DataFrame; got {type(df)}.")
    if not isinstance(location_col, str):
        raise TypeError(
            f"Expected a string for location_col; got {type(location_col)}."
        )
    # check if dataframe entered is empty
    if df.is_empty():
        raise ValueError(f"The dataframe {df} is empty.")
    # check if the location column exists
    # in the inputted dataframe
    if location_col not in df.columns:
        raise ValueError(
            f"Column '{location_col}' not found in the dataframe; got {df.columns}."
        )
    # get location table from forecasttools
    loc_table = forecasttools.location_table
    # check if values in location_col are a
    # subset of location_code in location table
    location_values = set(df[location_col].to_list())
    valid_values = set(loc_table["location_code"].to_list())
    difference = location_values.difference(valid_values)
    if difference:
        raise ValueError(
            f"Some values in {difference} (in col '{location_col}') are not valid jurisdictional codes."
        )
    # recode existing location codes with
    # with location abbreviations
    loc_recoded_df = df.with_columns(
        pl.col(location_col).replace(
            old=loc_table["location_code"], new=loc_table["short_name"]
        )
    )
    return loc_recoded_df


def to_location_table_column(location_format: str) -> str:
    """
    Maps a location format string to the
    corresponding column name in the hubserve
    location table. For example, "hubverse"
    maps to "location_code" in forecasttool's
    location_table.

    Parameters
    ----------
    location_format
        The format string ("abbr",
        "hubverse", or "long_name").

    Returns
    -------
    str
        Returns the corresponding column name
        from the location table.
    """
    # check inputted variable type
    assert isinstance(
        location_format, str
    ), f"Expected a string; got {type(location_format)}."
    # return proper column name from input format
    col_dict = {
        "abbr": "short_name",
        "hubverse": "location_code",
        "long_name": "long_name",
    }
    col = col_dict.get(location_format)
    if col is None:
        raise KeyError(
            f"Unknown location format {location_format}. Expected one of:\n{col_dict.keys()}."
        )
    return col


def location_lookup(
    location_vector: list[str], location_format: str
) -> pl.DataFrame:
    """
    Look up rows of the hubverse location
    table corresponding to the entries
    of a given location vector and format.
    Retrieves the rows from location_table
    in the forecasttools package
    corresponding to a given vector of
    location identifiers, with possible
    repeats.

    Parameters
    ----------
    location_vector
        A list of location values.

    location_format
        The format in which the location
        vector is coded. Permitted formats
        are: 'abbr', US two-letter
        jurisdictional abbreviation;
        'hubverse', legacy 2-digit FIPS code
        for states and territories; 'US' for
        the USA as a whole; 'long_name',
        full English name for the
        jurisdiction.

    Returns
    -------
    pl.DataFrame
        Rows from location_table that match
        the location vector, with repeats
        possible.
    """
    # check inputted variable types
    if not isinstance(location_vector, list):
        raise TypeError(f"Expected a list; got {type(location_vector)}.")
    if not all(isinstance(loc, str) for loc in location_vector):
        raise TypeError("All elements in location_vector must be of type str.")
    if not isinstance(location_format, str):
        raise TypeError(f"Expected a string; got {type(location_format)}.")
    valid_formats = ["abbr", "hubverse", "long_name"]
    if location_format not in valid_formats:
        raise ValueError(
            f"Invalid location format '{location_format}'. Expected one of: {valid_formats}."
        )
    # check that location vector not empty
    if not location_vector:
        raise ValueError("The location_vector is empty.")
    # get the join key based on the location format
    join_key = forecasttools.to_location_table_column(location_format)
    # create a dataframe for the location
    # vector with the column cast as string
    locs_df = pl.DataFrame({join_key: [str(loc) for loc in location_vector]})
    # inner join with the location_table
    # based on the join key
    locs = locs_df.join(forecasttools.location_table, on=join_key, how="inner")
    return locs
