import polars as pl

import forecasttools


def loc_abbr_to_flusight_code(
    df: pl.DataFrame, location_col: str
) -> pl.DataFrame:
    # get location table
    loc_table = forecasttools.location_table
    # map abbr name to codes
    mapping_dict = dict(
        zip(loc_table["short_name"], loc_table["location_code"])
    )
    # replace column values with codes
    df = df.with_columns(
        [
            pl.col(location_col)
            .map_elements(lambda x: mapping_dict.get(x, None))
            .alias(location_col)
        ]
    )
    return df
