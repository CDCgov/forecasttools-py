"""
Retrieves US 2020 Census data,
hospitalization count data, and
example forecast data.
"""

import os

import polars as pl


def get_census_data(
    save_path: str = os.getcwd(),
    save_as_csv: bool = False,
    url: str = "https://www2.census.gov/geo/docs/reference/state.txt",
):
    """
    Retrieves US 2020 Census data in a
    three column polars dataframe.
    """
    nation = pl.DataFrame(
        {
            "location_code": ["US"],
            "short_name": ["US"],
            "long_name": ["United States"],
        }
    )
    jurisdictions = pl.read_csv(url, separator="|").select(
        [
            pl.col("STATE").alias("location_code").cast(pl.Utf8),
            pl.col("STUSAB").alias("short_name"),
            pl.col("STATE_NAME").alias("long_name"),
        ]
    )
    flusight_location_table = nation.vstack(jurisdictions)
    if save_as_csv:
        flusight_location_table.write_csv("flusight_location_table.csv")
    return flusight_location_table
