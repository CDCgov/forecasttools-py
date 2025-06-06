"""
Converts daily resolution dataframes
to epiweekly dataframes.
"""

from datetime import datetime

import epiweeks
import polars as pl

from forecasttools.utils import ensure_listlike


def calculate_epi_week_and_year(date: str):
    """
    Converts an ISO8601 formatted
    date into an epiweek and epiyear.

    date
        An ISO8601 date.
    """
    epiweek = epiweeks.Week.fromdate(datetime.strptime(date, "%Y-%m-%d"))
    epiweek_df_struct = {
        "epiweek": epiweek.week,
        "epiyear": epiweek.year,
    }
    return epiweek_df_struct


def df_aggregate_to_epiweekly(
    df: pl.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    id_cols: list[str] = None,
    weekly_value_name: str = "weekly_value",
    strict: bool = True,
) -> pl.DataFrame:
    """
    Aggregate a dataframe of daily values (e.g.
    hospitalizations) to epiweekly total values
    and add epiweek and epiyear columns.

    Parameters
    ----------
    df
        Tidy data frame of daily values to aggregate.
    value_col
        The name of the column containing daily trajectory /
        timeseries values to aggregate. Defaults
        to ```"value"``.
    date_col
        The name of the column with dates.
        Defaults to ``"date"``.
    id_cols
        The name(s) of the column(s) that
        uniquely identify a single timeseries
        (e.g. a single posterior trajectory).
        Defaults to ``"draw"``.
    weekly_value_name
        The name to use for the output column
        containing weekly trajectory values.
        Defaults to ``"weekly_value"``.
    strict
        Whether to aggregate to epiweekly only
        for weeks in which all seven days have
        values. If ``False``, then incomplete weeks
        will be aggregated. Defaults to ``True``.

    Returns
    -------
    pl.DataFrame
        A dataframe with value_col aggregated
        across epiweek and epiyear.
    """
    # set default id_cols
    if id_cols is None:
        id_cols = ["draw"]
    id_cols = ensure_listlike(id_cols)
    # add epiweek and epiyear columns
    df = df.with_columns(
        pl.col(date_col)
        .map_elements(
            lambda elt: calculate_epi_week_and_year(elt),
            return_dtype=pl.Struct,
        )
        .alias("epi_struct_out")
    ).unnest("epi_struct_out")
    # group by epiweek, epiyear, and the id_cols
    group_cols = ["epiweek", "epiyear"] + id_cols
    grouped_df = df.group_by(group_cols)
    # number of elements per group
    n_elements = grouped_df.agg(pl.count().alias("n_elements"))
    problematic_trajectories = n_elements.filter(pl.col("n_elements") > 7)
    if not problematic_trajectories.is_empty():
        message = (
            f"Problematic trajectories with more than"
            f" 7 values per epiweek per year: {problematic_trajectories}"
        )
        raise ValueError(
            f"At least one trajectory has more than 7 values for a given"
            f" epiweek of a given year.\n{message}"
        )
    # check if any week has more than 7 dates
    if not n_elements["n_elements"].to_numpy().max() <= 7:
        raise ValueError(
            "At least one trajectory has more than 7 values "
            "for a given epiweek of a given epiyear.\n"
            "Problematic trajectories with more than 7 "
            "values: "
            f"{problematic_trajectories}"
        )
    # if strict, filter out groups that do not have exactly 7
    # contributing dates
    if strict:
        valid_groups = n_elements.filter(pl.col("n_elements") == 7)
        df = df.join(
            valid_groups.select(group_cols),
            on=group_cols,
            how="inner",
        )
    df = (
        df.group_by(group_cols)
        .agg(pl.col(value_col).sum().alias(weekly_value_name))
        .sort(group_cols)
    )
    return df
