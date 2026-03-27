"""
Converts daily resolution dataframes
to epiweekly dataframes.
"""

import datetime

import epiweeks
import polars as pl

from forecasttools.utils import ensure_listlike


def calculate_epi_week_and_year(date_input: datetime.date):
    """
    Converts a given date into an
    epiweek and epiyear.

    date_input
        A date as ``datetime.date``
        (including values from ``pl.Date`` columns).
    """
    if not isinstance(date_input, datetime.date):
        raise TypeError("date input must be datetime.date")

    epiweek = epiweeks.Week.fromdate(date_input)
    epiweek_df_struct = {
        "epiweek": epiweek.week,
        "epiyear": epiweek.year,
    }
    return epiweek_df_struct


def calculate_epiweek_enddate(epiyear: int, epiweek: int) -> datetime.date:
    """
    Given an epiweek and epiyear, return
    the enddate (Saturday) of that epiweek.

    epiyear
        Epidemiological year.
    epiweek
        Epidemiological week number.
    """
    return epiweeks.Week(epiyear, epiweek).enddate()


def df_aggregate_to_epiweekly(
    df: pl.DataFrame,
    value_col: str = "value",
    date_col: datetime.date = "date",
    id_cols: list[str] = None,
    weekly_value_name: str = "weekly_value",
    with_epiweek_end_date: bool = False,
    epiweek_end_date_name: str = "epiweek_end_date",
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
    with_epiweek_end_date
        Whether to annotate output with the last date
        of each epiweek. Defaults to ``False``.
    epiweek_end_date_name
        Name for the output epiweek end-date column.
        Defaults to ``"epiweek_end_date"``.
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
            return_dtype=pl.Struct(
                [pl.Field("epiweek", pl.Int64), pl.Field("epiyear", pl.Int64)]
            ),
        )
        .alias("epi_struct_out")
    ).unnest("epi_struct_out")
    # group by epiweek, epiyear, and the id_cols
    group_cols = ["epiweek", "epiyear"] + id_cols
    grouped_df = df.group_by(group_cols)
    # number of elements per group
    n_elements = grouped_df.agg(pl.len().alias("n_elements"))
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

    if with_epiweek_end_date:
        df = df.with_columns(
            pl.struct(["epiyear", "epiweek"])
            .map_elements(
                lambda elt: calculate_epiweek_enddate(
                    epiyear=elt["epiyear"],
                    epiweek=elt["epiweek"],
                ),
                return_dtype=pl.Date,
            )
            .alias(epiweek_end_date_name)
        )

    return df
