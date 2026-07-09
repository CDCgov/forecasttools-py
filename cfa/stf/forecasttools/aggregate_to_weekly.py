import datetime
from typing import Literal

import epiweeks
import polars as pl

WeekStandard = Literal["epiweek", "isoweek"]


def calculate_week_and_year(
    date_input: datetime.date,
    standard: WeekStandard = "epiweek",
) -> dict[str, int]:
    """Convert a date to week and weekyear for the requested weekly standard."""
    if not isinstance(date_input, datetime.date):
        raise TypeError(
            f"date_input must be a datetime.date, got {type(date_input).__name__}"
        )
    if standard not in ["epiweek", "isoweek"]:
        raise ValueError("standard must be one of {'epiweek', 'isoweek'}")

    if standard == "epiweek":
        epiweek = epiweeks.Week.fromdate(date_input)
        return {
            "week": epiweek.week,
            "weekyear": epiweek.year,
        }

    isocal = date_input.isocalendar()
    return {
        "week": isocal.week,
        "weekyear": isocal.year,
    }


def calculate_week_startdate(
    year: int,
    week: int,
    standard: WeekStandard = "epiweek",
) -> datetime.date:
    """Given a week and year, return the week start date."""
    if standard not in ["epiweek", "isoweek"]:
        raise ValueError("standard must be one of {'epiweek', 'isoweek'}")
    if standard == "epiweek":
        return epiweeks.Week(year, week).startdate()

    return datetime.date.fromisocalendar(year, week, 1)


def calculate_week_enddate(
    year: int,
    week: int,
    standard: WeekStandard = "epiweek",
) -> datetime.date:
    """Given a week and year, return the week end date."""
    if standard not in ["epiweek", "isoweek"]:
        raise ValueError("standard must be one of {'epiweek', 'isoweek'}")
    if standard == "epiweek":
        return epiweeks.Week(year, week).enddate()

    return datetime.date.fromisocalendar(year, week, 7)


def daily_to_weekly(
    df: pl.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    id_cols: str | list[str] = ".draw",
    weekly_value_name: str = "weekly_value",
    standard: WeekStandard = "epiweek",
    with_week_start_date: bool = False,
    with_week_end_date: bool = False,
    week_start_date_name: str = "week_start_date",
    week_end_date_name: str = "week_end_date",
    strict: bool = True,
) -> pl.DataFrame:
    """
    Aggregate a dataframe of daily values to weekly totals.

    Parameters
    ----------
    df
        Tidy data frame of daily values to aggregate.
    value_col
        Name of the column containing daily values. Defaults to ``"value"``.
    date_col
        Name of the date column. Defaults to ``"date"``.
    id_cols
        Column name(s) that uniquely identify a single timeseries.
    weekly_value_name
        Name for the output weekly value column.
        Defaults to ``"weekly_value"``.
    standard
        Weekly standard used to compute week and weekyear.
        Must be one of ``"epiweek"`` or ``"isoweek"``.
        Defaults to ``"epiweek"``.
    with_week_start_date
        Whether to annotate output with week start date.
    with_week_end_date
        Whether to annotate output with week end date.
    week_start_date_name
        Name for the output week start-date column.
        Defaults to ``"week_start_date"``.
    week_end_date_name
        Name for the output week end-date column.
        Defaults to ``"week_end_date"``.
    strict
        If ``True``, only aggregate weeks with exactly seven dates.

    Returns
    -------
    pl.DataFrame
        Dataframe with weekly aggregated values and optionally week metadata.
    """
    if value_col not in df.columns:
        raise ValueError(
            f"Specified value column '{value_col}' is missing from the dataframe"
        )
    if date_col not in df.columns:
        raise ValueError(
            f"Specified date column '{date_col}' is missing from the dataframe"
        )
    if standard not in ["epiweek", "isoweek"]:
        raise ValueError("standard must be one of {'epiweek', 'isoweek'}")

    if isinstance(id_cols, str):
        id_cols = [id_cols]
    else:
        id_cols = list(id_cols)
    missing_id_cols = [col for col in id_cols if col not in df.columns]
    if missing_id_cols:
        raise ValueError(
            f"Specified trajectory id column(s) {missing_id_cols} are missing from the dataframe"
        )

    df = df.with_columns(
        pl.col(date_col)
        .map_elements(
            lambda elt: calculate_week_and_year(elt, standard=standard),
            return_dtype=pl.Struct(
                [pl.Field("week", pl.Int64), pl.Field("weekyear", pl.Int64)]
            ),
        )
        .alias("week_struct")
    ).unnest("week_struct")

    group_cols = ["week", "weekyear"] + id_cols
    n_elements = df.group_by(group_cols).agg(pl.len().alias("n_elements"))
    problematic_trajectories = n_elements.filter(pl.col("n_elements") > 7)
    if not problematic_trajectories.is_empty():
        message = (
            "At least one trajectory has more than 7 values for a given "
            f"week of a given weekyear.\n"
            f"Problematic trajectories with more than 7 values: {problematic_trajectories}"
        )
        raise ValueError(message)

    if strict:
        valid_groups = n_elements.filter(pl.col("n_elements") == 7)
        df = df.join(valid_groups.select(group_cols), on=group_cols, how="inner")

    df = (
        df.group_by(group_cols)
        .agg(pl.col(value_col).sum().alias(weekly_value_name))
        .sort(group_cols)
    )

    if with_week_start_date:
        df = df.with_columns(
            pl.struct(["weekyear", "week"])
            .map_elements(
                lambda elt: calculate_week_startdate(
                    year=elt["weekyear"],
                    week=elt["week"],
                    standard=standard,
                ),
                return_dtype=pl.Date,
            )
            .alias(week_start_date_name)
        )

    if with_week_end_date:
        df = df.with_columns(
            pl.struct(["weekyear", "week"])
            .map_elements(
                lambda elt: calculate_week_enddate(
                    year=elt["weekyear"],
                    week=elt["week"],
                    standard=standard,
                ),
                return_dtype=pl.Date,
            )
            .alias(week_end_date_name)
        )

    return df
