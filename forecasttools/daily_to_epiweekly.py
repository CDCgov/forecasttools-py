from datetime import datetime

import epiweeks
import polars as pl


def calculate_epidate(date):
    epiweek = epiweeks.Week.fromdate(date)
    return epiweek.week, epiweek.year


def daily_to_epiweekly(
    forecast_df: pl.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    id_cols: list[str] = [".draw"],
    weekly_value_name: str = "weekly_value",
    strict: bool = False,
) -> pl.DataFrame:
    """
    Aggregate daily forecast draws to epiweekly.
    """
    # check intended df columns are in received df
    # forecast_df_cols = forecast_df.columns
    # = [value_col, date_col] + id_cols
    # assert set(required_cols).issubset(set(forecast_df_cols)),f"Column mismatch between require columns {required_cols} and forecast dateframe columns {forecast_df_cols}."
    # add epiweek and epiyear columns
    forecast_df = forecast_df.with_columns(
        [
            pl.col(date_col)
            .map_elements(
                lambda x: calculate_epidate(datetime.strptime(x, "%Y-%m-%d"))[
                    0
                ]
            )
            .alias("epiweek"),
            pl.col(date_col)
            .map_elements(
                lambda x: calculate_epidate(datetime.strptime(x, "%Y-%m-%d"))[
                    1
                ]
            )
            .alias("epiyear"),
        ]
    )
    # group by epiweek, epiyear, and the id_cols
    group_cols = ["epiweek", "epiyear"] + id_cols
    grouped_df = forecast_df.group_by(group_cols)
    # number of elements per group
    n_elements = grouped_df.agg(pl.count().alias("n_elements"))
    problematic_trajectories = n_elements.filter(pl.col("n_elements") > 7)
    if not problematic_trajectories.is_empty():
        message = f"Problematic trajectories with more than 7 values per epiweek per year: {problematic_trajectories}"
        raise ValueError(
            f"At least one trajectory has more than 7 values for a given epiweek of a given year.\n{message}"
        )
    # check if any week has more than 7 dates
    if not n_elements["n_elements"].to_numpy().max() <= 7:
        raise ValueError(
            "At least one trajectory has more than 7 values for a given epiweek of a given year."
        )
    # if strict, filter out groups that do not have exactly 7 contributing dates
    if strict:
        valid_groups = n_elements.filter(pl.col("n_elements") == 7)
        forecast_df = forecast_df.join(
            valid_groups.select(group_cols), on=group_cols
        )
    # aggregate; sum values in the specified value_col
    df = (
        forecast_df.group_by(group_cols)
        .agg(pl.col(value_col).sum().alias(weekly_value_name))
        .sort(["epiyear", "epiweek", ".draw"])
    )
    return df
