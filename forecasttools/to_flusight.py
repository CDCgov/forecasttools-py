"""
Takes epiweekly quantilized Polars dataframe
and performs final conversion to the FluSight
formatted output.
"""

from datetime import datetime, timedelta

import epiweeks
import polars as pl


def get_flusight_target_end_dates(
    reference_date: str,
    horizons: list[str] | None = None,
) -> pl.DataFrame:
    """
    Generates remaining FluSight format
    columns from a reference date for use
    in a epiweekly quantilized dataframe.

    Parameters
    ----------
    reference_date
        The target forecast week, the first
        week in (usually) a four week forecast.
    horizons
        The indices marking the forecast period,
        typically -1 to 3 (including 0) corresponding
        to one week prior to reference_date and
        three weeks after. Defaults to None.

    Returns
    -------
    pl.DataFrame
        A dataframe of columns necessary for
        the FluSight submission.
    """
    # set default horizons in case of no specification
    if horizons is None:
        horizons = list(range(-1, 4))
    # create list of ref. date, target, horizon, end date, and epidate
    reference_date_dt = datetime.strptime(
        reference_date, "%Y-%m-%d"
    )
    data_df = pl.DataFrame(
        list(
            map(
                lambda h: {
                    "reference_date": reference_date,
                    "target": "wk inc flu hosp",
                    "horizon": h,
                    "target_end_date": (
                        reference_date_dt
                        + timedelta(weeks=h)
                    ).date(),
                    "epidate": epiweeks.Week.fromdate(
                        reference_date_dt
                        + timedelta(weeks=h)
                    ),
                },
                horizons,
            )
        )
    )
    # unnest epidate column
    data_df = data_df.with_columns(
        pl.col(["epidate"]).map_elements(
            lambda elt: {
                "epiweek": elt.week,
                "epiyear": elt.year,
            },
            return_dtype=pl.Struct,
        )
    ).unnest("epidate")
    return data_df


def get_flusight_table(
    quantile_forecasts: pl.DataFrame,
    reference_date: str,
    quantile_value_col: str = "quantile_value",
    quantile_level_col: str = "quantile_level",
    location_col: str = "location",
    epiweek_col: str = "epiweek",
    epiyear_col: str = "epiyear",
    horizons=None,
    excluded_locations=None,
) -> pl.DataFrame:
    """
    Takes epiweekly quantilized Polars dataframe
    and adds target ends dates for FluSight
    formatted output dataframe.

    Parameters
    ----------
    quantile_forecasts
        A Polars dataframe of quantilized
        epiweekly forecasts for a single
        jurisdiction.
    reference_date
        The target week for which to begin
        forecasts.
    quantile_value_name
        The name of the column containing the
        outputted quantile values. Defaults to
        "quantile_value".
    quantile_level_name
        The name of the column containing the
        quantiles levels. Defaults to
        "quantile_level".
    location_col
        The name of the dataframe's location
        column. Defaults to "location".
    epiweek_col
        The name of the column containing
        epiweeks. Defaults to "epiweek".
    epiweek_col
        The name of the column containing
        epiyears. Defaults to "epiyear".
    horizons
        The indices marking the forecast period,
        typically -1 to 3 (including 0) corresponding
        to one week prior to reference_date and
        three weeks after. If None, defaults to
        list(range(-1, 4)).
    excluded_locations
        A list of US location codes to ignore
        certain locations. If None, defaults to
        ["60", "78"].

    Returns
    -------
    pl.DataFrame
        A flusight formatted dataframe.
    """
    # default horizons and locations
    if horizons is None:
        horizons = list(range(-1, 4))
    if excluded_locations is None:
        excluded_locations = ["60", "78"]
    # get target end dates
    targets = get_flusight_target_end_dates(
        reference_date, horizons=horizons
    )
    # filter and select relevant columns
    quants = quantile_forecasts.select(
        [
            pl.col(quantile_value_col).alias(
                "value"
            ),
            pl.col(location_col).alias(
                "location"
            ),
            pl.col(epiweek_col).alias("epiweek"),
            pl.col(epiyear_col).alias("epiyear"),
            pl.col(quantile_level_col).alias(
                "quantile_level"
            ),
        ]
    ).filter(
        ~pl.col("location").is_in(
            excluded_locations
        )
    )
    # inner join between targets and quantile forecasts
    output_table = targets.join(
        quants,
        on=["epiweek", "epiyear"],
        how="inner",
    )
    output_table = output_table.with_columns(
        [
            pl.lit("quantile").alias(
                "output_type"
            ),
            pl.col("quantile_level")
            .round(4)
            .alias("output_type_id"),
        ]
    )
    # final output selection and sorting
    output_table = output_table.select(
        [
            "reference_date",
            "target",
            "horizon",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ]
    ).sort(
        [
            "location",
            "reference_date",
            "target",
            "horizon",
            "target_end_date",
            "output_type",
            "output_type_id",
        ]
    )
    return output_table
