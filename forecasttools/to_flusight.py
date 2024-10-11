from datetime import datetime, timedelta

import epiweeks
import polars as pl


def get_flusight_target_end_dates(
    reference_date: str, horizons: list[str] | None = None
) -> pl.DataFrame:
    # set default horizons in case of no specification
    if horizons is None:
        horizons = list(range(-1, 4))
    # create list of ref. date, target, horizon, end date, and epidate
    reference_date_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    data_df = pl.DataFrame(
        list(
            map(
                lambda h: {
                    "reference_date": reference_date,
                    "target": "wk inc flu hosp",
                    "horizon": h,
                    "target_end_date": (
                        reference_date_dt + timedelta(weeks=h)
                    ).date(),
                    "epidate": epiweeks.Week.fromdate(
                        reference_date_dt + timedelta(weeks=h)
                    ),
                },
                horizons,
            )
        )
    )
    # unnest epidate column
    data_df = data_df.with_columns(
        pl.col(["epidate"]).map_elements(
            lambda elt: {"epiweek": elt.week, "epiyear": elt.year},
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
    # default horizons and locations
    if horizons is None:
        horizons = list(range(-1, 4))
    if excluded_locations is None:
        excluded_locations = ["60", "78"]
    # get target end dates
    targets = get_flusight_target_end_dates(reference_date, horizons=horizons)
    # filter and select relevant columns
    quants = quantile_forecasts.select(
        [
            pl.col(quantile_value_col).alias("value"),
            pl.col(location_col).alias("location"),
            pl.col(epiweek_col).alias("epiweek"),
            pl.col(epiyear_col).alias("epiyear"),
            pl.col(quantile_level_col).alias("quantile_level"),
        ]
    ).filter(~pl.col("location").is_in(excluded_locations))
    # inner join between targets and quantile forecasts
    output_table = targets.join(quants, on=["epiweek", "epiyear"], how="inner")
    output_table = output_table.with_columns(
        [
            pl.lit("quantile").alias("output_type"),
            pl.col("quantile_level").round(4).alias("output_type_id"),
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
