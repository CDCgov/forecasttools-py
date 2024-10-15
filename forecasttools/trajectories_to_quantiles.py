"""
Take value column of a dataframe with
epiweekly draws and converts to quantiles
over specified quantile levels.
"""

import numpy as np
import polars as pl


def trajectories_to_quantiles(
    trajectories: pl.DataFrame,
    quantiles: list[float] = None,
    timepoint_cols: list[str] = ["timepoint"],
    value_col: str = "value",
    quantile_value_name: str = "quantile_value",
    quantile_level_name: str = "quantile_level",
    id_cols: list[str] = None,
) -> pl.DataFrame:
    """
    Converts the value columns of a polars
    dataframe of epiweekly draws (trajectories)
    to quantile values over specified quantile
    levels.

    trajectories
        A polars dataframe of epiweekly draws for
        a single jurisdiction.
    quantiles
        A list of quantiles for the output values.
        Defaults to None.
    timepoint_cols
        A list of dataframe columns that
        identifies unique timepoints. Defaults
        to ["timepoint"].
    value_col
        The name of the column with the
        trajectory values. For example, "hosp",
        "cases", "values". Defaults to "value".
    quantile_value_name
        The name of the column containing the
        outputted quantile values. Defaults to
        "quantile_value".
    quantile_level_name
        The name of the column containing the
        quantiles levels. Defaults to
        "quantile_level".
    id_cols
        Dataframe columns to aggregate before
        the value_col is mapped to quantile
        levels. For example, if there are multiple
        locations present, one might group by
        ["location"]. Defaults to None.
    """
    # set default quantiles
    if quantiles is None:
        quantiles = (
            [0.01, 0.025]
            + [0.05 * elt for elt in range(1, 20)]
            + [0.975, 0.99]
        )

    # group trajectories based on timepoint_cols and id_cols
    group_cols = (
        timepoint_cols if id_cols is None else timepoint_cols + id_cols
    )
    # get quantiles across epiweek for forecast
    quantile_results = trajectories.group_by(group_cols).agg(
        [
            pl.col(value_col)
            .map_elements(
                lambda vals: np.quantile(vals, quantiles),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("quantile_values")
        ]
    )
    # resultant quantiles into individual rows
    quant_df = quantile_results.explode("quantile_values")
    # aligning rows with quantile levels
    quant_df = quant_df.with_columns(
        [
            pl.Series(quantiles * (len(quant_df) // len(quantiles))).alias(
                "quantile_levels"
            )
        ]
    )
    # renaming quantile columns
    quant_df = quant_df.rename(
        {
            "quantile_values": quantile_value_name,
            "quantile_levels": quantile_level_name,
        }
    )
    return quant_df.sort(["epiyear", "epiweek", "quantile_level"])
