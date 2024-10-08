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

    # compute quantiles
    def compute_quantiles(values):
        quantile_values = np.quantile(values, quantiles)
        return quantile_values

    # get quantiles across epiweek for forecast
    quantile_results = trajectories.group_by(group_cols).agg(
        [
            pl.col(value_col)
            .map_elements(
                lambda group_values: compute_quantiles(group_values),
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
