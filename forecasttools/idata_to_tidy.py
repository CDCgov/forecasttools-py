import re

import arviz as az
import polars as pl


def convert_idata_forecast_to_tidydraws(
    idata: az.InferenceData,
    groups: list[str]
) -> dict[str, pl.DataFrame]:
    tidy_dfs = {}
    idata_df = idata.to_dataframe()
    for group in groups:
        group_columns = [
            col for col in idata_df.columns
            if isinstance(col, tuple) and col[0] == group
        ]
        meta_columns = ["chain", "draw"]
        group_df = idata_df[meta_columns + group_columns]
        group_df.columns = [
            col[1] if isinstance(col, tuple) else col
            for col in group_df.columns
        ]
        group_pols_df = pl.from_pandas(group_df)
        value_columns = [col for col in group_pols_df.columns if col not in meta_columns]
        group_pols_df = group_pols_df.melt(
            id_vars=meta_columns,
            value_vars=value_columns,
            variable_name="variable",
            value_name="value"
        )
        group_pols_df = group_pols_df.with_columns(
            pl.col("variable").map_elements(lambda x: re.sub(r"\[.*\]", "", x)).alias("variable")
        )
        group_pols_df = group_pols_df.with_columns(
            ((pl.col("draw") - 1) % group_pols_df["draw"].n_unique() + 1).alias(".iteration")
        )
        group_pols_df = group_pols_df.rename({"chain": ".chain", "draw": ".draw"})
        tidy_dfs[group] = group_pols_df.select([".chain", ".draw", ".iteration", "variable", "value"])

    return tidy_dfs
