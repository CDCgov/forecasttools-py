"""
Contains functions for interfacing between
the tidy-verse and arviz, which includes
the conversion of idata objects (and hence
their groups) in tidy-usable objects.
"""

import re

import arviz as az
import polars as pl


def convert_idata_forecast_to_tidydraws(
    idata: az.InferenceData,
    groups: list[str]
) -> dict[str, pl.DataFrame]:
    """
    Creates a dictionary of polars dataframes
    from the groups of an arviz InferenceData
    object that when converted to a csv(s)
    and read in R is tidy-usable.

    Parameters
    ----------
    idata : az.InferenceData
        An InferenceData object generated
        from a numpyro forecast. Typically
        has the groups observed_data and
        posterior_predictive.
    groups : list[str]
        A list of groups belonging to the
        idata object.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary of groups from the idata
        convert to tidy-usable polars dataframe.
    """
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
