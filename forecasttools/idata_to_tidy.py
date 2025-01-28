"""
Contains functions for interfacing between
the tidy-verse and arviz, which includes
the conversion of idata objects (and hence
their groups) in tidy-usable objects.
"""


import arviz as az
import polars as pl


def convert_inference_data_to_tidydraws(
    idata: az.InferenceData,
    groups: list[str]
) -> dict[str, pl.DataFrame]:
    """
    Creates a dictionary of polars dataframes
    from the groups of an arviz InferenceData
    object for use with the tidybayes API.

    Parameters
    ----------
    idata : az.InferenceData
        An InferenceData object generated
        from a numpyro forecast. Typically
        has the groups observed_data and
        posterior_predictive.
    groups : list[str]
        A list of groups belonging to the
        idata object. Defaults to all groups
        in the InferenceData.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary of groups from the idata
        convert to tidy-usable polars dataframe.
    """
    if groups is None:
        groups = list(idata.groups())
    idata_df = pl.DataFrame(idata.to_dataframe())

    tidy_dfs = {
        group: (
            idata_df
            .select(["chain", "draw"] + [col for col in idata_df.columns if isinstance(col, tuple) and col[0] == group])
            .rename({col: col[1] for col in idata_df.columns if isinstance(col, tuple) and col[0] == group})
            .melt(
                id_vars=["chain", "draw"],
                variable_name="variable",
                value_name="value"
            )
            .with_columns(
                pl.col("variable").str.replace(r"\[.*\]", "").alias("variable")
            )
            .with_columns(
                ((pl.col("draw") - 1) % idata_df.select(pl.col("draw").n_unique()).item(0) + 1).alias(".iteration")
            )
            .rename({"chain": ".chain", "draw": ".draw"})
            .select([".chain", ".draw", ".iteration", "variable", "value"])
        )
        for group in groups if any(isinstance(col, tuple) and col[0] == group for col in idata_df.columns)
    }
    return tidy_dfs
