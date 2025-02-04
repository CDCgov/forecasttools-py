"""
Contains functions for interfacing between
the tidy-verse and arviz, which includes
the conversion of idata objects (and hence
their groups) in tidy-usable objects.
"""

import arviz as az
import polars as pl


def convert_inference_data_to_tidydraws(
    idata: az.InferenceData, groups: list[str]
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
        A list of groups to transform to
        tidy draws format. Defaults to all
        groups in the InferenceData.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary of groups from the idata
        for use with the tidybayes API.
    """
    available_groups = list(idata.groups())
    if groups is None:
        groups = available_groups
    else:
        invalid_groups = [
            group for group in groups if group not in available_groups
        ]
        if invalid_groups:
            raise ValueError(
                f"Invalid groups provided: {invalid_groups}."
                f" Available groups: {available_groups}"
            )

    idata_df = pl.DataFrame(idata.to_dataframe())
    tidy_dfs = {
        group: (
            idata_df.select(
                ["chain", "draw"]
                + [
                    col
                    for col in idata_df.columns
                    if col.startswith(f"('{group}',")
                ]
            )
            .rename(
                {
                    col: col.split(", ")[1].strip("')")
                    for col in idata_df.columns
                    if col.startswith(f"('{group}',")
                }
            )
            .melt(
                id_vars=["chain", "draw"],
                variable_name="variable",
                value_name="value",
            )
            .with_columns(
                pl.col("variable").str.replace(r"\[.*\]", "").alias("variable")
            )
            .with_columns(
                ((pl.col("draw") - 1) % pl.col("draw").n_unique() + 1).alias(
                    ".iteration"
                )
            )
            .rename({"chain": ".chain", "draw": ".draw"})
            .select([".chain", ".iteration", ".draw", "variable", "value"])
        )
        for group in groups
    }
    return tidy_dfs
