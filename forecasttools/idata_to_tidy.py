"""
Contains functions for interfacing between
the tidy-verse and arviz, which includes
the conversion of idata objects (and hence
their groups) in tidy-usable objects.
"""

import arviz as az
import polars as pl
import polars.selectors as cs


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
        An InferenceData object.
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
                f"Requested groups {invalid_groups} not found"
                " in this InferenceData object."
                f" Available groups: {available_groups}"
            )

    idata_df = pl.DataFrame(idata.to_dataframe())

    tidy_dfs = {
        group: (
            idata_df.select("chain", "draw", cs.starts_with(f"('{group}',"))
            .rename(
                {
                    col: col.split(", ")[1].strip("')")
                    for col in idata_df.columns
                    if col.startswith(f"('{group}',")
                }
            )
            # draw in arviz is iteration in tidybayes
            .rename({"draw": ".iteration", "chain": ".chain"})
            .unpivot(
                index=[".chain", ".iteration"],
                variable_name="variable",
                value_name="value",
            )
            .with_columns(
                pl.col("variable").str.replace(r"\[.*\]", "").alias("variable")
            )
            .with_columns(pl.col(".iteration") + 1, pl.col(".chain") + 1)
            .with_columns(
                (pl.col(".iteration").n_unique()).alias("draws_per_chain"),
            )
            .with_columns(
                (
                    ((pl.col(".chain") - 1) * pl.col("draws_per_chain"))
                    + pl.col(".iteration")
                ).alias(".draw")
            )
            .pivot(
                values="value",
                index=[".chain", ".iteration", ".draw"],
                columns="variable",
                aggregate_function="first",
            )
            .sort([".chain", ".iteration", ".draw"])
            # .drop(["n_chains", "draws_per_chain"])
            # .select([".chain", ".iteration", ".draw", "variable", "value"])
        )
        for group in groups
    }
    return tidy_dfs
