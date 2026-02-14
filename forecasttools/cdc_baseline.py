"""
Compute the CDC Baseline forecast used by
FluSight and COVIDHub
"""

import numpy as np
import polars as pl


def get_lead_residuals(
    df: pl.LazyFrame,
    leads: int | list[int],
    timepoint_col: str = "date",
    value_col: str = "value",
    id_cols: str | list[str] = None,
    lead_name: str = "lead",
    residual_name: str = "residual",
) -> pl.DataFrame:
    """
    Compute the residuals (signed forecast errors) for a
    flateline forecast a given number of timesteps ahead.

    Parameters
    ----------
    df
        DataFrame of values to forecast.

    leads
        Timestep(s) ahead for which to compute forecast errors.

    timepoint_col
        Column containing timepoint values.

    value_col
        Column containing observed (forecast target) values.

    id_cols
        Optional id column or columns that identify distinct
        individual timeseries / forecasting problems.

    lead_name
        Name for the column in the output DataFrame
        containing the timestep ahead ("lead") values.

    residual_name
        Name for the column in the output DataFrame containing
        the values of the forecast errors (residuals).

    Return
    ------
    The dataframe for residuals for the given lead values.
    """
    if id_cols is None:
        id_cols = []

    return (
        df.select(id_cols, pl.col(timepoint_col), pl.col(value_col))
        .sort(id_cols, pl.col(timepoint_col))
        .with_columns(
            [
                (pl.col(value_col) - pl.col(value_col).shift(lead))
                .over(id_cols)
                .alias(f"{lead}")
                for lead in leads
            ]
        )
        .unpivot(
            index=id_cols + [timepoint_col, value_col],
            value_name=residual_name,
            variable_name=lead_name,
        )
        .with_columns(pl.col(lead_name).cast(pl.Int64))
        .drop_nulls(pl.col(residual_name))
    )


def symmetrize_residuals(
    df: pl.LazyFrame, residuals_col: str = "residual", sort_cols=None
) -> pl.DataFrame:
    """
    Add negative residuals alongside
    positive ones, doubling the length
    of a DataFrame.

    Parameters
    ----------
    df
        DataFrame to extend.

    residuals_col
        Column of residuals

    sort_cols
        Column(s) by which to sort the result. If ``None``,
        the result will not be sorted. Default ``None``.
    """
    if sort_cols is None:
        sort_cols = []
    return pl.concat([df, df.with_columns(-pl.col(residuals_col))]).sort(
        sort_cols
    )


def get_residual_ecdf(
    df: pl.LazyFrame,
    group_cols: list[str],
    residual_col="residual",
    fineness: int = 100000,
    residual_ecdf_name="residual_ecdf",
) -> pl.DataFrame:
    """ """
    return df.group_by(group_cols).agg(
        pl.map_groups(
            residual_col,
            lambda x: np.quantile(x, q=np.linspace(0, 1, fineness)),
        )
        .cast(pl.List(pl.Float64))
        .alias(residual_ecdf_name)
    )


def with_predictive_samples(
    df: pl.DataFrame,
    max_ahead: int,
    value_col: str = "value",
    residual_ecdf_col: str = "residual_ecdf",
):
    """
    Annotate a dataframe with predictive samples at horizons
    0 through max_ahead - 1.

    Parameters
    ----------
    df
        DataFrame to annotate.

    max_ahead
        Furthest horizon to forecast.

    value_col
        Column containing the baseline values to propagate forward
        as the median forecast.

    resid_ecdf_col
        List column containing empirical CDFs of
        (symmetrized) residuals.

    Return
    ------
    The DataFrame, with additional list columns corresponding
    to forecasts at horizons 0 through max_ahead - 1.

    Notes
    -----
    Note that non-negativity clipping may be inconsistent
    between step 1 and subsequent steps:
    https://github.com/cmu-delphi/epipredict/blob/e2f633192b07527e840836a352af342093916a9e/R/layer_cdc_flatline_quantiles.R#L240-L266
    """

    if max_ahead < 1:
        raise ValueError(
            "Must forecast at least 1 time unit ahead."
            f"Got max_ahead = {max_ahead}"
        )
    return (
        df.with_columns(
            h_0=(pl.col(value_col) + pl.col(residual_ecdf_col)).list.eval(
                pl.element().clip(lower_bound=0)
            )
        )
        .with_columns(
            [
                (
                    pl.col("h_0")
                    + pl.sum_horizontal(
                        [
                            pl.col(residual_ecdf_col).list.sample(
                                fraction=1, with_replacement=True
                            )
                            for k in range(ahead)
                        ]
                    )
                ).alias(f"h_{ahead}")
                for ahead in range(1, max_ahead)
            ]
        )
        .with_columns(
            [
                (
                    pl.col(f"h_{ahead}")
                    - pl.col(f"h_{ahead}").list.median()
                    + pl.col("value")
                ).list.eval(pl.element().clip(lower_bound=0))
                for ahead in range(1, max_ahead)
            ]
        )
    )
