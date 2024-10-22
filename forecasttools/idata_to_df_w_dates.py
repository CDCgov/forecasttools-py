"""
Contains functions for converting
Arviz InferenceData objects to Polars
dataframes with dates and draws and
for working with conversions of
Polars dataframes to Tidy dataframes.
"""

from datetime import datetime

import arviz as az
import polars as pl

# convert_idata_wo_dates_to_df_w_dates


def forecast_as_df_with_dates(
    idata_wo_dates: az.InferenceData,
    start_date: str,
    location: str,
    obs_col: str,
    timepoint_col_name: str = "date",
    value_col_name: str = "hosp",
    param: str = "obs",
    obs_param: str = "obs",
) -> pl.DataFrame:
    """
    Converts an Arviz InferenceData object
    into a Polars dataframe contains dates
    and draws for a given location. The
    number of rows in the dataframe is equal
    to the number of posterior predictive
    samples times the duration of the time
    series (end_date - start_date).

    idata_wo_dates
        A Arviz InferenceData object representing
        posterior predictive samples (a forecast)
        for a single jurisdiction.
    start_date
        The first date used for fitting in
        the time series.
    end_date
        The final date used for fitting in
        the time series.
    location
        The abbreviation for the jurisdiction
        associated with `idata_wo_dates`.
    timepoint_col_name
        The name of the timepoint column in the
        outputted dataframe. Defaults to "date".
    value_col_name
        The name of the value column in the
        outputted dataframe. Defaults to "hosp".
    param
        The InferenceData parameter to extract
        from the posterior_predictive samples
        group. Defaults to "obs".
    obs_param
        The InferenceData parameter to extract
        from the observed_data samples
        group. Defaults to "obs".
    """
    # convert ISO8601 date to datetime date
    start_date = datetime.datetime(start_date, "%Y-%m-%d")
    # stack posterior predictive samples by chain and draw
    stacked_post_pred_samples = idata_wo_dates.posterior_predictive.stack(
        sample=("chain", "draw")
    )[obs_col].to_pandas()
    # forecast dateframe wide (chain, draws) as cols
    forecast_df_wide = pl.from_pandas(stacked_post_pred_samples)
    # get the number of dates
    n_dates = forecast_df_wide.height - 1
    # create a date column
    forecast_df_wide = forecast_df_wide.with_columns(
        date=pl.date_range(
            start=start_date, end=start_date + pl.duration(days=n_dates)
        ).cast(pl.Utf8)
    )
    # unpivoted dataframe (keep draws ungrouped by uniqueness)
    forecast_df_unpivoted = forecast_df_wide.unpivot(
        index="date"
    ).with_columns(
        draw=pl.col("variable").rank("dense").cast(pl.Int64),
        location=pl.lit("location"),
    )
    return forecast_df_unpivoted
