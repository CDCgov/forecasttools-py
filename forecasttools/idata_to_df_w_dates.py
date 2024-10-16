"""
Contains functions for converting
Arviz InferenceData objects to Polars
dataframes with dates and draws and
for working with conversions of
Polars dataframes to Tidy dataframes.
"""

from datetime import datetime, timedelta

import arviz as az
import numpy as np
import polars as pl


def forecast_as_df_with_dates(
    idata_wo_dates: az.InferenceData,
    start_date: str,
    end_date: str,
    location: str,
    timepoint_col_name: str = "date",
    value_col_name: str = "hosp",
    param: str = "obs",
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
        from the posterior samples. Defaults to
        "obs".
    """

    # extract number of posterior predictive samples
    stacked_post_pred_samples = idata_wo_dates.posterior_predictive.stack(
        sample=("chain", "draw")
    )[param].values
    num_timesteps, num_samples = stacked_post_pred_samples.shape
    # get number of days of forecast
    num_observed_days = idata_wo_dates.observed_data["obs"].shape[0]
    forecast_days = num_timesteps - num_observed_days
    # generate dates corresponding to the forecast
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    forecast_dates = [
        (end_date_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, forecast_days + 1)
    ]
    dates_repeated = np.tile(forecast_dates, num_samples)
    # generate draw indices repeated for each forecast date
    draw_indices = np.repeat(np.arange(1, num_samples + 1), forecast_days)
    # extract forecast from posterior predictive samples
    forecast_values = stacked_post_pred_samples[
        num_observed_days:, :
    ].flatten()
    # create forecast dataframe
    forecast_df = pl.DataFrame(
        {
            ".draw": draw_indices,
            timepoint_col_name: dates_repeated,
            value_col_name: forecast_values,
            "location": [location] * len(draw_indices),
        }
    )
    return forecast_df
