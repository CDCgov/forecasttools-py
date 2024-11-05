import importlib.resources

import arviz as az
import polars as pl

from forecasttools.daily_to_epiweekly import df_aggregate_to_epiweekly
from forecasttools.idata_w_dates_to_df import (
    add_dates_as_coords_to_idata,
    idata_forecast_w_dates_to_df,
)
from forecasttools.recode_locations import (
    loc_abbr_to_flusight_code,
    loc_flusight_code_to_abbr,
    location_lookup,
    to_location_table_column,
)
from forecasttools.to_flusight import get_flusight_table
from forecasttools.trajectories_to_quantiles import trajectories_to_quantiles

# location table (from Census data)
with importlib.resources.path(
    __package__, "location_table.parquet"
) as data_path:
    location_table = pl.read_parquet(data_path)

# load example flusight submission
with importlib.resources.path(
    __package__,
    "example_flusight_submission.parquet",
) as data_path:
    example_flusight_submission = pl.read_parquet(data_path)

# load example fitting data for COVID (NHSN, as of 2024-09-26)
with importlib.resources.path(
    __package__, "nhsn_hosp_COVID.parquet"
) as data_path:
    nhsn_hosp_COVID = pl.read_parquet(data_path)

# load example fitting data for influenza (NHSN, as of 2024-09-26)
with importlib.resources.path(
    __package__, "nhsn_hosp_flu.parquet"
) as data_path:
    nhsn_hosp_flu = pl.read_parquet(data_path)

# load light idata NHSN influenza forecast wo dates (NHSN, as of 2024-09-26)
with importlib.resources.path(
    __package__,
    "example_flu_forecast_wo_dates.nc",
) as data_path:
    nhsn_flu_forecast_wo_dates = az.from_netcdf(data_path)

# load light idata NHSN influenza forecast w dates (NHSN, as of 2024-09-26)
with importlib.resources.path(
    __package__, "example_flu_forecast_w_dates.nc"
) as data_path:
    nhsn_flu_forecast_w_dates = az.from_netcdf(data_path)


__all__ = [
    "location_table",
    "example_flusight_submission",
    "nhsn_hosp_COVID",
    "nhsn_hosp_flu",
    "nhsn_flu_forecast_wo_dates",
    "nhsn_flu_forecast_w_dates",
    "idata_forecast_w_dates_to_df",
    "add_dates_as_coords_to_idata",
    "trajectories_to_quantiles",
    "df_aggregate_to_epiweekly",
    "loc_abbr_to_flusight_code",
    "loc_flusight_code_to_abbr",
    "to_location_table_column",
    "location_lookup",
    "get_flusight_table",
]
