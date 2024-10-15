import importlib.resources

import arviz as az
import polars as pl

from .daily_to_epiweekly import daily_to_epiweekly
from .idata_to_df_w_dates import forecast_as_df_with_dates
from .recode_locations import loc_abbr_to_flusight_code
from .to_flusight import get_flusight_table
from .trajectories_to_quantiles import trajectories_to_quantiles

# location table (from Census data)
with importlib.resources.path(__package__, "location_table.csv") as data_path:
    location_table = pl.read_csv(data_path)

# load example flusight submission
with importlib.resources.path(
    __package__, "example_flusight_submission.csv"
) as data_path:
    dtypes_d = {"location": pl.Utf8}
    example_flusight_submission = pl.read_csv(data_path, dtypes=dtypes_d)

# load example fitting data for COVID (NHSN, as of 2024-09-26)
with importlib.resources.path(__package__, "nhsn_hosp_COVID.csv") as data_path:
    nhsn_hosp_COVID = pl.read_csv(data_path)

# load example fitting data for influenza (NHSN, as of 2024-09-26)
with importlib.resources.path(__package__, "nhsn_hosp_flu.csv") as data_path:
    nhsn_hosp_flu = pl.read_csv(data_path)

# load light idata NHSN influenza forecast (NHSN, as of 2024-09-26)
with importlib.resources.path(
    __package__, "example_flu_forecast.nc"
) as data_path:
    nhsn_flu_forecast = az.from_netcdf(data_path)


__all__ = [
    "location_table",
    "example_flusight_submission",
    "nhsn_hosp_COVID",
    "nhsn_hosp_flu",
    "nhsn_flu_forecast",
    "forecast_as_df_with_dates",
    "trajectories_to_quantiles",
    "daily_to_epiweekly",
    "loc_abbr_to_flusight_code",
    "get_flusight_table",
]
