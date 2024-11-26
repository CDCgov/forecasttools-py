import importlib.resources

import arviz as az
import polars as pl

from forecasttools.daily_to_epiweekly import df_aggregate_to_epiweekly
from forecasttools.idata_w_dates_to_df import (
    add_time_coords_to_idata_dimension,
    add_time_coords_to_idata_dimensions,
    generate_time_range_for_dim,
    idata_forecast_w_dates_to_df,
)
from forecasttools.recode_locations import (
    loc_abbr_to_hubverse_code,
    loc_hubverse_code_to_abbr,
    location_lookup,
    to_location_table_column,
)
from forecasttools.to_hubverse import get_hubverse_table
from forecasttools.trajectories_to_quantiles import trajectories_to_quantiles
from forecasttools.utils import (
    ensure_listlike,
    validate_and_get_idata_group,
    validate_and_get_idata_group_var,
    validate_idata_group_var_dim,
    validate_input_type,
    validate_iter_has_expected_types,
)

# location table (from Census data)
location_table_path = importlib.resources.files(__package__).joinpath(
    "location_table.parquet"
)
location_table = pl.read_parquet(location_table_path)

# load example flusight submission
example_flusight_submission_path = importlib.resources.files(
    __package__
).joinpath("example_flusight_submission.parquet")
dtypes_d = {"location": pl.Utf8}
example_flusight_submission = pl.read_parquet(example_flusight_submission_path)

# load example fitting data for COVID
# (NHSN, as of 2024-09-26)
nhsn_hosp_COVID_path = importlib.resources.files(__package__).joinpath(
    "nhsn_hosp_COVID.parquet"
)
nhsn_hosp_COVID = pl.read_parquet(nhsn_hosp_COVID_path)

# load example fitting data for influenza
# (NHSN, as of 2024-09-26)
nhsn_hosp_flu_path = importlib.resources.files(__package__).joinpath(
    "nhsn_hosp_flu.parquet"
)
nhsn_hosp_flu = pl.read_parquet(nhsn_hosp_flu_path)

# load idata NHSN influenza forecast
# (NHSN, as of 2024-09-26) without dates
example_flu_forecast_wo_dates_path = importlib.resources.files(
    __package__
).joinpath("example_flu_forecast_wo_dates.nc")
nhsn_flu_forecast_wo_dates = az.from_netcdf(example_flu_forecast_wo_dates_path)

# load idata NHSN influenza forecast
# (NHSN, as of 2024-09-26) with dates
example_flu_forecast_w_dates_path = importlib.resources.files(
    __package__
).joinpath("example_flu_forecast_w_dates.nc")
nhsn_flu_forecast_w_dates = az.from_netcdf(example_flu_forecast_w_dates_path)


__all__ = [
    "location_table",
    "example_flusight_submission",
    "nhsn_hosp_COVID",
    "nhsn_hosp_flu",
    "nhsn_flu_forecast_wo_dates",
    "nhsn_flu_forecast_w_dates",
    "idata_forecast_w_dates_to_df",
    "add_time_coords_to_idata_dimension",
    "trajectories_to_quantiles",
    "df_aggregate_to_epiweekly",
    "loc_abbr_to_hubverse_code",
    "loc_hubverse_code_to_abbr",
    "to_location_table_column",
    "location_lookup",
    "get_hubverse_table",
    "add_time_coords_to_idata_dimension",
    "add_time_coords_to_idata_dimensions",
    "validate_input_type",
    "validate_and_get_start_time",
    "validate_and_get_idata_group",
    "validate_and_get_idata_group_var",
    "validate_idata_group_var_dim",
    "generate_time_range_for_dim",
    "validate_iter_has_expected_types",
    "ensure_listlike",
]
