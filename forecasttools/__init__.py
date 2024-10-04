import importlib.resources

import polars as pl

# location table (from Census data)
with importlib.resources.path(__package__, "location_table.csv") as data_path:
    location_table = pl.read_csv(data_path)

# load example fitting data for COVID (NHSN, as of 2024-09-26)
with importlib.resources.path(__package__, "nhsn_hosp_COVID.csv") as data_path:
    nhsn_hosp_COVID = pl.read_csv(data_path)

# load example fitting data for influenza (NHSN, as of 2024-09-26)
with importlib.resources.path(__package__, "nhsn_hosp_flu.csv") as data_path:
    nhsn_hosp_flu = pl.read_csv(data_path)

__all__ = ["location_table", "nhsn_hosp_COVID", "nhsn_hosp_flu"]
