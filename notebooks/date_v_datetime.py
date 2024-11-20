"""
Quick file for examining polars datetime
and date ranges with different start dates
and time steps.
"""

# %% IMPORTS

from datetime import date, datetime, timedelta

import polars as pl

# %% SETUP FOR DATE / TIME RANGE


start_date_as_dt_01 = datetime(2024, 12, 12, 14, 30, 0)

start_date_as_dt_02 = date(2024, 12, 12)

interval_size = 40

time_step = timedelta(days=1)

# %% DATE / TIME RANGES

out_01 = pl.datetime_range(
    start=start_date_as_dt_01,
    end=start_date_as_dt_01 + (interval_size - 1) * time_step,
    interval=f"{time_step.days}d",
    closed="both",
    eager=True,
).to_list()

print(out_01, type(out_01[0]))

out_02 = pl.date_range(
    start=start_date_as_dt_02,
    end=start_date_as_dt_02 + (interval_size - 1) * time_step,
    interval=f"{time_step.days}d",
    closed="both",
    eager=True,
).to_list()

print(
    out_02,
    type(out_02[0]),
    isinstance(out_02[0], date),
    isinstance(out_02[0], datetime),
)
# %%
