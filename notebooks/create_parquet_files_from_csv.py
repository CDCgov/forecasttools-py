"""
Temporary file to convert csv files to
parquet files.
"""

# %% IMPORT LIBRARIES

import glob
import os
import subprocess

import polars as pl

import forecasttools

# %% CREATE PARQUET FILES

csv_files = glob.glob("../forecasttools/*.csv")
for csv in csv_files:
    dtypes_d = {"location": pl.Utf8}
    df_from_csv = pl.read_csv(csv, schema_overrides=dtypes_d)
    parquet_name = csv.replace("csv", "parquet")
    if not os.path.exists(parquet_name):
        df_from_csv.write_parquet(parquet_name)
        print(f"New file created:\n{parquet_name}")


# %% TEST FILES

forecasttools.example_flusight_submission
forecasttools.location_table
forecasttools.nhsn_hosp_flu
forecasttools.nhsn_hosp_COVID

# %% TEST ALL GOOD

for csv in csv_files:
    subprocess.run(["rm", csv])

# %%
