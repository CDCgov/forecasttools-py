"""
Temporary file to convert csv files to
parquet files.
"""

# %% IMPORT LIBRARIES

import glob

import polars as pl

# %%

csv_files = glob.glob("../forecasttools/*.csv")
for csv in csv_files:
    df_from_csv = pl.read_csv(csv, ignore_errors=True)
    parquet_name = csv.replace("csv", "parquet")
    df_from_csv.write_parquet(parquet_name)


# %%
