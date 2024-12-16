"""
Experiment file for testing group agnostic
conversion of Arviz InferenceData objects
(idata objects) to tidy_draws.
"""

# %% LIBRARIES USED

import datetime as dt
import os
import re
import subprocess
import tempfile

import arviz as az
import polars as pl
import xarray as xr

import forecasttools

xr.set_options(
    display_expand_data=False,
    display_expand_attrs=False,
)


# %% EXAMPLE IDATA W/ AND WO/ DATES

idata_w_dates = forecasttools.nhsn_flu_forecast_w_dates
idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates
print(idata_w_dates)

print(idata_w_dates.observed_data)


# %% WHEN IDATA IS CONVERTED TO DF THEN CSV

idata_wod_pandas_df = idata_wo_dates.to_dataframe()
idata_wod_pols_df = pl.from_pandas(idata_wod_pandas_df)
print(idata_wod_pols_df)

example_output = """
shape: (1_000, 1_515)
┌───────┬──────┬─────────────┬─────────────┬───┬────────────┬────────────┬────────────┬────────────┐
│ chain ┆ draw ┆ ('posterior ┆ ('posterior ┆ … ┆ ('prior_pr ┆ ('prior_pr ┆ ('prior_pr ┆ ('prior_pr │
│ ---   ┆ ---  ┆ ', 'alpha') ┆ ', 'beta_co ┆   ┆ edictive', ┆ edictive', ┆ edictive', ┆ edictive', │
│ i64   ┆ i64  ┆ ---         ┆ effs[0]'…   ┆   ┆ 'obs[97]'… ┆ 'obs[98]'… ┆ 'obs[99]'… ┆ 'obs[9]',… │
│       ┆      ┆ f32         ┆ ---         ┆   ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
│       ┆      ┆             ┆ f32         ┆   ┆ i32        ┆ i32        ┆ i32        ┆ i32        │
╞═══════╪══════╪═════════════╪═════════════╪═══╪════════════╪════════════╪════════════╪════════════╡
│ 0     ┆ 0    ┆ 20.363588   ┆ 0.334427    ┆ … ┆ 0          ┆ 13         ┆ 0          ┆ 46         │
│ 0     ┆ 1    ┆ 20.399645   ┆ 0.535402    ┆ … ┆ 0          ┆ 0          ┆ 0          ┆ 21         │
│ 0     ┆ 2    ┆ 22.719585   ┆ 0.777795    ┆ … ┆ 0          ┆ 0          ┆ 0          ┆ 5          │
│ 0     ┆ 3    ┆ 25.212839   ┆ 1.238166    ┆ … ┆ 0          ┆ 0          ┆ 0          ┆ 203        │
│ 0     ┆ 4    ┆ 24.964491   ┆ 0.912391    ┆ … ┆ 0          ┆ 1          ┆ 0          ┆ 33         │
│ …     ┆ …    ┆ …           ┆ …           ┆ … ┆ …          ┆ …          ┆ …          ┆ …          │
│ 0     ┆ 995  ┆ 23.382603   ┆ 0.688199    ┆ … ┆ 2          ┆ 2          ┆ 0          ┆ 1          │
│ 0     ┆ 996  ┆ 23.979273   ┆ 0.565145    ┆ … ┆ 0          ┆ 0          ┆ 0          ┆ 0          │
│ 0     ┆ 997  ┆ 23.99214    ┆ 0.743872    ┆ … ┆ 0          ┆ 109        ┆ 202        ┆ 1          │
│ 0     ┆ 998  ┆ 23.530113   ┆ 0.954449    ┆ … ┆ 0          ┆ 0          ┆ 0          ┆ 0          │
│ 0     ┆ 999  ┆ 22.403072   ┆ 1.035949    ┆ … ┆ 14         ┆ 2          ┆ 18         ┆ 0          │
└───────┴──────┴─────────────┴─────────────┴───┴────────────┴────────────┴────────────┴────────────┘
"""

# # doesn't immediately work with dates,
# # KeyError: Timestamp('2022-08-08 00:00:00')
# idata_pandas_df = idata_w_dates.to_dataframe()
# idata_pols_df = pl.from_pandas(idata_pandas_df)
# print(idata_pols_df)


# %% TRANSFORMATION PRIOR TO CSV CONVERSION


# %% CONVERSION OF DATAFRAME TO CSV

current_date_as_str = dt.datetime.now().date().isoformat()
save_dir = os.getcwd()
save_name = f"test_csv_idata_{current_date_as_str}.csv"
out_file = os.path.join(save_dir, save_name)
if not os.file.exists(out_file):
    idata_wod_pols_df.write_csv(out_file)


# %% FUNCTION FOR RUNNING R CODE VIA TEMPORARY FILES


def light_r_runner(r_code: str) -> None:
    """
    Run R code from Python as a temp file.
    """
    with tempfile.NamedTemporaryFile(suffix=".R", delete=False) as temp_r_file:
        temp_r_file.write(r_code.encode("utf-8"))
        temp_r_file_path = temp_r_file.name
    try:
        subprocess.run(["Rscript", temp_r_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"R script failed with error: {e}")
    finally:
        os.remove(temp_r_file_path)


# %% EXAMPLE TIDY_DRAWS (1)


# see: https://www.rdocumentation.org/packages/tidybayes/versions/3.0.7/topics/tidy_draws
r_code_example_from_docs = """
library(magrittr)

# load example dataset called line
data(line, package = "coda")
print(line)

# use tidy_draws() on line
tidy_data <- line %>%
  tidybayes::tidy_draws()
print(tidy_data)
"""

light_r_runner(r_code_example_from_docs)

# %% EXAMPLE TIDY_DRAWS (2)


r_code_spread_draws = """
# example posterior samples
posterior_samples <- dplyr::tibble(
  .chain = c(1, 1, 1, 2, 2, 2),
  .iteration = c(1, 2, 3, 1, 2, 3),
  .draw = c(1, 2, 3, 4, 5, 6),
  alpha = c(1.1, 1.3, 1.2, 1.4, 1.5, 1.6),
  beta = c(2.2, 2.3, 2.1, 2.5, 2.6, 2.4)
)

# load into tidy data
tidy_data <- tidybayes::tidy_draws(
  posterior_samples)

# examine tidy data
print(tidy_data)

# spread draws for all variables
spread_vars <- posterior_samples %>%
  tidybayes::spread_draws(alpha, beta)
dplyr::glimpse(spread_vars)
"""


# %% OPTION 1 FOR IDATA TO TIDY_DRAWS


def option_1_convert_idata_to_tidy_draws(
    idata: az.InferenceData,
) -> pl.DataFrame:
    """
    Converts ENTIRE ArviZ InferenceData object into
    a tidy_draws-compatible Polars DataFrame, ready
    for parquet export and use in R tidybayes.
    """
    # iterate over idata groups, stacking & extracting
    dfs = []
    for group_name in idata._groups_all:
        print(group_name.upper())  # DISPLAY TO USER
        group = getattr(idata, group_name, None)
        if group is not None:
            # stack non-chain, non-draw dims
            stacked_group_df = (
                group.stack(sample=("chain", "draw"), create_index=False)
                .to_dataframe()
                .reset_index()
            )
            print(stacked_group_df)  # DISPLAY TO USER
            # rename draw and chain if existing
            stacked_group_df = stacked_group_df.rename(
                columns={"draw": "draw_idx", "chain": "chain_idx"}
            )
            # add identifier (e.g. posterior) as repeated col to group
            stacked_group_df["group"] = group_name
            print(stacked_group_df)  # DISPLAY TO USER
            # extract stacked group to list for future concatenation
            dfs.append(pl.from_pandas(stacked_group_df))
    # vertically concatenate all groups
    # NOTE: failure to do this given different group column sizes
    # tidy_df = pl.concat(dfs, how="diagonal")


option_1_convert_idata_to_tidy_draws(idata_w_dates)

# %% OPTION 2 FOR IDATA TO TIDY_DRAWS


def option_2_convert_idata_to_tidy_draws(
    idata: az.InferenceData, idata_group_name: str
) -> pl.DataFrame:
    """
    Converts a specified group within an ArviZ
    InferenceData object into a tidy_draws-compatible
    Polars DataFrame. Dimensions "chain" and "draw"
    must be present.
    """
    # retrieve specified idata group
    group = getattr(idata, idata_group_name, None)
    if group is None:
        raise ValueError(
            f"Group '{idata_group_name}' not found in idata object."
        )
    # make sure required dims are present
    try:
        required_dims = {"chain", "draw"}
        missing_dims = required_dims - set(group.dims)
        assert not missing_dims, f"Missing required dimensions: {missing_dims}"
        # stack sample dimensions (chain, draw) to get a long df
        df = (
            group.stack(sample=("chain", "draw"), create_index=False)
            .to_dataframe()
            .reset_index()
        )
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        df = group.to_dataframe().reset_index()  # correct procedure?
    # convert back to Polars DataFrame, drop sample from stacking if present
    df = pl.from_pandas(df)
    if "sample" in df.columns:
        df = df.drop("sample")
    # create .chain, .iteration, and .draw columns
    df = df.with_columns(
        [
            (pl.col("chain") + 1).alias(".chain"),  # add 1 for conversion to R
            (pl.col("draw") + 1).alias(
                ".iteration"
            ),  # add 1 for conversion to R
            (
                (pl.col("chain") * df["draw"].n_unique()) + pl.col("draw") + 1
            ).alias(
                ".draw"
            ),  # add 1 for conversion to R; shift draw range to unique
        ]
    ).drop(
        ["chain", "draw"]
    )  # drop original chain, draw
    # NOTE: anything needed for cleaning for R col names?
    # pivot to long format to have variables in a single column

    # need to correct this further...
    tidy_df = df.unpivot(
        index=[".chain", ".iteration", ".draw"],
        value_name="value",  # change?
        variable_name="name",
    )
    return tidy_df


print(idata_w_dates.posterior_predictive)
postp_tidy_df = option_2_convert_idata_to_tidy_draws(
    idata=idata_w_dates, idata_group_name="posterior_predictive"
)
print(postp_tidy_df)

# %% Another attempt


# load csv
inference_data_path = "idata.csv"
df = pl.read_csv(inference_data_path)

# clean
df = df.rename({col: re.sub(r"[()'|, \d+]", "", col) for col in df.columns})

# rename and mutate cols
df = df.with_columns(
    [
        pl.col("chain").alias(".chain").cast(pl.Int32),
        pl.col("draw").alias(".iteration").cast(pl.Int32),
    ]
)

# create .draw column
df = df.with_columns((df[".chain"] * 1000 + df[".iteration"]).alias(".draw"))

# pivot longer
df = df.melt(
    id_vars=[".chain", ".iteration", ".draw"],
    variable_name="name",
    value_name="value",
)

# extract cols
df = df.with_columns(
    [
        pl.col("name")
        .str.extract(r"([^,]+), ([^,]+)", 1)
        .alias("distribution"),
        pl.col("name").str.extract(r"([^,]+), ([^,]+)", 2).alias("name"),
    ]
)

df.write_csv("tidy_draws_ready.csv")


# %% NOTES

# https://python.arviz.org/en/v0.11.4/api/generated/arviz.from_numpyro.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.to_json.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.to_dataframe.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.from_netcdf.html
# https://docs.pola.rs/api/python/stable/reference/api/polars.from_pandas.html
# https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.unpivot.html#polars.DataFrame.unpivot
