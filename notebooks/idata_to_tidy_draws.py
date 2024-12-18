"""
The following is an experimental file for
testing that forecasttools-py can be used to
convert an Arviz InferenceData object to a
tibble that has had tidy_draws() called on it.
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
# see: https://www.rdocumentation.org/packages/tidybayes/versions/3.0.7/topics/tidy_draws


# output saved below as a docstring
light_r_runner(r_code_example_from_docs)
example_output_line = """
$line1
       alpha       beta     sigma
1   7.173130 -1.5662000 11.233100
2   2.952530  1.5033700  4.886490
3   3.669890  0.6281570  1.397340
4   3.315220  1.1827200  0.662879
5   3.705440  0.4904370  1.362130
6   3.579100  0.2069700  1.043500
7   2.702060  0.8825530  1.290430
8   2.961360  1.0851500  0.459322
9   3.534060  1.0692600  0.634257
10  2.094710  1.4807700  0.912919
11  3.065240  0.3783490  1.188500
12  2.739010  0.4447120  0.576963
13  3.089290 -0.0130600  2.400330
14  2.704130  1.1558500  1.794230
15  2.521740  0.8563680  0.943078
16  3.732160  0.6574610  0.903465
17  2.673390  0.5944110  0.731041
18  3.017790  0.8877700  0.622143
19  2.701230  0.7459180  0.790993
20  2.978370  0.9936190  0.740969
21  2.876460  1.0280600  0.582580
22  3.191340  0.5536110  0.710707
23  2.797080  1.0188600  0.870071
...
[1] "mcmc"

attr(,"class")
[1] "mcmc.list"
"""
example_out_tidy_data = """
# A tibble: 400 × 6
   .chain .iteration .draw alpha   beta  sigma
    <int>      <int> <int> <dbl>  <dbl>  <dbl>
 1      1          1     1  7.17 -1.57  11.2
 2      1          2     2  2.95  1.50   4.89
 3      1          3     3  3.67  0.628  1.40
 4      1          4     4  3.32  1.18   0.663
 5      1          5     5  3.71  0.490  1.36
 6      1          6     6  3.58  0.207  1.04
 7      1          7     7  2.70  0.883  1.29
 8      1          8     8  2.96  1.09   0.459
 9      1          9     9  3.53  1.07   0.634
10      1         10    10  2.09  1.48   0.913
# ℹ 390 more rows
"""


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

print(
    sorted(idata_wod_pols_df.columns)
)  # e.g. "('posterior_predictive', 'obs[138]', 138)"

# # doesn't immediately work with dates,
# # KeyError: Timestamp('2022-08-08 00:00:00')
# idata_pandas_df = idata_w_dates.to_dataframe()
# idata_pols_df = pl.from_pandas(idata_pandas_df)
# print(idata_pols_df)

# %% TRANSFORMATION TO TIDY DATA


def save_tidydraws_polars_forecast_df(tidydraws_like_df: pl.DataFrame):
    # adds dots to the column name
    # save the file as a csv in the specified
    # location
    pass


def convert_idata_forecast_to_tidydraws(
    idata_wo_dates: az.InferenceData,
) -> pl.DataFrame:
    """
    Takes an Arviz InferenceData object
    containing a forecast (i.e. has posterior
    predictive column) and converts it to a
    polars dataframe that resembles an MCMC
    data object in R once
    `tidybayes::tidy_draws()` has been called
    on it, i.e. the object contains the posterior
    predictive column along with the columns
    .chain .iteration .draw.
    """
    # convert idata to pandas then polars df
    idata_wod_pandas_df = idata_wo_dates.to_dataframe()
    idata_wod_pols_df = pl.from_pandas(idata_wod_pandas_df)
    # extract and rename relevant columns (
    # i.e. those for chain, draw, and
    # posterior predictive, without punctuation)
    acceptable_cols = ["chain", "draw", "iteration", "posterior_predictive"]
    relevant_cols = [
        col
        for col in idata_wod_pols_df.columns
        if any(accept in col for accept in acceptable_cols)
    ]
    idata_wod_pols_df = idata_wod_pols_df.select(sorted(relevant_cols))

    def clean_column_names(cols):
        cleaned_cols = {}
        for col in cols:
            if "posterior_predictive" in col:
                # split by comma, take the
                # second part and strip
                # extra whitespace
                new_col = re.sub(r"[()\[\]]", "", col.split(",")[1].strip())
            else:
                # keep col (chain, iter)
                new_col = col
            cleaned_cols[col] = new_col
        return cleaned_cols

    idata_wod_pols_df = idata_wod_pols_df.rename(
        clean_column_names(relevant_cols)
    )
    # create iteration column from draw and
    # chain columns
    num_iterations_per_chain = idata_wod_pols_df.shape[0]
    idata_wod_pols_df = idata_wod_pols_df.with_columns(
        ((pl.col("draw") - 1) % num_iterations_per_chain + 1).alias(
            "iteration"
        )
    )
    # unpivot along value col
    idata_wod_pols_df = idata_wod_pols_df.unpivot(
        index=idata_wod_pols_df.columns,  # exclude non-posterior columns
        variable_name="distribution_name",
        value_name="value",
    )

    # df_out = idata_wod_pols_df.unpivot(
    #     on=pl.col("*").exclude(["chain", "draw"]),
    #     variable_name="iteration",
    #     value_name="obs"
    # )
    # df_out = df_out.with_columns(
    #     pl.col("iteration")
    #     .str.extract(r"obs(\d+)", 1)
    #     .cast(pl.Int64)
    #     .alias("iteration")
    # )

    # df_out = idata_wod_pols_df.unpivot(
    #     on=pl.col("*").exclude(["chain", "iteration", "draw"]),
    #     variable_name="observation",
    #     value_name="value"
    # )

    return idata_wod_pols_df


print(convert_idata_forecast_to_tidydraws(idata_wo_dates))


# %% CONVERSION OF DATAFRAME TO CSV

current_date_as_str = dt.datetime.now().date().isoformat()
save_dir = os.getcwd()
save_name = f"test_csv_idata_{current_date_as_str}.csv"
out_file = os.path.join(save_dir, save_name)
if not os.file.exists(out_file):
    idata_wod_pols_df.write_csv(out_file)


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


# # load csv
# inference_data_path = "idata.csv"
# df = pl.read_csv(inference_data_path)

# # clean
# df = df.rename({col: re.sub(r"[()'|, \d+]", "", col) for col in df.columns})

# # rename and mutate cols
# df = df.with_columns(
#     [
#         pl.col("chain").alias(".chain").cast(pl.Int32),
#         pl.col("draw").alias(".iteration").cast(pl.Int32),
#     ]
# )

# # create .draw column
# df = df.with_columns((df[".chain"] * 1000 + df[".iteration"]).alias(".draw"))

# # pivot longer, TODO: use unpivot again, get
# # out of melt mentality
# df = df.melt(
#     id_vars=[".chain", ".iteration", ".draw"],
#     variable_name="name",
#     value_name="value",
# )

# # extract cols
# df = df.with_columns(
#     [
#         pl.col("name")
#         .str.extract(r"([^,]+), ([^,]+)", 1)
#         .alias("distribution"),
#         pl.col("name").str.extract(r"([^,]+), ([^,]+)", 2).alias("name"),
#     ]
# )

# df.write_csv("tidy_draws_ready.csv")


# %% NOTES

# https://python.arviz.org/en/v0.11.4/api/generated/arviz.from_numpyro.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.to_json.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.to_dataframe.html
# https://python.arviz.org/en/v0.11.4/api/generated/arviz.InferenceData.from_netcdf.html
# https://docs.pola.rs/api/python/stable/reference/api/polars.from_pandas.html
# https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.unpivot.html#polars.DataFrame.unpivot

# Reminders
# Chain: one MCMC run used to sample from posterior distribution.
# Draw: single parameter sample from posterior distribution at a specific iteration of the MCMC process.
# Iteration: single step in MCMC run; each iter corresponds to single draw from the chain at that step.
