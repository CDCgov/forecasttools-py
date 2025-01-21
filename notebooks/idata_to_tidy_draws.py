"""
The following is an experimental file for
testing that forecasttools-py can be used to
convert an Arviz InferenceData object to a
tibble that has had tidy_draws() called on it.
"""

# %% LIBRARIES USED

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
print(idata_w_dates.observed_data.dims)


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

# %% FUNCTION FOR CONVERSION

def convert_idata_forecast_to_tidydraws(
    idata: az.InferenceData,
    groups: list[str]
) -> dict[str, pl.DataFrame]:
    tidy_dfs = {}
    idata_df = idata.to_dataframe()
    for group in groups:
        group_columns = [
            col for col in idata_df.columns
            if isinstance(col, tuple) and col[0] == group
        ]
        meta_columns = ["chain", "draw"]
        group_df = idata_df[meta_columns + group_columns]
        group_df.columns = [
            col[1] if isinstance(col, tuple) else col
            for col in group_df.columns
        ]
        group_pols_df = pl.from_pandas(group_df)
        value_columns = [col for col in group_pols_df.columns if col not in meta_columns]
        group_pols_df = group_pols_df.melt(
            id_vars=meta_columns,
            value_vars=value_columns,
            variable_name="variable",
            value_name="value"
        )
        group_pols_df = group_pols_df.with_columns(
            pl.col("variable").map_elements(lambda x: re.sub(r"\[.*\]", "", x)).alias("variable")
        )
        group_pols_df = group_pols_df.with_columns(
            ((pl.col("draw") - 1) % group_pols_df["draw"].n_unique() + 1).alias(".iteration")
        )
        group_pols_df = group_pols_df.rename({"chain": ".chain", "draw": ".draw"})
        tidy_dfs[group] = group_pols_df.select([".chain", ".draw", ".iteration", "variable", "value"])

    return tidy_dfs


# These are some errors received:

# <ipython-input-26-ba2c53f5dbbf>:27: DeprecationWarning: `DataFrame.melt` is deprecated. Use `unpivot` instead, with `index` instead of `id_vars` and `on` instead of `value_vars`
#   group_pols_df = group_pols_df.melt(
# <ipython-input-26-ba2c53f5dbbf>:34: MapWithoutReturnDtypeWarning: Calling `map_elements` without specifying `return_dtype` can lead to unpredictable results. Specify `return_dtype` to silence this warning.
#   group_pols_df = group_pols_df.with_columns(
# <ipython-input-26-ba2c53f5dbbf>:27: DeprecationWarning: `DataFrame.melt` is deprecated. Use `unpivot` instead, with `index` instead of `id_vars` and `on` instead of `value_vars`
#   group_pols_df = group_pols_df.melt(
# <ipython-input-26-ba2c53f5dbbf>:34: MapWithoutReturnDtypeWarning: Calling `map_elements` without specifying `return_dtype` can lead to unpredictable results. Specify `return_dtype` to silence this warning.
#   group_pols_df = group_pols_df.with_columns(


# %% RUN CONVERSION OF TIDY COLS

groups = ["posterior", "posterior_predictive"]
tidy_draws_dict = convert_idata_forecast_to_tidydraws(
  idata_wo_dates,
  groups)

print(tidy_draws_dict)




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
# %% FUNCTION TO CONVERT IDATA GROUPS TO TIDY


def convert_idata_forecast_to_tidydraws(
    idata: az.InferenceData, groups: list[str]
) -> dict[str, pl.DataFrame]:
    tidy_dfs = {}

    for group in groups:
        # Convert the specified group to a Pandas DataFrame
        group_df = idata[group].to_dataframe()
        group_pols_df = pl.from_pandas(group_df)

        # Extract chain, draw, and group-specific variable columns
        group_columns = [col for col in group_pols_df.columns if group in col]
        meta_columns = ["chain", "draw"]
        relevant_cols = meta_columns + group_columns

        group_pols_df = group_pols_df.select(relevant_cols)

        # Clean variable column names
        def clean_column_names(cols):
            cleaned_cols = {}
            for col in cols:
                if group in col:
                    new_col = re.sub(r"[()\[\]]", "", col.split(",")[1].strip()) if "," in col else col
                else:
                    new_col = col
                cleaned_cols[col] = new_col
            return cleaned_cols

        group_pols_df = group_pols_df.rename(clean_column_names(group_pols_df.columns))

        # Create .iteration column
        group_pols_df = group_pols_df.with_columns(
            ((pl.col("draw") - 1) % group_pols_df["draw"].n_unique() + 1).alias(".iteration")
        )

        # Reorder columns: .chain, .draw, .iteration, followed by variables
        variable_columns = [col for col in group_pols_df.columns if col not in meta_columns + [".iteration"]]
        tidy_cols = [".chain", ".draw", ".iteration"] + variable_columns
        group_pols_df = group_pols_df.select(tidy_cols)

        # Store in the dictionary
        tidy_dfs[group] = group_pols_df

    return tidy_dfs
