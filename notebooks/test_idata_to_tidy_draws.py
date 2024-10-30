"""
Experiment file for testing
group agnostic conversion of
idata to tidy_draws.
"""

# %% LIBRARY IMPORTS

import os
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

# %% EXAMPLE IDATA W/ DATES

idata_w_dates = forecasttools.nhsn_flu_forecast_w_dates

print(idata_w_dates)

print(idata_w_dates.observed_data)

# %% PLAYING AROUND WITH TIDY_DRAWS


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


r_code_load = """
library(magrittr)


print("Create example tidy_draws dataframe.")
set.seed(213412312)
example_draws <- dplyr::tibble(
  .chain = rep(1, 100),
  .iteration = rep(1:100),
  .draw = 1:100,
  mu = stats::rnorm(100, 0, 1),
  sigma = stats::rnorm(100, 1, 0.5)
)

print("Check if the example_draws is data.frame.")
is(example_draws, "data.frame")

print("Structure of example_draws.")
dplyr::glimpse(example_draws)
"""

light_r_runner(r_code_load)

# %% OPTION 1 FOR IDATA TO TIDY DRAWS


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


# DAMON'S R CODE FOR THIS DATA (FROM IDATA CSV)

# arviz_split <- function(x) {
# x %>%
#     select(-distribution) %>%
#     split(f = as.factor(x$distribution))
# }

# pyrenew_samples <-
# read_csv(inference_data_path) %>%
# rename_with(\(varname) str_remove_all(varname, "\\(|\\)|\\'|(, \\d+)")) |>
# rename(
#     .chain = chain,
#     .iteration = draw
# ) |>
# mutate(across(c(.chain, .iteration), \(x) as.integer(x + 1))) |>
# mutate(
#     .draw = tidybayes:::draw_from_chain_and_iteration_(.chain, .iteration),
#     .after = .iteration
# ) |>
# pivot_longer(-starts_with("."),
#     names_sep = ", ",
#     names_to = c("distribution", "name")
# ) |>
# arviz_split() |>
# map(\(x) pivot_wider(x, names_from = name) |> tidy_draws())
