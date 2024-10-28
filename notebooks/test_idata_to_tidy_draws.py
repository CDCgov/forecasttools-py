"""
Experiment file for testing
group agnostic conversion of
idata to tidy_draws.
"""

# %% LIBRARY IMPORTS

import os
import subprocess
import tempfile

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

print("Using spread_draws.")
after_spread_draws <- example_draws %>%
  tidybayes::spread_draws(mu, sigma)

print("Spread draws structure:")
dplyr::glimpse(after_spread_draws)
"""

light_r_runner(r_code_load)


# %% INTERMEDIATE REPRESENTATIONS


print(idata_w_dates.groups)

for group_name in idata_w_dates.groups():
    print(idata_w_dates[group_name])


# %% FINAL REPRESENTATION


# %% FUNCTION TO BRING IT ALL TOGETHER


# COLLEAGUES R CODE FOR THIS DATA (FROM IDATA CSV)

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
