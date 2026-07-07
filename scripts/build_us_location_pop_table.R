#!/usr/bin/env Rscript

output_path <- "cfa/stf/forecasttools/data/us_location_pop.csv"

forecasttools::us_location_table |>
  dplyr::left_join(
    forecasttools::us_location_pop,
    by = "name"
  ) |>
  readr::write_csv(
    output_path,
    na = ""
  )
