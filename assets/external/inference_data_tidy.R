format_split_text <- function(x, concat_char = "|") {
  group <- x[1]
  non_group <- x[-1]
  pre_bracket <- stringr::str_extract(non_group[1], "^.*(?=\\[)")
  if (is.na(pre_bracket)) {
    formatted_text <- glue::glue("{group}{concat_char}{non_group}")
  } else {
    bracket_contents <- non_group[-1] |>
      stringr::str_replace_all("\\s", "_") |>
      stringr::str_c(collapse = ",")
    formatted_text <- glue::glue(
      "{group}{concat_char}{pre_bracket}[{bracket_contents}]"
    )
  }
  formatted_text
}

#' Convert InferenceData column names to tidy column names
#'
#' InferenceData column names for scalar variables are of the form
#' `"('group', 'var_name')"`, while column names for array variables are of the
#'  form `"('group', 'var_name[i,j]', 'i_name', 'j_name')"`.
#'  This function converts these column names to a format that is useful for
#'  creating tidy_draws data frames.
#'  `"('group', 'var_name')"` becomes `"group|var_name"`
#'  `"('group', 'var_name[i,j]', 'i_name', 'j_name')"` becomes
#'  `"group|var_name[i_name, j_name]"`
#'
#' @param column_names A character vector of InferenceData column names
#'
#' @return A character vector of tidy column names
#' @examples
#' forecasttools:::idata_names_to_tidy_names(c(
#'   "('group', 'var_name')",
#'   "group|var_name[i_name, j_name]"
#' ))
idata_names_to_tidy_names <- function(column_names) {
  column_names |>
    stringr::str_remove_all("^\\(|\\)$") |>
    # remove opening and closing parentheses
    stringr::str_split(", ") |>
    purrr::map(\(x) stringr::str_remove_all(x, "^\\'|\\'$")) |>
    # remove opening and closing quotes
    purrr::map(\(x) stringr::str_remove_all(x, '\\"')) |> # remove double quotes
    purrr::map_chr(format_split_text) # reformat groups and brackets
}

#' Convert InferenceData DataFrame to nested tibble of tidy_draws
#'
#' @param idata InferenceData DataFrame (the result of calling
#' arviz.InferenceData.to_dataframe in Python)
#'
#' @return A nested tibble, with columns group and data. Each element of data is
#' a tidy_draws data frame
#' @export

inferencedata_to_tidy_draws <- function(idata) {
  idata |>
    dplyr::rename(
      .chain = chain,
      .iteration = draw
    ) |>
    dplyr::rename_with(idata_names_to_tidy_names,
      .cols = -tidyselect::starts_with(".")
    ) |>
    dplyr::mutate(dplyr::across(
      c(.chain, .iteration),
      \(x) as.integer(x + 1) # convert to 1-indexed
    )) |>
    dplyr::mutate(
      .draw = tidybayes:::draw_from_chain_and_iteration_(.chain, .iteration),
      .after = .iteration
    ) |>
    tidyr::pivot_longer(-starts_with("."),
      names_sep = "\\|",
      names_to = c("group", "name")
    ) |>
    dplyr::group_by(group) |>
    tidyr::nest() |>
    dplyr::mutate(data = purrr::map(data, \(x) {
      tidyr::drop_na(x) |>
        tidyr::pivot_wider(names_from = name) |>
        tidybayes::tidy_draws()
    }))
}
