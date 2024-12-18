arviz_split <- function(x) {
x %>%
    select(-distribution) %>%
    split(f = as.factor(x$distribution))
}

pyrenew_samples <-
read_csv(inference_data_path) %>%
rename_with(\(varname) str_remove_all(varname, "\\(|\\)|\\'|(, \\d+)")) |>
rename(
    .chain = chain,
    .iteration = draw
) |>
mutate(across(c(.chain, .iteration), \(x) as.integer(x + 1))) |>
mutate(
    .draw = tidybayes:::draw_from_chain_and_iteration_(.chain, .iteration),
    .after = .iteration
) |>
pivot_longer(-starts_with("."),
    names_sep = ", ",
    names_to = c("distribution", "name")
) |>
arviz_split() |>
map(\(x) pivot_wider(x, names_from = name) |> tidy_draws())
