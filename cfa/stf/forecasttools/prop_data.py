import polars as pl

OBSERVED_ED_VISITS_VAR = "observed_ed_visits"
OTHER_ED_VISITS_VAR = "other_ed_visits"
PROP_DISEASE_ED_VISITS_VAR = "prop_disease_ed_visits"


def append_prop_data(
    data: pl.DataFrame,
    observed_var: str = OBSERVED_ED_VISITS_VAR,
    other_var: str = OTHER_ED_VISITS_VAR,
    prop_var: str = PROP_DISEASE_ED_VISITS_VAR,
) -> pl.DataFrame:
    """Append disease ED visit proportion rows to combined surveillance data."""
    required_columns = {"date", ".variable", ".value"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required column(s): {missing}")

    value_vars = [observed_var, other_var]
    id_columns = [col for col in data.columns if col not in {".variable", ".value"}]

    prop_data = data.filter(pl.col(".variable").is_in(value_vars)).pivot(
        on=".variable",
        index=id_columns,
        values=".value",
    )
    prop_data = prop_data.select(
        [
            col
            for col in prop_data.columns
            if not prop_data.get_column(col).is_null().all()
        ]
    )
    prop_data = prop_data.with_columns(
        pl.lit(prop_var).alias(".variable"),
        (pl.col(observed_var) / (pl.col(observed_var) + pl.col(other_var))).alias(
            ".value"
        ),
    ).drop(value_vars)

    return pl.concat([data, prop_data], how="diagonal_relaxed").sort(
        "date", ".variable", maintain_order=True
    )
