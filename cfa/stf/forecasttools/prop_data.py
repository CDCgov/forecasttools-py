import polars as pl
import polars.selectors as cs


def append_prop_data(
    data: pl.DataFrame,
    observed_var: str = "observed_ed_visits",
    other_var: str = "other_ed_visits",
    prop_var: str = "prop_disease_ed_visits",
) -> pl.DataFrame:
    """Append disease ED visit proportion rows to combined surveillance data."""
    required_columns = {"date", ".variable", ".value"}
    if missing_columns := required_columns.difference(data.columns):
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required column(s): {missing}")

    value_vars = [observed_var, other_var]

    prop_data = data.filter(pl.col(".variable").is_in(value_vars)).pivot(
        on=".variable",
        index=cs.exclude(".variable", ".value"),
        values=".value",
    )
    all_null_columns = (
        prop_data.select(cs.all().is_null().all()).row(0, named=True).items()
    )
    all_null_columns = [
        column for column, column_is_all_null in all_null_columns if column_is_all_null
    ]
    if all_null_columns:
        prop_data = prop_data.select(cs.exclude(all_null_columns))

    prop_data = prop_data.with_columns(
        pl.lit(prop_var).alias(".variable"),
        (pl.col(observed_var) / (pl.col(observed_var) + pl.col(other_var))).alias(
            ".value"
        ),
    ).drop(value_vars)

    return pl.concat([data, prop_data], how="diagonal_relaxed").sort(
        "date", ".variable", maintain_order=True
    )
