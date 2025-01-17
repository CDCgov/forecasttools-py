# %% LIBRARIES USED


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
print(idata_wo_dates)

# %% WHEN IDATA IS CONVERTED TO DF THEN CSV

idata_wod_pandas_df = idata_wo_dates.to_dataframe()
idata_wod_pols_df = pl.from_pandas(idata_wod_pandas_df)
print(idata_wod_pols_df)

# %% UPDATED ATTEMPT


def format_to_tidy_draws(df):
    df.columns = [
        "_".join(map(str, col)).strip() if isinstance(col, tuple) else col
        for col in df.columns
    ]
    tidy_df = df.reset_index().melt(
        id_vars=["chain", "draw"], var_name="variable", value_name="value"
    )
    tidy_df["variable"] = tidy_df["variable"].apply(
        lambda x: x.split("_")[-1] if "posterior_predictive" in x else x
    )
    tidy_df[".iteration"] = tidy_df["draw"] + 1
    tidy_df = tidy_df.rename(columns={"chain": ".chain", "draw": ".draw"})
    tidy_df = tidy_df[[".chain", ".draw", ".iteration", "variable", "value"]]
    return tidy_df


idata_wd_df = idata_w_dates.to_dataframe()
# print(idata_wd_df)
out = format_to_tidy_draws(idata_wd_df)
print(out)


# %% CONVERSION FUNCTIONS


def format_split_text(x, concat_char="|"):
    group = x[0]
    non_group = x[1:]
    if "[" in non_group[0]:
        pre_bracket = non_group[0].split("[")[0]
        bracket_contents = [elem.replace(" ", "_") for elem in non_group[1:]]
        formatted_text = (
            f"{group}{concat_char}{pre_bracket}[{','.join(bracket_contents)}]"
        )
    else:
        formatted_text = f"{group}{concat_char}{''.join(non_group)}"
    return formatted_text


def idata_names_to_tidy_names(column_names):
    tidy_names = []
    for col in column_names:
        cleaned = col.strip("()").replace("'", "").replace('"', "")
        parts = [part.strip() for part in cleaned.split(",")]
        tidy_names.append(format_split_text(parts))
    return tidy_names


def inferencedata_to_tidy_draws(idata_df):
    idata_df = idata_df.rename({"chain": ".chain", "draw": ".iteration"})
    idata_df = idata_df.with_columns(
        [
            (pl.col(".chain") + 1).alias(".chain"),
            (pl.col(".iteration") + 1).alias(".iteration"),
        ]
    )
    max_iteration = idata_df[".iteration"].max()
    idata_df = idata_df.with_columns(
        ((pl.col(".chain") - 1) * max_iteration + pl.col(".iteration")).alias(
            ".draw"
        )
    )
    tidy_column_names = idata_names_to_tidy_names(idata_df.columns)
    idata_df.columns = tidy_column_names
    long_df = idata_df.melt(
        id_vars=[".chain", ".iteration", ".draw"],
        variable_name="group|name",
        value_name="value",
    )
    long_df = long_df.with_columns(
        long_df["group|name"].str.split_exact("|", 1).alias(["group", "name"])
    ).drop("group|name")
    nested = long_df.groupby("group").agg(
        pl.col([".draw", "name", "value"]).pivot(
            index=".draw", columns="name", values="value"
        )
    )
    return nested


postp_tidy_df = inferencedata_to_tidy_draws(idata_df=idata_wod_pols_df)
print(postp_tidy_df)
# %%
