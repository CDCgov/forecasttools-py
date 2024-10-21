"""
Creating a new idata object with
dates to change the functionality
of idata_to_df_w_dates.
"""

# %% IMPORTS

import os
from datetime import datetime, timedelta

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import patsy
import polars as pl
from numpy.typing import NDArray

import forecasttools

# %% CHECK FILE PATH


def check_file_save_path(file_save_path: str) -> None:
    """
    Checks whether a file path is valid.

    file_save_path
        The file path to be checked.
    """
    directory = os.path.dirname(file_save_path)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory is not writable: {directory}")
    if os.path.exists(file_save_path):
        raise FileExistsError(f"File already exists at: {file_save_path}")


# %% SPLINE REGRESSION MODEL


def model(basis_matrix, y=None):
    # priors
    shift = numpyro.sample("shift", dist.Normal(0.0, 2.0))
    beta_coeffs = numpyro.sample(
        "beta_coeffs", dist.Normal(jnp.zeros(basis_matrix.shape[1]), 2.0)
    )
    shift_mu = jnp.dot(basis_matrix, beta_coeffs) + shift
    mu_exp = jnp.exp(shift_mu)
    alpha = numpyro.sample("alpha", dist.Exponential(1.0))
    # likelihood
    numpyro.sample("obs", dist.NegativeBinomial2(mu_exp, alpha), obs=y)


# %% SPLINE BASIS MATRIX


def spline_basis(X, degree: int = 4, df: int = 8) -> NDArray:
    basis = patsy.dmatrix(
        "bs(x, df=df, degree=degree, include_intercept=True) - 1",
        {"x": X, "df": df, "degree": degree},
        return_type="matrix",
    )
    return np.array(basis)


# %% PLOT AND OR SAVE FORECAST


def plot_and_or_save_forecast(
    idata: az.InferenceData,
    X: NDArray,
    y: NDArray,
    title: str,
    start_date: str,
    end_date: str,
    last_fit: int,
    X_act: NDArray,
    y_act: NDArray,
    save_to_pdf: bool = False,
    use_log: bool = False,
):
    """
    Includes hard-coded variables. For the
    author's testing and no more.
    """
    x_data = idata.posterior_predictive["obs_dim_0"]
    y_data = idata.posterior_predictive["obs"]
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.95,
        color="skyblue",
        smooth=False,
        fill_kwargs={"alpha": 0.2, "label": "95% Credible"},
        ax=axes,
    )
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.75,
        color="skyblue",
        smooth=False,
        fill_kwargs={"alpha": 0.4, "label": "75% Credible"},
        ax=axes,
    )
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.5,
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.6, "label": "50% Credible"},
        ax=axes,
    )
    axes.plot(
        X,
        y,
        marker="o",
        color="black",
        linewidth=1.0,
        markersize=3.0,
        label="Observed",
    )
    if (X_act is not None) and (y_act is not None):
        axes.plot(
            X_act,
            y_act,
            marker="o",
            color="red",
            linewidth=1.0,
            markersize=3.0,
            label="Actual",
        )
    if use_log:
        axes.set_yscale("log")
        axes.set_ylabel("(Log) Hospital Admissions", fontsize=17.5)
    if not use_log:
        axes.set_ylabel("Hospital Admissions", fontsize=17.5)
    median_ts = y_data.median(dim=["chain", "draw"])
    axes.plot(x_data, median_ts, color="blue", label="Median")
    axes.legend()
    axes.axvline(last_fit, color="black", linestyle="--")
    axes.set_title(
        f"{title}",
        fontsize=20,
    )
    axes.set_xlabel("Time", fontsize=17.5)

    plt.show()


# %% ADD DATES TO AN INFERENCE DATA OBJECT


def add_dates_to_idata_object(
    idata: az.InferenceData,
    start_date: str,
) -> az.InferenceData:
    """
    Takes an InferenceData object w/
    observed_data and posterior_predictive
    groups and adds date indexing
    """
    pass


# %% MAKE A FORECAST


def make_forecast(
    nhsn_data: str,
    start_date: str,
    end_date: str,
    juris_subset: list[str],
    forecast_days: int,
    save_path: str = os.path.join(os.getcwd(), "forecast.nc"),
    show_plot: bool = True,
    save_idata: bool = False,
    use_log: bool = False,
) -> None:
    """
    Generates a forecast for specified
    dates using a spline regression model.
    """
    # check dataset path
    check_file_save_path(save_path)
    # clean data and organize data, cleaning null values
    nhsn_data = nhsn_data.with_columns(
        pl.col("hosp").cast(pl.Int64),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    ).filter(pl.col("hosp").is_not_null(), pl.col("state").is_in(juris_subset))
    nhsn_data_ready = nhsn_data.filter(
        (
            pl.col("date")
            >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
        & (
            pl.col("date")
            <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
    )
    # get the actual values, if they exist
    try:
        forecast_end_date = datetime.strptime(
            end_date, "%Y-%m-%d"
        ) + timedelta(days=forecast_days)
        nhsn_data_actual = nhsn_data.filter(
            (
                pl.col("date")
                >= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
            & (pl.col("date") <= pl.lit(forecast_end_date))
        )
    except Exception as e:
        nhsn_data_actual = None
        print(f"The following error occurred: {e}")
    # define some shared inference values
    random_seed = 2134312
    num_samples = 1000
    num_warmup = 500
    # get posterior samples and make forecasts for each selected state
    for state in juris_subset:
        # get the state data
        state_nhsn = nhsn_data_ready.filter(pl.col("state") == state)
        # get observation (fitting) data y, X
        y = state_nhsn["hosp"].to_numpy()
        X = np.arange(y.shape[0])
        # set up inference, NUTS/MCMC
        kernel = numpyro.infer.NUTS(
            model=model,
            max_tree_depth=12,
            target_accept_prob=0.85,
            init_strategy=numpyro.infer.init_to_uniform(),
        )
        mcmc = numpyro.infer.MCMC(
            kernel, num_warmup=num_warmup, num_samples=num_samples
        )
        # create spline basis for obs period and forecast period
        last = X[-1]
        X_future = np.hstack(
            (X, np.arange(last + 1, last + 1 + forecast_days))
        )
        sbm = spline_basis(X_future)
        # get posterior samples
        mcmc.run(rng_key=jr.key(random_seed), basis_matrix=sbm[: len(X)], y=y)
        posterior_samples = mcmc.get_samples()
        # get prior predictive
        prior_pred = numpyro.infer.Predictive(model, num_samples=num_samples)(
            rng_key=jr.key(random_seed), basis_matrix=sbm[: len(X)]
        )
        # get posterior predictive forecast
        posterior_pred_for = numpyro.infer.Predictive(
            model, posterior_samples=posterior_samples
        )(rng_key=jr.key(random_seed), basis_matrix=sbm)
        # create initial inference data object(s) and store
        idata = az.from_numpyro(
            posterior=mcmc,
            posterior_predictive=posterior_pred_for,
            prior=prior_pred,
        )
        # get actual data, if it exists
        if isinstance(nhsn_data_actual, pl.DataFrame):
            actual_data = nhsn_data_actual.filter(pl.col("state") == state)
            y_act = actual_data["hosp"].to_numpy()
            X_act = np.arange(last - 1, last + forecast_days)
        if not isinstance(nhsn_data_actual, pl.DataFrame):
            y_act = None
            X_act = None
        # add dates to idata object

        # save idata object(s)
        if save_idata:
            idata.to_netcdf(save_path)
        # plot forecast (if desired) from idata light
        if show_plot:
            plot_and_or_save_forecast(
                idata=idata,
                X=X,
                y=y,
                title=f"Hospital Admissions ({state}, {start_date}-{end_date})",
                start_date=start_date,
                end_date=end_date,
                last_fit=last,
                X_act=X_act,
                y_act=y_act,
                use_log=use_log,
            )


# %% EXECUTE MODE

make_forecast(
    nhsn_data=forecasttools.nhsn_hosp_flu,
    start_date="2022-08-08",
    end_date="2022-12-08",
    juris_subset=["TX"],
    forecast_days=28,
    save_path="../forecasttools/example_flu_forecast_w_dates.nc",
    save_idata=False,
    use_log=True,
)

# %%

# )
