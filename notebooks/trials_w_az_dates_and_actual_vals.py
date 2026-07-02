"""
Test file for this PR to examine
forecasting workflows that make
use of `scoringutils`. There should
be a way to get dates associated
with the entire forecasting process
in an idata object and then also,
once some time has passed, get the
actual values as a new group.
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

# %% CHECK FILE PATH FROM DATA.py


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
    state: str,
    start_date: str,
    end_date: str,
    last_fit: int,
    X_act: NDArray,
    y_act: NDArray,
    save_to_pdf: bool = False,
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

    median_ts = y_data.median(dim=["chain", "draw"])
    axes.plot(x_data, median_ts, color="blue", label="Median")
    axes.legend()
    axes.axvline(last_fit, color="black", linestyle="--")
    axes.set_title(
        f"Hospital Admissions ({state}, {start_date} to {end_date})",
        fontsize=15,
    )
    axes.set_xlabel("Time", fontsize=20)
    axes.set_ylabel("Hospital Admissions", fontsize=20)
    plt.show()


# %% MAKE A FORECAST


def make_forecast(
    nhsn_dataset_path: str,
    juris_subset: list[str],
    start_date: str,
    end_date: str,
    forecast_days: int,
    save_path: str = os.path.join(os.getcwd(), "forecast.nc"),
    show_plot: bool = True,
    save_idata: bool = False,
) -> None:
    """
    Generates a forecast for specified
    dates and jurisdictions using a spline
    regression model.

    Parameters
    ----------
    nhsn_dataset_path
        The path to the NHSN influenza
        dataset.
    save_path
        The path to where the outputted
        parquet file should be saved.
        Defaults to current directory.
    start_date
        Where to begin data fitting.
    end_date
        Where to end data fitting.
    forecast_days
        The number of days to forecast.
    juris_subset
        The jurisdictions for which
        to make forecasts.
    show_plot
        Whether to show the forecast.
        Defaults to True.
    save_idata
        Whether to actually save the output.
        Defaults to True.

    Returns
    -------
    tuple
        A plotted forecast and or a
        NetCDF Arviz object.
    """
    # check dataset path
    check_file_save_path(save_path)
    # load dataset
    nhsn = pl.read_csv(nhsn_dataset_path)
    # check if provided jurisdictions are in NHSN jurisdictions
    nhsn_juris = list(nhsn["state"].unique().to_numpy())
    assert set(juris_subset).issubset(
        set(nhsn_juris)
    ), f"There are jurisdictions present that are not found in the dataset.\nEntered {juris_subset}, Available: {nhsn_juris}"
    # clean data and organize data, cleaning null values
    nhsn = nhsn.filter(
        pl.col("hosp").is_not_null(), pl.col("state").is_in(juris_subset)
    ).with_columns(
        pl.col("hosp").cast(pl.Int64),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    )
    nhsn_ready = nhsn.filter(
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
        nhsn_actual = nhsn.filter(
            (
                pl.col("date")
                >= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
            & (pl.col("date") <= pl.lit(forecast_end_date))
        )
    except Exception as e:
        nhsn_actual = None
        print(f"The following error occurred: {e}")
    # define some shared inference values
    random_seed = 2134312
    num_samples = 1000
    num_warmup = 500
    # store aggregated idata objects
    idatas = []
    # get posterior samples and make forecasts for each selected state
    for state in juris_subset:
        # get the state data
        state_nhsn = nhsn_ready.filter(pl.col("state") == state)
        # get observation (fitting) data y, X
        y = state_nhsn["hosp"].to_numpy()
        X = np.arange(y.shape[0])
        # set up inference, NUTS/MCMC
        kernel = numpyro.infer.NUTS(model=model)
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
        idatas.append(idata)
        # get actual data, if it exists
        if isinstance(nhsn_actual, pl.DataFrame):
            actual_data = nhsn_actual.filter(pl.col("state") == state)
            y_act = actual_data["hosp"].to_numpy()
            X_act = np.arange(last, last + 1 + forecast_days)
        if not isinstance(nhsn_actual, pl.DataFrame):
            y_act = None
            X_act = None
        # save idata object(s)
        if save_idata:
            idata.to_netcdf(save_path)
        # plot forecast (if desired) from idata light
        if show_plot:
            plot_and_or_save_forecast(
                idata=idata,
                X=X,
                y=y,
                state=state,
                start_date=start_date,
                end_date=end_date,
                last_fit=last,
                X_act=X_act,
                y_act=y_act,
            )


# %% EXECUTION

make_forecast(
    nhsn_dataset_path="../forecasttools/nhsn_hosp_flu.csv",
    juris_subset=["AZ", "FL", "VT"],
    start_date="2023-08-08",
    end_date="2024-02-15",
    forecast_days=28,
)

# %% SCORING UTILS SIMPLE

data = {
    "location": ["DE", "DE", "IT", "IT"],
    "forecast_date": ["2021-01-01", "2021-01-01", "2021-07-12", "2021-07-12"],
    "target_end_date": [
        "2021-01-02",
        "2021-01-02",
        "2021-07-24",
        "2021-07-24",
    ],
    "target_type": ["Cases", "Deaths", "Deaths", "Deaths"],
    "model": [None, None, "epiforecasts-EpiNow2", "epiforecasts-EpiNow2"],
    "horizon": [None, None, 2, 2],
    "quantile_level": [None, None, 0.975, 0.990],
    "predicted": [None, None, 611, 719],
    "observed": [127300, 4534, 78, 78],
}

df = pl.DataFrame(data)

# save to parquet
print(df)
