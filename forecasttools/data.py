"""
Retrieves US 2020 Census data,
NHSN (as of 2024-09-26) influenza
hospitalization counts, NHSN
(as of 2024-09-26) COVID-19
hospitalization counts, and
a forecasts for a single
jurisdiction.
"""

import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import os

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import patsy
import polars as pl
import requests


def make_census_dataset(
    save_directory: str,
    file_save_name: str,
    create_save_directory: bool = False,
) -> None:
    """
    Retrieves US 2020 Census data in a
    three column polars dataframe, then
    saves the dataset as a csv in a given
    directory, if it does not already exist.

    Parameters
    ----------
    save_directory
        The directory to save the outputted csv in.
    file_save_name
        The name of the file to save.
    create_save_directory
        Whether to make the save directory, if it
        does not exist.
    """
    # check if the save directory exists
    if not os.path.exists(save_directory):
        if create_save_directory:
            os.makedirs(save_directory)
            print(f"Directory {save_directory} created.")
        else:
            raise FileNotFoundError(
                f"The directory {save_directory} does not exist."
            )
    # check if save directory is actual directory
    if not os.path.isdir(save_directory):
        raise NotADirectoryError(
            f"The path {save_directory} is not a directory."
        )
    # check if the save file already exists
    file_path = os.path.join(save_directory, file_save_name)
    if os.path.exists(file_path):
        raise FileExistsError(f"The file {file_path} already exists.")
    else:
        # check if the census url is still valid
        url = "https://www2.census.gov/geo/docs/reference/state.txt"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"Census url is valid: {url}")
            else:
                raise ValueError(
                    f"Census url bad status: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to access Census url: {e}")
        # create location table with code, abbreviation, and full name
        nation = pl.DataFrame(
            {
                "location_code": ["US"],
                "short_name": ["US"],
                "long_name": ["United States"],
            }
        )
        jurisdictions = pl.read_csv(url, separator="|").select(
            [
                pl.col("STATE").alias("location_code").cast(pl.Utf8),
                pl.col("STUSAB").alias("short_name"),
                pl.col("STATE_NAME").alias("long_name"),
            ]
        )
        location_table = nation.vstack(jurisdictions)
        location_table.write_csv(file_path)
        print(
            f"The file {file_save_name} has been created at {save_directory}."
        )


def make_nshn_fitting_dataset(
    dataset: str,
    nhsn_dataset_path: str,
    save_directory: str,
    file_save_name: str,
    create_save_directory: bool = False,
) -> None:
    """
    Create a polars dataset with columns date,
    state, and hosp and save a CSV. Can be used
    for COVID or influenza. This function DOES
    NOT use the API endpoint, and instead expects
    a CSV.

    Parameters
    ----------
    dataset
        Name of the dataset to create. Either "COVID",
        "flu", or "both".
    nhsn_dataset_path
        Path to the NHSN dataset (csv file).
    save_directory
        The directory to save the outputted csv in.
    file_save_name
        The name of the file to save.
    create_save_directory
        Whether to make the save directory, if it
        does not exist.
    """
    # check that dataset parameter is possible
    assert dataset in [
        "COVID",
        "flu",
    ], 'Dataset {dataset} must be one of "COVID", "flu"'
    # check if the save directory exists
    if not os.path.exists(save_directory):
        if create_save_directory:
            os.makedirs(save_directory)
            print(f"Directory {save_directory} created.")
        else:
            raise FileNotFoundError(
                f"The directory {save_directory} does not exist."
            )
    # check if save directory is an actual directory
    if not os.path.isdir(save_directory):
        raise NotADirectoryError(
            f"The path {save_directory} is not a directory."
        )
    # check if the save (output) file already exists
    file_save_path = os.path.join(save_directory, file_save_name)
    if os.path.exists(file_save_path):
        raise FileExistsError(f"The file {file_save_path} already exists.")
    # create dataset(s) from NHSN data
    else:
        # check that a data file exists
        if not os.path.exists(nhsn_dataset_path):
            raise FileNotFoundError(
                f"The file {nhsn_dataset_path} does not exist."
            )
        else:
            # check that the loaded CSV has the needed columns
            df_cols = pl.scan_csv(nhsn_dataset_path).columns
            required_cols = [
                "state",
                "date",
                "previous_day_admission_adult_covid_confirmed",
                "previous_day_admission_influenza_confirmed",
            ]
            if len(set(required_cols).intersection(set(df_cols))) != len(
                required_cols
            ):
                raise ValueError(
                    f"NHSN dataset missing required columns: {set(required_cols) - set(required_cols).intersection(set(df_cols))}"
                )
            # fully load and save NHSN dataframe
            df = pl.read_csv(nhsn_dataset_path)
            if dataset == "COVID":
                df_covid = (
                    df.select(
                        [
                            "state",
                            "date",
                            "previous_day_admission_adult_covid_confirmed",
                        ]
                    )
                    .rename(
                        {
                            "previous_day_admission_adult_covid_confirmed": "hosp"
                        }
                    )
                    .sort(["state", "date"])
                )
                df_covid.write_csv(file_save_path)
            if dataset == "flu":
                df_flu = (
                    df.select(
                        [
                            "state",
                            "date",
                            "previous_day_admission_influenza_confirmed",
                        ]
                    )
                    .rename(
                        {"previous_day_admission_influenza_confirmed": "hosp"}
                    )
                    .sort(["state", "date"])
                )
                df_flu.write_csv(file_save_path)
            print(
                f"The file {file_save_name} has been created at {save_directory}."
            )


def make_nhsn_fitted_forecast(
    nhsn_dataset_path: str,
    save_directory: str,
    file_save_name: str,
    start_date: str = "2023/08/08",
    end_date: str = "2024/02/01",
    forecast_days: int = 30,
    jurisdiction_subset: list[str] = ["AZ"],
    create_save_directory: bool = False,
):
    """ """
    # check if the save directory exists
    if not os.path.exists(save_directory):
        if create_save_directory:
            os.makedirs(save_directory)
            print(f"Directory {save_directory} created.")
        else:
            raise FileNotFoundError(
                f"The directory {save_directory} does not exist."
            )
    # check if save directory is an actual directory
    if not os.path.isdir(save_directory):
        raise NotADirectoryError(
            f"The path {save_directory} is not a directory."
        )
    # check if the save (output) file already exists
    file_save_path = os.path.join(save_directory, file_save_name)
    if os.path.exists(file_save_path):
        raise FileExistsError(f"The file {file_save_path} already exists.")
    # check that dataset exists
    if not os.path.exists(nhsn_dataset_path):
        raise FileNotFoundError(
            f"The file {nhsn_dataset_path} does not exist."
        )
    # check that the loaded CSV has the needed columns
    df_cols = pl.scan_csv(nhsn_dataset_path).columns
    required_cols = ["state", "date", "hosp"]
    if len(set(required_cols).intersection(set(df_cols))) != len(
        required_cols
    ):
        raise ValueError(
            f"NHSN dataset missing required columns: {set(required_cols) - set(required_cols).intersection(set(df_cols))}"
        )
    # check if jurisdictions are acceptable
    # assert len(set(jurisdiction_subset).intersection(set(df_cols))) == len(jurisdiction_subset),"There are jurisdictions present that are not found in the dataset."
    # load dataset, cleaning null values
    nhsn = pl.read_csv(nhsn_dataset_path)
    nhsn = nhsn.filter(pl.col("hosp").is_not_null())
    nhsn = nhsn.filter(pl.col("state").is_in(jurisdiction_subset))
    nhsn = nhsn.with_columns(
        pl.col("hosp").cast(pl.Int64),
        pl.col("date").str.strptime(pl.Date, "%Y/%m/%d"),
    )
    nhsn = nhsn.filter(
        (
            pl.col("date")
            >= pl.lit(start_date).str.strptime(pl.Date, "%Y/%m/%d")
        )
        & (
            pl.col("date")
            <= pl.lit(end_date).str.strptime(pl.Date, "%Y/%m/%d")
        )
    )
    nhsn = nhsn.group_by("state").agg([pl.col("hosp")])

    # spline basis function
    def spline_basis(X, degree=3, df=5):
        basis = patsy.dmatrix(
            "bs(x, df=df, degree=degree, include_intercept=True) - 1",
            {"x": X, "df": df, "degree": degree},
            return_type="matrix",
        )
        return np.array(basis)

    # spline regression model
    def model(X, basis_matrix, y=None):
        # priors
        beta_coeffs = npro.sample(
            "beta_coeffs", dist.Normal(jnp.zeros(basis_matrix.shape[1]), 0.1)
        )
        mu = jnp.dot(basis_matrix, beta_coeffs)
        mu_exp = jnp.exp(mu)
        # likelihood
        npro.sample("obs", dist.Poisson(mu_exp), obs=y)

    # define some shared inference values
    random_seed = 2134312
    num_samples, num_warmup = 1000, 500
    # get posterior samples and make forecasts for each selected state
    for state in jurisdiction_subset:
        # get the state data
        state_nhsn = nhsn.filter(pl.col("state") == state)
        # get actual data y, X
        y = state_nhsn["hosp"].to_numpy()[0]
        X = np.arange(y.shape[0])
        # set up inference, NUTS/MCMC
        kernel = npro.infer.NUTS(model=model)
        mcmc = npro.infer.MCMC(
            kernel, num_warmup=num_warmup, num_samples=num_samples
        )
        sbm = spline_basis(X)
        mcmc.run(rng_key=jr.key(random_seed), X=X, basis_matrix=sbm, y=y)
        posterior_samples = mcmc.get_samples()
        # get prior predictive
        prior_pred = npro.infer.Predictive(model, num_samples=num_samples)(
            rng_key=jr.key(random_seed), X=X, basis_matrix=sbm
        )
        # get posterior predictive fit
        posterior_pred_fit = npro.infer.Predictive(
            model, posterior_samples=posterior_samples
        )(rng_key=jr.key(random_seed), X=X, basis_matrix=sbm)
        # get posterior predictive forecast
        last = X[-1]
        X_future = np.hstack(
            (X, np.arange(last + 1, last + 1 + forecast_days))
        )
        sbm_future = spline_basis(X_future)
        posterior_pred_for = npro.infer.Predictive(
            model, posterior_samples=posterior_samples
        )(rng_key=jr.key(random_seed), X=X_future, basis_matrix=sbm_future)
        # create inference data object
        constant_data = {"X": X, "X_future": X_future}
        idata = az.from_numpyro(
            posterior=mcmc,
            posterior_predictive=posterior_pred_fit,
            prior=prior_pred,
            constant_data=constant_data,
            predictions=posterior_pred_for,
            predictions_constant_data={"obs": X_future},
        )
        # plot the posterior predictive fit results
        axes = az.plot_ts(
            idata,
            y="obs",
            y_hat="obs",
            y_forecasts="predictions",
            num_samples=100,
            y_kwargs={
                "color": "blue",
                "linewidth": 1.0,
                "marker": "o",
                "markersize": 3.0,
                "linestyle": "solid",
            },
            y_hat_plot_kwargs={"color": "skyblue", "alpha": 0.05},
            y_mean_plot_kwargs={
                "color": "black",
                "linestyle": "--",
                "linewidth": 2.5,
            },
            backend_kwargs={"figsize": (8, 6)},
            textsize=15.0,
        )
        ax = axes[0][0]
        ax.set_xlabel("Time", fontsize=20)
        ax.axvline(last, color="black", linestyle="--")
        ax.set_ylabel("Hospital Admissions", fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            ["Observed", "Sample Mean", "Posterior Samples"],
            loc="best",
        )
        plt.show()


# make_nhsn_fitted_forecast(
#     nhsn_dataset_path="nhsn_hosp_flu.csv",
#     save_directory=os.getcwd(),
#     file_save_name="example_flu_forecast.csv")

# make_census_dataset(
#     save_directory=os.getcwd(),
#     file_save_name="location_table.csv")

# make_nshn_fitting_dataset(
#     dataset="flu",
#     nhsn_dataset_path="NHSN_RAW_20240926.csv",
#     save_directory=os.getcwd(),
#     file_save_name="nhsn_hosp_flu.csv"
# )

# make_nshn_fitting_dataset(
#     dataset="COVID",
#     nhsn_dataset_path="NHSN_RAW_20240926.csv",
#     save_directory=os.getcwd(),
#     file_save_name="nhsn_hosp_COVID.csv"
# )
