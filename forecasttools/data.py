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
    file_save_path = os.path.join(save_directory, file_save_name)
    if os.path.exists(file_save_path):
        raise FileExistsError(f"The file {file_save_path} already exists.")
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
        location_table.write_csv(file_save_path)
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
            # change date formatting to ISO8601
            df = df.with_columns(df["date"].str.replace_all("/", "-"))
            # pathogen specific df saving
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


def make_nhsn_fitted_forecast_idata(
    nhsn_dataset_path: str,
    save_directory: str,
    file_save_name: str,
    start_date: str = "2022/08/08",
    end_date: str = "2023/12/08",
    forecast_days: int = 28,
    juris_subset: list[str] = ["TX"],
    create_save_directory: bool = False,
    show_plot: bool = True,
    save_idata: bool = True,
):
    """
    Make an example forecast idata object
    using a spline regression model on
    NHSN influenza hospitalization count
    data.
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
    df_scan = pl.scan_csv(nhsn_dataset_path)
    df_cols = df_scan.columns
    required_cols = ["state", "date", "hosp"]
    if len(set(required_cols).intersection(set(df_cols))) != len(
        required_cols
    ):
        raise ValueError(
            f"NHSN dataset missing required columns: {set(required_cols) - set(required_cols).intersection(set(df_cols))}"
        )
    # load dataset
    nhsn = pl.read_csv(nhsn_dataset_path)
    # check if provided jurisdictions are in NHSN jurisdictions
    nhsn_juris = list(nhsn["state"].unique().to_numpy())
    assert len(set(juris_subset).intersection(set(nhsn_juris))) == len(
        juris_subset
    ), f"There are jurisdictions present that are not found in the dataset.\nEntered {juris_subset}, Available: {nhsn_juris}"
    # clean data and organize data, cleaning null values
    nhsn = nhsn.filter(pl.col("hosp").is_not_null())
    nhsn = nhsn.filter(pl.col("state").is_in(juris_subset))
    nhsn = nhsn.with_columns(
        pl.col("hosp").cast(pl.Int64),
        pl.col("date").str.strptime(pl.Date, "%Y/%m/%d"),
    )
    nhsn = nhsn.filter(
        (
            pl.col("date")
            >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
        & (
            pl.col("date")
            <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
    )
    nhsn = nhsn.group_by("state").agg([pl.col("hosp")])

    # spline basis function
    def spline_basis(X, degree=4, df=8):
        basis = patsy.dmatrix(
            "bs(x, df=df, degree=degree, include_intercept=True) - 1",
            {"x": X, "df": df, "degree": degree},
            return_type="matrix",
        )
        return np.array(basis)

    # spline regression model(s)
    def model(basis_matrix, y=None):
        # priors
        shift = npro.sample("shift", dist.Normal(0.0, 5.0))
        beta_coeffs = npro.sample(
            "beta_coeffs", dist.Normal(jnp.zeros(basis_matrix.shape[1]), 5.0)
        )
        shift_mu = jnp.dot(basis_matrix, beta_coeffs) + shift
        mu_exp = jnp.exp(shift_mu)
        alpha = npro.sample("alpha", dist.Exponential(1.0))
        # likelihood
        npro.sample("obs", dist.NegativeBinomial2(mu_exp, alpha), obs=y)

    def model2(X, basis_matrix, y=None):
        # priors
        time_drift = npro.sample("time_drift", dist.Normal(0, 3.0))
        drift_effect = time_drift * X
        beta_coeffs = npro.sample(
            "beta_coeffs", dist.Normal(jnp.zeros(basis_matrix.shape[1]), 3.0)
        )
        mu = jnp.dot(basis_matrix, beta_coeffs) + drift_effect
        mu_exp = jnp.exp(mu)
        alpha = npro.sample("alpha", dist.Exponential(1.0))
        # likelihood
        npro.sample("obs", dist.NegativeBinomial2(mu_exp, alpha), obs=y)

    # define some shared inference values
    random_seed = 2134312
    num_samples, num_warmup = 1000, 500
    # get posterior samples and make forecasts for each selected state
    for state in juris_subset:
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
        prior_pred = npro.infer.Predictive(model, num_samples=num_samples)(
            rng_key=jr.key(random_seed), basis_matrix=sbm[: len(X)]
        )
        # get posterior predictive forecast
        posterior_pred_for = npro.infer.Predictive(
            model, posterior_samples=posterior_samples
        )(rng_key=jr.key(random_seed), basis_matrix=sbm)
        # create initial inference data object(s)
        idata = az.from_numpyro(
            posterior=mcmc,
            posterior_predictive=posterior_pred_for,
            prior=prior_pred,
        )
        # save idata object(s)
        idata.to_netcdf(file_save_path)
        # plot forecast (if desired) from idata light
        if show_plot:
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
            median_ts = y_data.median(dim=["chain", "draw"])
            axes.plot(x_data, median_ts, color="blue", label="Median")
            axes.legend()
            axes.axvline(last, color="black", linestyle="--")
            axes.set_title(
                f"Hospital Admissions ({state}, {start_date}-{end_date})",
                fontsize=15,
            )
            axes.set_xlabel("Time", fontsize=20)
            axes.set_ylabel("Hospital Admissions", fontsize=20)
            plt.show()


def read_example_flusight_submission(
    save_directory: str,
    file_save_name: str,
    create_save_directory: bool = False,
) -> None:
    """
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
    file_save_path = os.path.join(save_directory, file_save_name)
    if os.path.exists(file_save_path):
        raise FileExistsError(f"The file {file_save_path} already exists.")
    else:
        # check if the FluSight example url is still valid
        url = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/cfa-flumech/2023-10-14-cfa-flumech.csv"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"FluSight example url is valid: {url}")
            else:
                raise ValueError(
                    f"FluSight example url bad status: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to access FluSight example url: {e}")
        # read csv from URL, convert to polars
        submission_df = pl.read_csv(url, infer_schema_length=7500)
        # save the dataframe
        submission_df.write_csv(file_save_path)
        print(
            f"The file {file_save_name} has been created at {save_directory}."
        )
