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


def get_and_save_flusight_submission(
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
