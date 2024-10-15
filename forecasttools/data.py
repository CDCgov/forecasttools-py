"""
Retrieves US 2020 Census data,
NHSN (as of 2024-09-26) influenza
hospitalization counts, NHSN
(as of 2024-09-26) COVID-19
hospitalization counts, and
an example FluSight submission.
"""

import os
from urllib import error, request

import polars as pl


def check_url(url: str) -> bool:
    """
    Checks whether a URL is valid.
    Used for checking the links to the
    example FluSight submission and the
    US 2020 Census data.

    url
        The url to be checked.
    """
    try:
        response = request.urlopen(url)
        return response.status == 200
    except error.URLError as e:
        print(f"URL Error: {e.reason}")
        return False
    except error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        return False


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


def make_census_dataset(
    file_save_path: str,
) -> None:
    """
    Retrieves US 2020 Census data in a
    three column Polars dataframe, then
    saves the dataset as a csv in a given
    directory, if it does not already exist.

    file_save_path
        The path for where to save the output file.
    """
    # check that file and directory paths are valid
    check_file_save_path(file_save_path)
    # check if the census url is still valid
    url = "https://www2.census.gov/geo/docs/reference/state.txt"
    check_url(url)
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
    print(f"The file {file_save_path} has been saved.")


def make_nshn_fitting_dataset(
    dataset: str,
    nhsn_dataset_path: str,
    file_save_path: str,
) -> None:
    """
    Create a polars dataset with columns date,
    state, and hosp and save a CSV. Can be used
    for COVID or influenza. This function DOES
    NOT use the API endpoint, and instead expects
    a CSV.

    dataset
        Name of the dataset to create. Either
        "COVID" or "flu".
    nhsn_dataset_path
        Path to the NHSN dataset (csv file).
    file_save_path
        The path for where to save the output file.
    """
    # check that dataset parameter is possible
    assert dataset in [
        "COVID",
        "flu",
    ], 'Dataset {dataset} must be one of "COVID", "flu"'
    # check the file path is valid
    check_file_save_path(file_save_path)
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
        if not set(required_cols).issubset(set(df_cols)):
            raise ValueError(
                f"NHSN dataset missing required columns: {set(required_cols) - set(df_cols)}"
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
                    {"previous_day_admission_adult_covid_confirmed": "hosp"}
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
                .rename({"previous_day_admission_influenza_confirmed": "hosp"})
                .sort(["state", "date"])
            )
            df_flu.write_csv(file_save_path)
        print(f"The file {file_save_path} has been created.")


def get_and_save_flusight_submission(
    file_save_path: str,
) -> None:
    """
    Saves an example FluSight submission as a csv.

    file_save_path
        The path for where to save the output file.
    """
    # check if the save file exists
    check_file_save_path(file_save_path)
    # check if the FluSight example url is still valid
    url = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/cfa-flumech/2023-10-14-cfa-flumech.csv"
    check_url(url)
    # read csv from URL, convert to polars
    submission_df = pl.read_csv(url, infer_schema_length=7500)
    # save the dataframe
    submission_df.write_csv(file_save_path)
    print(f"The file {file_save_path} has been saved.")
