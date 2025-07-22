from datetime import date

import polars as pl
from cfasodapy import Query

from forecasttools.utils import ensure_listlike

# Dataset IDs and metadata for commonly
# used (within STF) data.cdc.gov datasets
data_cdc_gov_datasets = pl.DataFrame(
    {
        "key": ["nhsn_hrd_prelim", "nhsn_hrd_final", "nssp_prop_ed_visits"],
        "id": ["mpgq-jmmr", "ua7e-t2fy", "rdmq-nq56"],
        "date_column": ["weekendingdate", "weekendingdate", "week_end"],
        "location_column": ["jurisdiction", "jurisdiction", "county"],
    }
)


def get_dataset_info(dataset_key: str) -> dict:
    """Look up dataset information by key."""
    filtered = data_cdc_gov_datasets.filter(pl.col("key") == dataset_key)
    if filtered.height == 0:
        raise ValueError(
            f"Dataset key '{dataset_key}' not found in dataset table"
        )
    return filtered.row(0, named=True)


def _parse_comma_separated_values(cs_string: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [value.strip() for value in cs_string.split(",")]


def get_data_cdc_gov_dataset(
    dataset_key: str,
    start_date: str | date = None,
    end_date: str | date = None,
    additional_col_names: str | list[str] = None,
    locations: str | list[str] = None,
    limit: int = 10000,
    app_token: str = None,
) -> pl.DataFrame:
    """
    Pull data from data.cdc.gov using dataset key.

    Parameters
    -----------
    dataset_key
        Key identifying the dataset. Must be one of
        the keys in `forecasttools.data_cdc_gov_datasets`.
    start_date
        Start date for data to pull in YYYY-MM-DD format
        string or datetime.date object.
    end_date
        End date for data to pull in YYYY-MM-DD format
        string or datetime.date object.
    additional_col_names
        Columns to select in addition to date
        and location columns. Can be a single column name
        string, comma-separated names or a list of column
        names. If None, only date and location columns are
        selected. Defaults to None.
    locations
        Location(s) to filter on the location column.
        Can be a single location string, comma-separated
        string, or list of locations. If None, all
        locations are included. Defaults to None.
    limit
        Maximum number of rows to return.
        Defaults to 10,000.
    app_token
        Socrata app token for authentication

    Returns
    --------
    pl.DataFrame
        A polars DataFrame with the requested data
    """
    dataset_info = get_dataset_info(dataset_key)

    domain = "data.cdc.gov"
    dataset_id = dataset_info["id"]
    date_col = dataset_info["date_column"]
    location_col = dataset_info["location_column"]

    where_clauses = []
    if start_date:
        where_clauses.append(f"{date_col} >= '{start_date}'")
    if end_date:
        where_clauses.append(f"{date_col} <= '{end_date}'")
    if locations:
        locations = ensure_listlike(locations)
        if len(locations) == 1:
            locations = _parse_comma_separated_values(locations[0])
        locations_str = "', '".join(locations)
        where_clauses.append(f"{location_col} IN ('{locations_str}')")

    where = " AND ".join(where_clauses) if where_clauses else None

    select = [date_col, location_col]
    if additional_col_names:
        additional_col_names = ensure_listlike(additional_col_names)
        select += [
            col
            for col in additional_col_names
            if col not in [date_col, location_col]
        ]

    q = Query(
        domain=domain,
        id=dataset_id,
        where=where,
        select=select,
        limit=limit,
        app_token=app_token,
    )

    data = pl.from_dicts(q.get_all())
    return data


def get_nhsn(
    start_date: date = None,
    end_date: date = None,
    app_token: str = None,
    dataset_key: str = "nhsn_hrd_prelim",
    additional_col_names: str | list[str] = "totalconfc19newadm",
    locations: str | list[str] = None,
    limit: int = 10000,
) -> pl.DataFrame:
    """
    Get NHSN Hospital Respiratory Data.
    """
    return get_data_cdc_gov_dataset(
        dataset_key=dataset_key,
        start_date=start_date,
        end_date=end_date,
        additional_col_names=additional_col_names,
        app_token=app_token,
        locations=locations,
        limit=limit,
    )
