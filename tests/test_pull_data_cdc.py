# test_pull_data_cdc.py

from datetime import date
from unittest.mock import MagicMock, call, patch

import forecasttools


@patch("forecasttools.pull_data_cdc_gov.Query")
def test_get_data_cdc_gov_dataset_basic(mocker):
    mock_instance = MagicMock()
    mock_instance.get_all.return_value = [
        {
            "weekendingdate": "2024-11-02",
            "jurisdiction": "US",
            "value": "100",
            "percent": ".21",
        },
        {
            "weekendingdate": "2024-11-09",
            "jurisdiction": "US",
            "value": "95",
            "percent": ".19",
        },
        {
            "weekendingdate": "2024-11-02",
            "jurisdiction": "CA",
            "value": "10",
            "percent": ".12",
        },
        {
            "weekendingdate": "2024-11-09",
            "jurisdiction": "CA",
            "value": "9",
            "percent": ".31",
        },
    ]
    mocker.return_value = mock_instance

    forecasttools.get_data_cdc_gov_dataset(
        dataset_key="nhsn_hrd_prelim",
        start_date=date(2024, 11, 2),
        end_date=date(2024, 11, 9),
        additional_col_names="value",
    )
    forecasttools.get_data_cdc_gov_dataset(
        dataset_key="nhsn_hrd_prelim",
        start_date=date(2024, 11, 2),
        end_date=date(2024, 11, 9),
        additional_col_names="percent",
        locations=["CA", "US"],
        limit=1,
    )
    forecasttools.get_data_cdc_gov_dataset(
        dataset_key="nhsn_hrd_prelim",
    )

    assert mocker.call_count == 3

    expected_call_all = call(
        domain="data.cdc.gov",
        id="mpgq-jmmr",
        where=("weekendingdate >= '2024-11-02' AND weekendingdate <= '2024-11-09'"),
        select=["weekendingdate", "jurisdiction", "value"],
        limit=10000,
        app_token=None,
    )
    expected_call_ca = call(
        domain="data.cdc.gov",
        id="mpgq-jmmr",
        where=(
            "weekendingdate >= '2024-11-02' "
            "AND weekendingdate <= '2024-11-09' "
            "AND jurisdiction IN ('CA', 'US')"
        ),
        select=["weekendingdate", "jurisdiction", "percent"],
        limit=1,
        app_token=None,
    )
    expected_call_default = call(
        domain="data.cdc.gov",
        id="mpgq-jmmr",
        where=None,
        select=["weekendingdate", "jurisdiction"],
        limit=10000,
        app_token=None,
    )

    assert mocker.call_args_list == [
        expected_call_all,
        expected_call_ca,
        expected_call_default,
    ]
