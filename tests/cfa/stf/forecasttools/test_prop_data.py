import polars as pl
import polars.testing as plt
import pytest

import cfa.stf.forecasttools as ft


def test_append_prop_data_appends_proportion_rows():
    data = pl.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
            ],
            "location": ["US", "US", "US", "US", "US"],
            ".variable": [
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "some_other_variable",
            ],
            ".value": [70, 20, 80, 30, 999],
        }
    )

    result = ft.append_prop_data(data)

    expected = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "location": ["US", "US", "US", "US", "US", "US", "US"],
            ".variable": [
                "observed_ed_visits",
                "other_ed_visits",
                "prop_disease_ed_visits",
                "some_other_variable",
                "observed_ed_visits",
                "other_ed_visits",
                "prop_disease_ed_visits",
            ],
            ".value": [20.0, 80.0, 0.2, 999.0, 30.0, 70.0, 0.3],
        }
    )
    plt.assert_frame_equal(result, expected)


def test_append_prop_data_preserves_additional_identifier_columns():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            "location": ["CA", "CA", "NY", "NY"],
            "age_group": ["all", "all", "all", "all"],
            ".variable": [
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
            ],
            ".value": [1, 3, 4, 6],
        }
    )

    result = ft.append_prop_data(data).filter(
        pl.col(".variable") == "prop_disease_ed_visits"
    )

    expected = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "location": ["CA", "NY"],
            "age_group": ["all", "all"],
            ".variable": ["prop_disease_ed_visits", "prop_disease_ed_visits"],
            ".value": [0.25, 0.4],
        }
    )
    plt.assert_frame_equal(result, expected)


def test_append_prop_data_errors_when_required_column_is_missing():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01"],
            ".variable": ["observed_ed_visits"],
        }
    )

    with pytest.raises(ValueError, match=r"\.value"):
        ft.append_prop_data(data)
