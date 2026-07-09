"""
Tests for daily_to_epiweekly.py functions.
"""

import datetime

import polars as pl
import pytest

from cfa.stf.forecasttools import daily_to_weekly


@pytest.mark.parametrize(
    ("standard", "start_date", "expected_week_start_date", "expected_week_end_date"),
    [
        (
            "epiweek",
            datetime.date(2025, 10, 4),
            datetime.date(2025, 10, 5),
            datetime.date(2025, 10, 11),
        ),
        (
            "isoweek",
            datetime.date(2025, 10, 4),
            datetime.date(2025, 10, 6),
            datetime.date(2025, 10, 12),
        ),
    ],
)
def test_daily_to_weekly_supports_standards(
    standard: str,
    start_date: datetime.date,
    expected_week_start_date: datetime.date,
    expected_week_end_date: datetime.date,
):
    """Test aggregation behavior for each supported weekly standard."""
    df = pl.DataFrame(
        {
            "location": ["TX"] * 10,
            "date": [start_date + datetime.timedelta(days=i) for i in range(10)],
            "value": [1] * 10,
        }
    )

    out = daily_to_weekly(
        df=df,
        id_cols=["location"],
        standard=standard,
        with_week_start_date=True,
        with_week_end_date=True,
    )

    assert out.get_column("week").item() == 41
    assert out.get_column("weekyear").item() == 2025
    assert out.get_column("week_start_date").item() == expected_week_start_date
    assert out.get_column("week_end_date").item() == expected_week_end_date
    assert out.get_column("weekly_value").item() == 7


def test_daily_to_weekly_invalid_standard_raises():
    """Unknown standard values should be rejected."""
    df = pl.DataFrame(
        {
            ".draw": [0] * 7,
            "date": [
                datetime.date(2025, 10, 6) + datetime.timedelta(days=i)
                for i in range(7)
            ],
            "value": [1] * 7,
        }
    )

    with pytest.raises(ValueError, match="standard"):
        daily_to_weekly(df=df, standard="mystandard")


@pytest.mark.parametrize(
    ("standard", "date_input", "expected_weekyear", "expected_week"),
    [
        ("epiweek", datetime.date(2015, 12, 31), 2015, 52),
        ("isoweek", datetime.date(2015, 12, 31), 2015, 53),
        ("epiweek", datetime.date(2016, 1, 1), 2015, 52),
        ("isoweek", datetime.date(2016, 1, 1), 2015, 53),
        ("epiweek", datetime.date(2016, 1, 2), 2015, 52),
        ("isoweek", datetime.date(2016, 1, 2), 2015, 53),
        ("epiweek", datetime.date(2016, 1, 3), 2016, 1),
        ("isoweek", datetime.date(2016, 1, 3), 2015, 53),
        ("epiweek", datetime.date(2021, 1, 3), 2021, 1),
        ("isoweek", datetime.date(2021, 1, 3), 2020, 53),
    ],
)
def test_calculate_week_and_year_boundary_diverge(
    standard: str,
    date_input: datetime.date,
    expected_weekyear: int,
    expected_week: int,
):
    df = pl.DataFrame(
        {
            "location": ["TX"],
            "date": [date_input],
            "value": [1],
        }
    )

    out = daily_to_weekly(
        df=df,
        id_cols=["location"],
        standard=standard,
        strict=False,
    )

    assert out.get_column("week").item() == expected_week
    assert out.get_column("weekyear").item() == expected_weekyear
