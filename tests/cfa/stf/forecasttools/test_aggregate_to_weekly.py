"""
Tests for daily_to_epiweekly.py functions.
"""

import datetime

import polars as pl
import pytest

from cfa.stf.forecasttools import (
    ceiling_isoweek,
    ceiling_mmwr_epiweek,
    ceiling_week,
    daily_to_weekly,
    floor_isoweek,
    floor_mmwr_epiweek,
    floor_week,
)


@pytest.mark.parametrize(
    ("standard", "start_date", "expected_week_start_date", "expected_week_end_date"),
    [
        (
            "MMWR",
            datetime.date(2025, 10, 4),
            datetime.date(2025, 10, 5),
            datetime.date(2025, 10, 11),
        ),
        (
            "ISO",
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
        ("MMWR", datetime.date(2015, 12, 31), 2015, 52),
        ("ISO", datetime.date(2015, 12, 31), 2015, 53),
        ("USA", datetime.date(2016, 1, 1), 2015, 52),
        ("iso", datetime.date(2016, 1, 1), 2015, 53),
        ("mmwr", datetime.date(2016, 1, 2), 2015, 52),
        ("ISO", datetime.date(2016, 1, 2), 2015, 53),
        ("USA", datetime.date(2016, 1, 3), 2016, 1),
        ("ISO", datetime.date(2016, 1, 3), 2015, 53),
        ("MMWR", datetime.date(2021, 1, 3), 2021, 1),
        ("ISO", datetime.date(2021, 1, 3), 2020, 53),
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


@pytest.mark.parametrize(
    ("date_input", "expected_floor", "expected_ceiling"),
    [
        (
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 15),
        ),
        (
            datetime.date(2026, 3, 12),
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 15),
        ),
        (
            datetime.date(2026, 3, 15),
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 15),
        ),
    ],
)
def test_iso_week_floor_ceiling(date_input, expected_floor, expected_ceiling):
    """Test floor_isoweek and ceiling_isoweek with ISO standard (Monday start)."""
    assert floor_isoweek(date_input) == expected_floor
    assert ceiling_isoweek(date_input) == expected_ceiling


@pytest.mark.parametrize(
    ("date_input", "expected_floor", "expected_ceiling"),
    [
        (
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
        (
            datetime.date(2026, 3, 12),
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
        (
            datetime.date(2026, 3, 14),
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
    ],
)
def test_mmwr_week_floor_ceiling(date_input, expected_floor, expected_ceiling):
    """Test floor_mmwr_epiweek and ceiling_mmwr_epiweek with MMWR standard (Sunday start)."""
    assert floor_mmwr_epiweek(date_input) == expected_floor
    assert ceiling_mmwr_epiweek(date_input) == expected_ceiling


@pytest.mark.parametrize(
    ("date_input", "standard", "expected_floor", "expected_ceiling"),
    [
        (
            datetime.date(2026, 3, 12),
            "iso",
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 15),
        ),
        (
            datetime.date(2026, 3, 12),
            "ISO",
            datetime.date(2026, 3, 9),
            datetime.date(2026, 3, 15),
        ),
        (
            datetime.date(2026, 3, 12),
            "USA",
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
        (
            datetime.date(2026, 3, 12),
            "MMWR",
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
        (
            datetime.date(2026, 3, 12),
            "mmwr",
            datetime.date(2026, 3, 8),
            datetime.date(2026, 3, 14),
        ),
    ],
)
def test_floor_ceiling_week_with_standard(
    date_input, standard, expected_floor, expected_ceiling
):
    """Test floor_week and ceiling_week with various standards (case-insensitive)."""
    assert floor_week(date_input, standard=standard) == expected_floor
    assert ceiling_week(date_input, standard=standard) == expected_ceiling
