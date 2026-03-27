"""
Tests for daily_to_epiweekly.py functions.
"""

import datetime

import polars as pl

import forecasttools


def test_df_aggregate_to_epiweekly_adds_epiweek_end_date():
    """
    Test aggregated output includes Saturday end date
    for each epiweek.
    """
    df = pl.DataFrame(
        {
            "location": ["TX"] * 8,
            "date": [
                datetime.date(2023, 10, 8) + datetime.timedelta(days=i)
                for i in range(8)
            ],
            "value": [1, 1, 1, 1, 1, 1, 1, 4],
        }
    )

    out = forecasttools.df_aggregate_to_epiweekly(
        df=df,
        id_cols=["location"],
        with_epiweek_end_date=True,
    )

    assert "epiweek_end_date" in out.columns

    assert out.get_column("epiweek_end_date").item() == datetime.date(2023, 10, 14)

    assert out.get_column("weekly_value").item() == 7
