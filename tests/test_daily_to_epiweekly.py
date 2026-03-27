"""
Tests for daily_to_epiweekly.py functions.
"""

import polars as pl

import forecasttools


def test_df_aggregate_to_epiweekly_adds_epiweek_end_date():
    """
    Test aggregated output includes Saturday end date
    for each epiweek.
    """
    df = pl.DataFrame(
        {
            "location": ["TX", "TX", "TX", "TX", "TX", "TX", "TX", "TX"],
            "date": [
                "2023-10-08",
                "2023-10-09",
                "2023-10-10",
                "2023-10-11",
                "2023-10-12",
                "2023-10-13",
                "2023-10-14",
                "2023-10-15",
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

    assert out.get_column("epiweek_end_date").item() == "2023-10-14"

    assert out.get_column("weekly_value").item() == 7
