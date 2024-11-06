"""
Test file for functions contained
within recode_locations.py
"""

import polars as pl
import pytest

import forecasttools

SAMPLE_ABBR_LOC_DF = pl.DataFrame({"location": ["AL", "AK", "CA", "TX", "US"]})
EXPECTED_CODES = ["01", "02", "06", "48", "US"]
SAMPLE_CODE_LOC_DF = pl.DataFrame({"location": ["01", "02", "06", "48", "US"]})
EXPECTED_ABBRS = ["AL", "AK", "CA", "TX", "US"]
SAMPLE_LONG_LOC_DF = pl.DataFrame(
    {
        "location": [
            "Alabama",
            "Alaska",
            "California",
            "Texas",
            "United States",
        ]
    }
)

EXPECTED_LONG = [
    "Alabama",
    "Alaska",
    "California",
    "Texas",
    "United States",
]


@pytest.mark.parametrize(
    "function, df, location_col, expected_output",
    [
        (
            forecasttools.loc_abbr_to_hubverse_code,
            SAMPLE_ABBR_LOC_DF,
            "location",
            EXPECTED_CODES,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            SAMPLE_CODE_LOC_DF,
            "location",
            EXPECTED_ABBRS,
        ),
    ],
)
def test_recode_valid_locations(function, df, location_col, expected_output):
    """
    Test both recode functions (loc_abbr_to_hubverse_code
    and loc_hubverse_code_to_abbr) for valid
    location code and abbreviation output.
    """
    df_w_loc_recoded = function(df=df, location_col=location_col)
    loc_output = df_w_loc_recoded["location"].to_list()
    assert (
        loc_output == expected_output
    ), f"Expected {expected_output}, Got: {loc_output}"


@pytest.mark.parametrize(
    "function, df, location_col, invalid_location, expected_output",
    [
        (
            forecasttools.loc_abbr_to_hubverse_code,
            SAMPLE_ABBR_LOC_DF,
            "location",
            "XX",
            EXPECTED_CODES,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            SAMPLE_CODE_LOC_DF,
            "location",
            "99",
            EXPECTED_ABBRS,
        ),
    ],
)
def test_recode_invalid_location(
    function, df, location_col, invalid_location, expected_output
):
    """
    Test recode functions with invalid locations
    and ensure they are handled gracefully.
    """
    df_with_invalid = df.with_columns(
        pl.lit(invalid_location).alias(location_col)
    )
    df_w_loc_recoded = function(df=df_with_invalid, location_col=location_col)
    loc_output = df_w_loc_recoded["location"].to_list()
    print(loc_output)
    # assert loc_output == expected_output, \
    #     f"Expected {expected_output}, Got: {loc_output}"
