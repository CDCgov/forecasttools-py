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
    "function, df, location_col, expected_exception",
    [
        (
            forecasttools.loc_abbr_to_hubverse_code,
            "not_a_dataframe",  # not a dataframe type error
            "location_col",
            TypeError,
        ),
        (
            forecasttools.loc_abbr_to_hubverse_code,
            pl.DataFrame({"location": ["AL", "AK"]}),
            123,  # location column type failure
            TypeError,
        ),
        (
            forecasttools.loc_abbr_to_hubverse_code,
            pl.DataFrame({"location": ["AL", "AK"]}),
            "non_existent_col",  # location column name failure
            ValueError,
        ),
        (
            forecasttools.loc_abbr_to_hubverse_code,
            pl.DataFrame({"location": ["XX"]}),  # abbr value failure
            "location",
            ValueError,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            "not_a_dataframe",  # not a dataframe type error
            "location_col",
            TypeError,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            pl.DataFrame({"location": ["01", "02"]}),
            123,  # location column type failure
            TypeError,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            pl.DataFrame({"location": ["01", "02"]}),
            "non_existent_col",  # location column name failure
            ValueError,
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            pl.DataFrame({"location": ["99"]}),  # code value failure
            "location",
            ValueError,
        ),
    ],
)
def test_invalid_input_handling(
    function, df, location_col, expected_exception
):
    with pytest.raises(expected_exception):
        function(df, location_col)
