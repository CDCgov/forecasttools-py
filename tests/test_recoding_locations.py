"""
Test file for functions contained
within recode_locations.py
"""

import polars as pl

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
            "U.S. Virgin Islands",
            "Texas",
            "United States",
        ]
    }
)

EXPECTED_LONG = [
    "Alabama",
    "Alaska",
    "U.S. Virgin Islands",
    "Texas",
    "United States",
]


# missing column name e.g. "non_existent_col"
# invalid location in list
# empty dataframe


def test_loc_hubverse_code_abbr():
    # get an example dataframe w/ location column
    df_w_loc_as_codes = forecasttools.loc_hubverse_code_to_abbr(
        df=SAMPLE_CODE_LOC_DF, location_col="location"
    )
    # check that location abbreviations outputted are correct
    loc_abbrs_out = df_w_loc_as_codes["location"].to_list()
    assert (
        loc_abbrs_out == EXPECTED_ABBRS
    ), f"Expected {EXPECTED_ABBRS}, Got: {loc_abbrs_out}"


def test_loc_abbr_to_hubverse_code():
    # get an example dataframe w/ location column
    df_w_loc_as_abbr = forecasttools.loc_abbr_to_hubverse_code(
        df=SAMPLE_ABBR_LOC_DF, location_col="location"
    )
    # check that location codes outputted are correct
    loc_codes_out = df_w_loc_as_abbr["location"].to_list()
    assert (
        loc_codes_out == EXPECTED_CODES
    ), f"Expected {EXPECTED_CODES}, Got: {loc_codes_out}"
