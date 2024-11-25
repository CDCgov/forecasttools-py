"""
Test file for functions contained
within recode_locations.py
"""

import polars as pl
import pytest

import forecasttools


@pytest.mark.parametrize(
    "function, df, location_col, expected_output",
    [
        (
            forecasttools.loc_abbr_to_hubverse_code,
            pl.DataFrame({"location": ["AL", "AK", "CA", "TX", "US"]}),
            "location",
            ["01", "02", "06", "48", "US"],
        ),
        (
            forecasttools.loc_hubverse_code_to_abbr,
            pl.DataFrame({"location": ["01", "02", "06", "48", "US"]}),
            "location",
            ["AL", "AK", "CA", "TX", "US"],
        ),
    ],
)
def test_recode_valid_location_correct_input(
    function, df, location_col, expected_output
):
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
            pl.DataFrame(),
            "location",  # empty df failure
            ValueError,
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
            pl.DataFrame(),
            "location",  # empty df failure
            ValueError,
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
def test_loc_conversation_funcs_invalid_input(
    function, df, location_col, expected_exception
):
    """
    Test that loc_hubverse_code_to_abbr and
    loc_abbr_to_hubverse_code handle type
    errors for the dataframe and location
    column name, value errors for the
    location entries, and value errors if the
    dataframe is empty.
    """
    with pytest.raises(expected_exception):
        function(df, location_col)


@pytest.mark.parametrize(
    "location_format, expected_column",
    [
        ("abbr", "short_name"),
        ("hubverse", "location_code"),
        ("long_name", "long_name"),
    ],
)
def test_to_location_table_column_correct_input(
    location_format, expected_column
):
    """
    Test to_location_table_column for
    expected column names
    when given different location formats.
    """
    result_column = forecasttools.to_location_table_column(location_format)
    assert (
        result_column == expected_column
    ), f"Expected column '{expected_column}' for format '{location_format}', but got '{result_column}'"


@pytest.mark.parametrize(
    "location_format, expected_exception",
    [
        (123, AssertionError),  # invalid location type
        ("unknown_format", KeyError),  # bad location name
    ],
)
def test_to_location_table_column_exception_handling(
    location_format, expected_exception
):
    """
    Test to_location_table_column for
    exception handling.
    """
    with pytest.raises(expected_exception):
        forecasttools.to_location_table_column(location_format)


@pytest.mark.parametrize(
    "location_vector, location_format, expected_exception",
    [
        ("invalid_string", "abbr", TypeError),  # invalid location vec type
        ([1, 2, 3], "abbr", TypeError),  # non-string elts in location vec
        (
            ["AL", "CA"],
            123,
            TypeError,
        ),  # invalid location format type (not str)
        (
            ["AL", "CA"],
            "invalid_format",
            ValueError,
        ),  # invalid location_format value (not one of valid)
        ([], "abbr", ValueError),  # empty location_vector (edge)
        (["AL", "CA"], "abbr", None),  # valid inputs (expected no exception)
    ],
)
def test_location_lookup_exceptions(
    location_vector, location_format, expected_exception
):
    """
    Test location_lookup for exception handling
    and input validation.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            forecasttools.location_lookup(location_vector, location_format)
    else:
        result = forecasttools.location_lookup(
            location_vector, location_format
        )
        assert isinstance(
            result, pl.DataFrame
        ), "Expected a Polars DataFrame as output."
