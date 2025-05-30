"""
Tests adding dates to idata objects. Items
that can be tested include: invalid
start_iso_date str or datetime, invalid idata
object, invalid group name, invalid variable
name, invalid dimension name, or invalid time
step, where invalid usually means a Value
or Type error can be produced.
"""

from datetime import date, datetime, timedelta

import arviz as az
import numpy as np
import pytest
import xarray as xr

import forecasttools

IDATA_WO_DATES = forecasttools.nhsn_flu_forecast_wo_dates


@pytest.mark.parametrize(
    "start_date_iso, time_step, expected_error",
    [
        (date(2022, 8, 1), timedelta(days=1), None),  # valid case, str date
        (
            date(2022, 8, 1),
            timedelta(days=1),
            None,
        ),  # valid case, datetime date
        ("invalid-date", timedelta(days=1), TypeError),  # invalid date string
        (
            date(2022, 8, 1),
            "invalid-timedelta",
            TypeError,
        ),  # invalid time step
        (
            date(2022, 8, 1),
            timedelta(days=0),
            ValueError,
        ),  # time_step can't be 0
    ],
)
def test_add_time_coords_to_idata_dimension(
    start_date_iso, time_step, expected_error
):
    """
    Tests instances where invalid start_iso_date
    (as str and datetime dates)
    an invalid case with str date, and
    invalid time step cases.
    """
    group = "posterior_predictive"
    variable = "obs"
    dimension = "obs_dim_0"

    # test raises expected error or works
    if expected_error:
        with pytest.raises(expected_error):
            forecasttools.add_time_coords_to_idata_dimension(
                IDATA_WO_DATES,
                group,
                variable,
                dimension,
                start_date_iso,
                time_step,
            )
    else:
        idata = forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            group,
            variable,
            dimension,
            start_date_iso,
            time_step,
        )
        # dim correctly updated?
        assert dimension in idata.posterior_predictive[variable].coords
        assert (
            len(idata.posterior_predictive[variable].coords[dimension])
            == idata.posterior_predictive[variable].sizes[dimension]
        )


@pytest.mark.parametrize(
    "start_date_iso, time_step",
    [
        (date(2022, 8, 1), timedelta(days=1)),  # datetime input
        (date(2022, 8, 1), timedelta(days=1)),  # string input
    ],
)
def test_start_date_as_str_or_datetime(start_date_iso, time_step):
    idata = forecasttools.add_time_coords_to_idata_dimension(
        IDATA_WO_DATES,
        "posterior_predictive",
        "obs",
        "obs_dim_0",
        start_date_iso,
        time_step,
    )
    assert "obs_dim_0" in idata.posterior_predictive["obs"].coords
    assert (
        len(idata.posterior_predictive["obs"].coords["obs_dim_0"])
        == idata.posterior_predictive["obs"].sizes["obs_dim_0"]
    )


@pytest.mark.parametrize(
    "idata, group, variable, dimension, time_step, expected_error",
    [
        (
            IDATA_WO_DATES,
            "observed_data",
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            None,
        ),  # different valid
        # valid cases
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            timedelta(weeks=1),  # fail when 1?
            None,
        ),  # valid
        # invalid idata
        (
            [],
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # invalid idata: list instead of InferenceData
        (
            "string",
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # invalid idata: string
        # invalid group
        (
            IDATA_WO_DATES,
            123,
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # invalid group: int instead of str
        (
            IDATA_WO_DATES,
            None,
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # invalid group: None instead of str
        # invalid variable
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            123,
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # invalid variable: int instead of str
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            None,
            "obs_dim_0",
            timedelta(days=1),
            TypeError,
        ),  # Invalid variable: None instead of str
        # invalid dimension
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            123,
            timedelta(days=1),
            TypeError,
        ),  # invalid dimension: int instead of str
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            None,
            timedelta(days=1),
            TypeError,
        ),  # invalid dimension: None instead of str
        # invalid time_step
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            "1 day",
            TypeError,
        ),  # invalid time_step: string instead of timedelta
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            123,
            TypeError,
        ),  # invalid time_step: int instead of timedelta
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            timedelta(days=0),
            ValueError,
        ),  # invalid time_step: 0 days
    ],
)
def test_input_types_add_coords(
    idata, group, variable, dimension, time_step, expected_error
):
    """
    Tests the validation of input types for
    the add_time_coords_to_idata_dimension
    function.
    """

    # check for expected error or
    # for successful execution
    if expected_error:
        with pytest.raises(expected_error):
            forecasttools.add_time_coords_to_idata_dimension(
                idata=idata,
                group=group,
                variable=variable,
                dimension=dimension,
                start_date_iso=date(2022, 8, 1),
                time_step=time_step,
            )
    else:
        # if no error is expected,
        # execute the function
        old_dim = idata[group][variable][dimension].values
        idata_out = forecasttools.add_time_coords_to_idata_dimension(
            idata=idata,
            group=group,
            variable=variable,
            dimension=dimension,
            start_date_iso=date(2022, 8, 1),
            time_step=time_step,
        )

        new_dim = idata_out[group][variable][dimension].values
        assert not np.array_equal(old_dim, new_dim)
        # validate that the function executed
        # correctly and the dimension has been modified
        assert dimension in idata_out.posterior_predictive[variable].coords
        assert (
            len(idata_out.posterior_predictive[variable].coords[dimension])
            == idata_out.posterior_predictive[variable].sizes[dimension]
        )


def test_dates_add_time_coords_to_idata_dimension():
    # set up test data
    coords = {"obs_dim_0": [0, 1, 2, 3, 5]}
    data = xr.Dataset(
        {"obs": ("obs_dim_0", np.array([10, 20, 30, 40, 50]))}, coords=coords
    )
    idata = az.InferenceData(observed_data=data)

    # test parameters
    group = "observed_data"
    variable = "obs"
    dimension = "obs_dim_0"
    start_date_iso = date(2024, 11, 19)
    time_step = timedelta(days=2)

    # expected output dates
    expected_dates = np.array(
        [
            date(2024, 11, 19),
            date(2024, 11, 21),
            date(2024, 11, 23),
            date(2024, 11, 25),
            date(2024, 11, 27),
        ]
    ).astype("datetime64[ns]")

    # function call
    updated_idata = forecasttools.add_time_coords_to_idata_dimension(
        idata=idata,
        group=group,
        variable=variable,
        dimension=dimension,
        start_date_iso=start_date_iso,
        time_step=time_step,
    )

    # extract the updated time coordinates
    updated_coords = updated_idata.observed_data.coords[dimension].values

    # assert that the updated coordinates are equal to the expected dates
    np.testing.assert_array_equal(updated_coords, expected_dates)


def test_datetimes_add_time_coords_to_idata_dimension():
    # set up test data
    coords = {"obs_dim_0": [0, 1, 2, 3, 5]}
    data = xr.Dataset(
        {"obs": ("obs_dim_0", np.array([10, 20, 30, 40, 50]))}, coords=coords
    )
    idata = az.InferenceData(observed_data=data)

    # test parameters
    group = "observed_data"
    variable = "obs"
    dimension = "obs_dim_0"
    start_date = datetime(2022, 8, 1, 12)
    time_step = timedelta(days=1.5)

    # expected output datetimes
    expected_datetimes = np.array(
        [
            datetime(2022, 8, 1, 12),
            datetime(2022, 8, 3, 0),
            datetime(2022, 8, 4, 12),
            datetime(2022, 8, 6, 0),
            datetime(2022, 8, 7, 12),
        ]
    ).astype("datetime64[ns]")

    # function call
    updated_idata = forecasttools.add_time_coords_to_idata_dimension(
        idata=idata,
        group=group,
        variable=variable,
        dimension=dimension,
        start_date_iso=start_date,
        time_step=time_step,
    )

    # extract the updated time coordinates
    updated_coords = updated_idata.observed_data.coords[dimension].values

    print(updated_coords)

    # assert that the updated coordinates are equal to the expected datetimes
    np.testing.assert_array_equal(updated_coords, expected_datetimes)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("group1", ["group1"]),  # input is string, should be converted to list
        (
            ["group1", "group2"],
            ["group1", "group2"],
        ),  # input is list, should stay same
        ("single", ["single"]),  # single string input
        ([], []),  # empty list should remain empty
    ],
)
def test_ensure_listlike(input_value, expected_output):
    """
    Test that ensure_listlike converts a
    string to a list and leaves list
    unchanged.
    """
    out = forecasttools.ensure_listlike(input_value)
    assert out == expected_output, (
        f"Expected {expected_output}, but got {out}."
    )


@pytest.mark.parametrize(
    "input_value, expected_error, param_name",
    [
        (["group1", "group2"], None, "groups"),  # valid, list of strings
        ("group1", None, "groups"),  # valid, single string
        (["group1", 123], TypeError, "groups"),  # invalid, non-string in list
        (123, TypeError, "groups"),  # invalid, non-string in single value
        (["var1", "var2"], None, "variables"),  # valid, list of strings
        ("var1", None, "variables"),  # valid, single string
        (["var1", 123], TypeError, "variables"),  # invalid, non-string in list
        (["dim1", "dim2"], None, "dimensions"),  # valid, list of strings
        ("dim1", None, "dimensions"),  # valid, single string
        (
            ["dim1", 123],
            TypeError,
            "dimensions",
        ),  # invalid, non-string in list
    ],
)
def test_validate_iter_has_expected_types(
    input_value, expected_error, param_name
):
    """
    Test that validate_iter_has_expected_types
    properly validates that all entries in
    groups, variables, and dimensions are
    strings.
    """
    if expected_error:
        with pytest.raises(expected_error):
            forecasttools.validate_iter_has_expected_types(
                input_value, str, param_name
            )
    else:
        forecasttools.validate_iter_has_expected_types(
            input_value, str, param_name
        )


@pytest.mark.parametrize(
    "idata, groups, variables, dimensions, expected_error",
    [
        # valid
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            None,
        ),  # all valid types
        (
            IDATA_WO_DATES,
            ["posterior_predictive"],
            ["obs"],
            ["obs_dim_0"],
            None,
        ),  # all valid types in lists
        # invalid cases
        (
            [],
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            TypeError,
        ),  # invalid idata: list instead of InferenceData
        (
            "string",
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            TypeError,
        ),  # invalid idata: string instead of InferenceData
        (
            IDATA_WO_DATES,
            123,
            "obs",
            "obs_dim_0",
            TypeError,
        ),  # invalid groups: int instead of str or list[str]
        (
            IDATA_WO_DATES,
            None,
            "obs",
            "obs_dim_0",
            TypeError,
        ),  # invalid groups: None
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            123,
            "obs_dim_0",
            TypeError,
        ),  # invalid variables: int instead of str or list[str]
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            None,
            "obs_dim_0",
            TypeError,
        ),  # invalid variables: None
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            123,
            TypeError,
        ),  # invalid dimensions: int instead of str or list[str]
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            None,
            TypeError,
        ),  # invalid dimensions: None
    ],
)
def test_validate_input_types_in_add_time_coords_to_idata_dimensions(
    idata, groups, variables, dimensions, expected_error
):
    """
    Tests all the validate_input_type calls
    within the add_time_coords_to_idata_dimensions
    function.
    """
    if expected_error:
        with pytest.raises(expected_error):
            forecasttools.add_time_coords_to_idata_dimensions(
                idata=idata,
                groups=groups,
                variables=variables,
                dimensions=dimensions,
                start_date_iso=date(2022, 8, 1),
                time_step=timedelta(days=1),
            )
    else:
        forecasttools.add_time_coords_to_idata_dimensions(
            idata=idata,
            groups=groups,
            variables=variables,
            dimensions=dimensions,
            start_date_iso=date(2022, 8, 1),
            time_step=timedelta(days=1),
        )
