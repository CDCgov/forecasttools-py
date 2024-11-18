"""
Tests adding dates to idata objects. Items
that can be tested include: invalid
start_iso_date str or datetime, invalid idata
object, invalid group name, invalid variable
name, invalid dimension name, or invalid time
step, where invalid usually means a Value
or Type error can be produced.
"""

from datetime import datetime, timedelta

import pytest

import forecasttools

IDATA_WO_DATES = forecasttools.nhsn_flu_forecast_wo_dates


@pytest.mark.parametrize(
    "start_date_iso, time_step, expected_error",
    [
        ("2022-08-01", timedelta(days=1), None),  # valid case, str date
        (
            datetime(2022, 8, 1),
            timedelta(days=1),
            None,
        ),  # valid case, datetime date
        ("invalid-date", timedelta(days=1), ValueError),  # invalid date string
        ("2022-08-01", "invalid-timedelta", TypeError),  # invalid time step
        (
            datetime(2022, 8, 1),
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


def test_invalid_group_name():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "invalid_group",
            "obs",
            "obs_dim_0",
            "2022-08-01",
            timedelta(days=1),
        )
    # with pytest.raises(ValueError):
    #     forecasttools.add_time_coords_to_idata_dimensions(
    #         idata=IDATA_WO_DATES,
    #         "invalid_group",
    #         "obs",
    #         "obs_dim_0",
    #         "2022-08-01",
    #         timedelta(days=1),
    #     )


def test_invalid_variable_name():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "posterior_predictive",
            "invalid_variable",  # invalid variable name
            "obs_dim_0",
            "2022-08-01",
            timedelta(days=1),
        )


def test_invalid_dimension_name():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "invalid_dimension",
            "2022-08-01",
            timedelta(days=1),
        )


def test_valid_datetime_start_date():
    start_date_iso = datetime(2022, 8, 1)
    time_step = timedelta(days=1)
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


def test_valid_str_start_date():
    start_date_iso = "2022-08-01"
    time_step = timedelta(days=1)
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


def test_object_list_of_strs():
    """
    For add_time_coords_to_idata_dimensions,
    test whether certain arguments are, in
    fact, lists of strings.
    """
    pass


@pytest.mark.parametrize(
    "idata, group, variable, dimension, time_step, expected_error",
    [
        # valid cases
        (
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "obs_dim_0",
            timedelta(days=1),
            None,
        ),  # valid
        (
            IDATA_WO_DATES,
            "observed_data",
            "obs",
            "obs_dim_0",
            timedelta(days=2),
            None,
        ),  # different valid
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
                start_date_iso="2022-08-01",
                time_step=time_step,
            )
    else:
        # if no error is expected,
        # execute the function
        idata_out = forecasttools.add_time_coords_to_idata_dimension(
            idata=idata,
            group=group,
            variable=variable,
            dimension=dimension,
            start_date_iso="2022-08-01",
            time_step=time_step,
        )
        # validate that the function executed
        # correctly and the dimension has been modified
        assert dimension in idata_out.posterior_predictive[variable].coords
        assert (
            len(idata_out.posterior_predictive[variable].coords[dimension])
            == idata_out.posterior_predictive[variable].sizes[dimension]
        )
        # old_dim = idata[group][variable][dimension]
        # new_dim = idata_out[group][variable][dimension]
        # assert not np.array_equal(old_dim, new_dim)
