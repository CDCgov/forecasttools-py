"""
Tests adding dates to idata objects.
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
        ("2022-08-01", "invalid-timedelta", TypeError),  # invalid time_step
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


def test_invalid_group():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "invalid_group",
            "obs",
            "obs_dim_0",
            "2022-08-01",
            timedelta(days=1),
        )


def test_invalid_variable():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "posterior_predictive",
            "invalid_variable",
            "obs_dim_0",
            "2022-08-01",
            timedelta(days=1),
        )


def test_invalid_dimension():
    with pytest.raises(ValueError):
        forecasttools.add_time_coords_to_idata_dimension(
            IDATA_WO_DATES,
            "posterior_predictive",
            "obs",
            "invalid_dimension",
            "2022-08-01",
            timedelta(days=1),
        )


def test_valid_datetime():
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


def test_valid_str_date():
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
