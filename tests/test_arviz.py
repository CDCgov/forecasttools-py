import copy
import datetime as dt

import arviz as az
import numpy as np
import xarray as xr

import forecasttools as ft

IDATA_WO_DATES = ft.nhsn_flu_forecast_wo_dates


def test_replace_all_dim_suffix_basic():
    """Test basic functionality of ft.arviz.replace_all_dim_suffix."""
    new_suffixes = ["time"]
    all_dims_original = ft.arviz.get_all_dims(IDATA_WO_DATES)

    result = ft.arviz.replace_all_dim_suffix(IDATA_WO_DATES, new_suffixes)
    all_dims_desired = {x.replace("dim_0", "time") for x in all_dims_original}

    all_dims_result = ft.arviz.get_all_dims(result)
    assert all_dims_result == all_dims_desired


def test_replace_all_dim_suffix_inplace():
    """Test inplace modification."""
    all_dims_original = ft.arviz.get_all_dims(IDATA_WO_DATES)

    idata_copy = copy.deepcopy(IDATA_WO_DATES)
    new_suffixes = ["time"]

    result = ft.arviz.replace_all_dim_suffix(idata_copy, new_suffixes, inplace=True)

    # Should return None when inplace=True
    assert result is None

    # Check that original object was modified
    all_dims_desired = {x.replace("dim_0", "time") for x in all_dims_original}
    all_dims_result = ft.arviz.get_all_dims(idata_copy)

    assert all_dims_result == all_dims_desired


def test_replace_all_dim_suffix_custom_prefix():
    """Test with custom dimension prefix."""
    # Create test data with custom prefix
    coords = {"test_0": [0, 1, 2], "test_1": [0, 1]}
    data = xr.Dataset(
        {"obs": (["test_0", "test_1"], np.random.rand(3, 2))}, coords=coords
    )
    idata = az.InferenceData(posterior=data)

    new_suffixes = ["time", "location"]
    result = ft.arviz.replace_all_dim_suffix(idata, new_suffixes, dim_prefix="test_")
    set(new_suffixes)

    all_dims_result = ft.arviz.get_all_dims(result)
    assert all_dims_result == set(new_suffixes)


def test_replace_all_dim_suffix_empty_suffixes():
    """Test with empty new_suffixes list."""
    new_suffixes = []
    all_dims_original = ft.arviz.get_all_dims(IDATA_WO_DATES)
    result = ft.arviz.replace_all_dim_suffix(IDATA_WO_DATES, new_suffixes)
    all_dims_result = ft.arviz.get_all_dims(result)
    # Should return unchanged InferenceData
    assert all_dims_result == all_dims_original


def test_assign_coords_from_start_step_basic():
    dim_name = "beta_coeffs_dim_0"
    start = dt.date(2020, 1, 1)
    interval = dt.timedelta(days=1)

    result = ft.arviz.assign_coords_from_start_step(
        IDATA_WO_DATES, dim_name, start, interval=interval, inplace=False
    )

    assert isinstance(result, az.InferenceData)
    assert result is not IDATA_WO_DATES

    for group in result.groups():
        ds = getattr(result, group)
        if dim_name in ds.dims:
            result_coords = ds.coords[dim_name]
            assert result_coords[0] == start
            assert result_coords[1] == start + interval


def test_assign_coords_from_start_step_inplace():
    idata_copy = copy.deepcopy(IDATA_WO_DATES)
    dim_name = "beta_coeffs_dim_0"
    start = dt.date(2019, 9, 29)
    interval = dt.timedelta(days=7)

    result = ft.arviz.assign_coords_from_start_step(
        idata_copy, dim_name, start, interval=interval, inplace=True
    )
    assert result is None

    for group in idata_copy.groups():
        ds = getattr(idata_copy, group)
        if dim_name in ds.dims:
            result_coords = ds.coords[dim_name]
            assert result_coords[0] == start
            assert result_coords[1] == start + interval


def test_assign_coords_from_start_step_no_matching_dim():
    dim_name = "nonexistent_dimension_xyz"
    start = dt.date(2020, 1, 1)

    result = ft.arviz.assign_coords_from_start_step(
        IDATA_WO_DATES, dim_name, start, inplace=False
    )

    for group in IDATA_WO_DATES.groups():
        original_ds = getattr(IDATA_WO_DATES, group)
        result_ds = getattr(result, group)
        assert original_ds.equals(result_ds)
