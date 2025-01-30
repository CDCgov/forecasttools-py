# import arviz as az
# import numpy as np
# import polars as pl
# import pytest
# import xarray as xr

# import forecasttools


# @pytest.fixture
# def mock_inference_data():
#     np.random.seed(42)
#     posterior_predictive = xr.Dataset(
#         {
#             "observed_hospital_admissions": ("chain", np.random.randn(2, 100)),
#         },
#         coords={"chain": [0, 1]}
#     )

#     idata = az.from_dict(posterior_predictive=posterior_predictive)

#     return idata



# def test_valid_conversion(mock_inference_data):
#     result = forecasttools.convert_inference_data_to_tidydraws(mock_inference_data, ["posterior_predictive"])
#     assert isinstance(result, dict)
#     assert "posterior_predictive" in result
#     assert isinstance(result["posterior_predictive"], pl.DataFrame)

#     df = result["posterior_predictive"]
#     assert all(col in df.columns for col in [".chain", ".draw", ".iteration", "variable", "value"])

#     assert df[".draw"].n_unique() == df[".draw"].shape[0]


# def test_invalid_group(mock_inference_data):
#     with pytest.raises(ValueError, match="Invalid groups provided"):
#         forecasttools.convert_inference_data_to_tidydraws(mock_inference_data, ["invalid_group"])

# def test_empty_group_list(mock_inference_data):
#     result = forecasttools.convert_inference_data_to_tidydraws(mock_inference_data, [])
#     assert result == {}
