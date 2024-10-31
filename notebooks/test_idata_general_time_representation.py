"""
Experiment file for testing
general representation of timepoints
in idata object (intervals not just
1D).
"""

# %% LIBRARY IMPORTS

# from datetime import datetime, timedelta

# import arviz as az
# import jax.random as jr
# import numpyro
# import numpyro.distributions as dist
# import polars as pl

import forecasttools

# %% LOAD IDATA WO DATES

idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

print(idata_wo_dates["observed_data"]["obs_dim_0"])

print(idata_wo_dates["posterior_predictive"])


# %% ATOMIC FUNCTION FOR MODIFYING SINGLE VARIABLE'S DIMENSIONS


# %% FUNCTION FOR MODIFYING SINGLE GROUP'S VARIABLES
