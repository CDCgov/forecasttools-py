"""
Experiment file for testing
general representation of timepoints
in idata object (intervals not just
1D).
"""

# %% LIBRARY IMPORTS

from datetime import datetime, timedelta

import arviz as az
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import polars as pl

import forecasttools

# %% LOAD IDATA WO DATES

idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

print(idata_wo_dates["observed_data"]["obs_dim_0"])

print(idata_wo_dates["posterior_predictive"])


# %% OPTION 3 FOR ADDING DATES


def option_3_add_dates_as_coords_to_idata(
    idata_wo_dates: az.InferenceData,
    group_date_mapping: dict[str, tuple[str, timedelta, str]],
) -> az.InferenceData:
    """
    Modifies an InferenceData object by
    assigning date arrays to selected
    groups.
    """
    # create initial idata object from received object
    idata_w_dates = idata_wo_dates.copy()

    # iterate over selected groups
    # NOTE: policy not decided for non-selected groups
    for group_name, (
        start_date_iso,
        time_step,
        dim_name,
    ) in group_date_mapping.items():
        # get idata group
        idata_group = getattr(idata_w_dates, group_name, None)
        # skip if group or dim_name is not located
        if idata_group is None:
            print(f"Warning: Group '{group_name}' not found in idata.")
            continue
        if dim_name not in idata_group.dims:
            print(
                f"Warning: Dimension '{dim_name}' not found in group '{group_name}'."
            )
            continue
        # convert start date to a datetime object
        start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
        # get the interval size for this dimension
        interval_size = idata_group.sizes[dim_name]
        # generate date range using the specified group time_step
        # otherwise use 1d; currently not type str.
        interval_dates = (
            pl.date_range(
                start=start_date_as_dt,
                end=start_date_as_dt + (interval_size - 1) * time_step,
                interval=f"{time_step.days}d" if time_step.days else "1d",
                closed="both",
                eager=True,
            )
            .to_numpy()
            .astype("datetime64[ns]")
        )
        # update coordinates of group for specified dimension
        idata_group_with_dates = idata_group.assign_coords(
            {dim_name: interval_dates}
        )
        # set the modified group back to the idata object
        setattr(idata_w_dates, group_name, idata_group_with_dates)
    return idata_w_dates


option_3_idata_w_dates = option_3_add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates,
    group_date_mapping={
        "observed_data": ("2022-08-08", timedelta(days=1), "obs_dim_0"),
        "posterior_predictive": (
            "2022-08-08",
            timedelta(weeks=1),
            "obs_dim_0",
        ),
    },
)

print(option_3_idata_w_dates["observed_data"])
print(option_3_idata_w_dates["posterior_predictive"])

# %% OPTION 4 FOR ADDING DATES (MODIFIED OPTION 3)

# NOTE: thinking about how to make this rely less
# on for-looping but it represents a first attempt
# (hastily done) to address comments in the PR


# NOTE: as of 4:20PM EST this is untested because
# I need to create an idata object  that actually
# has the variables below in posterior predictive
# I think the logic of the below should roughly be
# correct though


def option_4_add_dates_as_coords_to_idata(
    idata_wo_dates: az.InferenceData,
    group_variable_date_mapping: dict[
        str, dict[str, tuple[str, timedelta, str]]
    ],
) -> az.InferenceData:
    """
    Modifies an InferenceData object by
    assigning date arrays to selected
    variables within specified groups.
    Allows different variables within the same group
    to have separate date intervals.
    """
    # copy the idata object to avoid modifying the original
    idata_w_dates = idata_wo_dates.copy()
    # iterate over each group in the mapping
    for (
        group_name,
        variable_date_mapping,
    ) in group_variable_date_mapping.items():
        # get the group from the idata object
        idata_group = getattr(idata_w_dates, group_name, None)
        if idata_group is None:
            print(f"Warning: Group '{group_name}' not found in idata.")
            continue
        # iterate over each variable in the group's variable mapping
        for variable_name, (
            start_date_iso,
            time_step,
            dim_name,
        ) in variable_date_mapping.items():
            # check if the variable exists in the group
            if variable_name not in idata_group.data_vars:
                print(
                    f"Warning: Variable '{variable_name}' not found in group '{group_name}'."
                )
                continue
            # check if the specified dimension exists in the variable's dimensions
            variable_dims = idata_group[variable_name].dims
            if dim_name not in variable_dims:
                print(
                    f"Warning: Dimension '{dim_name}' not found in variable '{variable_name}' within group '{group_name}'."
                )
                continue
            # convert start date to a datetime object
            start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
            # determine the size of the specified dimension
            interval_size = idata_group[variable_name].sizes[dim_name]
            # generate date range using the specified time_step and interval size
            interval_dates = (
                pl.date_range(
                    start=start_date_as_dt,
                    end=start_date_as_dt + (interval_size - 1) * time_step,
                    interval=f"{time_step.days}d" if time_step.days else "1d",
                    closed="both",
                    eager=True,
                )
                .to_numpy()
                .astype("datetime64[ns]")
            )
            # update the variable's coordinates with the generated date range
            idata_group = idata_group.assign_coords({dim_name: interval_dates})
        setattr(idata_w_dates, group_name, idata_group)
    return idata_w_dates


# %% ENSURE OPTION 4 WORKS AS INTENDED (MODEL SETUP)


# NOTE: reusing some items from test_general_add_dates

rng_key = jr.key(213123)


def model(obs=None, obs2=None, rt=None):
    """
    Simple Numpyro model for multi-variable
    posterior_predictive group in idata
    """
    # priors (sample names and values chosen somewhat arbitrarily)
    lambda_obs = numpyro.sample(
        "lambda_obs", dist.Exponential(1.0).expand([28])
    )
    lambda_obs2 = numpyro.sample(
        "lambda_obs2", dist.Exponential(1.5).expand([10])
    )
    lambda_rt = numpyro.sample("lambda_rt", dist.Exponential(2.0).expand([12]))
    # likelihoods
    numpyro.sample("obs", dist.Poisson(lambda_obs), obs=obs)
    numpyro.sample("obs2", dist.Poisson(lambda_obs2), obs=obs2)
    numpyro.sample("rt", dist.Normal(lambda_rt, 0.1), obs=rt)


def generate_weekly_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("weekly_data", dist.Poisson(4.5).expand([12]))


def generate_biweekly_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("biweekly_data", dist.Poisson(6.0).expand([10]))


def generate_daily_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("daily_data", dist.Poisson(8.0).expand([28]))


rt_weekly_data = generate_weekly_data(rng_key_int=47)
obs2_biweekly_data = generate_biweekly_data(rng_key_int=47)
obs_daily_data = generate_daily_data(rng_key_int=47)

kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(kernel, num_samples=1000, num_warmup=500)
mcmc.run(
    rng_key, obs=obs_daily_data, obs2=obs2_biweekly_data, rt=rt_weekly_data
)


posterior_samples = mcmc.get_samples()

posterior_pred_samples = numpyro.infer.Predictive(
    model, posterior_samples=posterior_samples
)(rng_key=rng_key)

idata_wo_dates_new = az.from_numpyro(
    posterior=mcmc, posterior_predictive=posterior_pred_samples
)

print(idata_wo_dates_new["posterior_predictive"])

# %% ENSURE OPTION 4 WORKS AS INTENDED (MODEL RUNNING)


option_4_idata_w_dates = option_4_add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates_new,
    group_variable_date_mapping={
        "observed_data": {
            "obs": (
                "2022-08-08",
                timedelta(days=14),
                "obs_dim_0",
            ),
        },
        "posterior_predictive": {
            "obs": (
                "2022-08-08",
                timedelta(days=7),
                "obs_dim_0",  # change to f"{var_name}_dim_0"?
            ),
            "obs2": (
                "2022-08-08",
                timedelta(weeks=2),
                "obs2_dim_0",
            ),
            "rt": (
                "2022-08-01",
                timedelta(weeks=1),
                "rt_dim_0",
            ),
        },
    },
)


# %% EXAMINE EACH VAR


# print(idata_wo_dates_new["posterior_predictive"]["obs2"].coords)
# print(idata_wo_dates_new["posterior_predictive"]["obs"].coords)
# print(idata_wo_dates_new["posterior_predictive"]["rt"].coords)

print(option_4_idata_w_dates["posterior_predictive"]["obs2"].coords)
print(option_4_idata_w_dates["posterior_predictive"]["obs"].coords)
print(option_4_idata_w_dates["posterior_predictive"]["rt"].coords)

# %%


# FEATURED IN PR COMMENTS

# updated_idata = add_dates_across_groups(
#     idata=idata_wo_dates,
#     groups=["posterior_predictive", "observed_data"],
#     variable_date_mapping=variable_date_mapping = {
#         "obs": ("2022-08-08", timedelta(days=14), "obs_dim_0"),
#         "obs2": ("2022-08-08", timedelta(weeks=1), "obs2_dim_0"),
#         "rt": ("2022-08-01", timedelta(weeks=1), "rt_dim_0"),
#     }
# )

# updated_idata = add_dates_across_groups(
#     idata=idata_wo_dates,
#     groups=["posterior_predictive", "observed_data"],
#     variable_date_mapping= {
#         "obs": {
#             "observed_data": ("2022-08-08", timedelta(days=14), "obs_dim_0"),
#             "posterior_predictive": ("2022-08-08", timedelta(weeks=2), "obs_post_dim"),
#         },
#         "obs2": {
#             "posterior_predictive": ("2022-08-08", timedelta(days=7), "obs2_post_dim"),
#         },
#         "rt": {
#             "all": ("2022-08-01", timedelta(weeks=1), "rt_dim_0"), # change across all groups
#         },
#     }
# )
