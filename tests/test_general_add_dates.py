"""
Tests adding dates to InferenceData
objects with different dim names.

poetry run pytest tests
"""

# %% LIBRARY IMPORTS


import arviz as az
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

# %% MODELS FOR TESTING ADDING DATES


def weekly_rt_nowcast(y=None):
    """Weekly Rt nowcast (random walk)."""
    rt = numpyro.sample("Rt", dist.Normal(1.0, 0.1))
    rt_walk = numpyro.sample("random_walk", dist.Normal(rt, 0.2).expand([12]))
    numpyro.sample("y", dist.Poisson(jnp.exp(rt_walk)), obs=y)


def biweekly_hospitalizations(obs=None):
    """Bi-weekly hospitalizations."""
    lambda_ = numpyro.sample("lambda", dist.Exponential(1.0).expand([10]))
    numpyro.sample("obs", dist.Poisson(lambda_), obs=obs)


def daily_hospitalizations(observations=None):
    """Daily hospitalizations."""
    lambda_ = numpyro.sample("lambda", dist.Exponential(1.5).expand([28]))
    numpyro.sample("observations", dist.Poisson(lambda_), obs=observations)


# %% MCMC RUNNER AND IDATA GETTER


def run_mcmc(
    model, obs_data, rng_key_int, num_chains, num_samples=500, num_warmup=200
):
    """Gets MCMC object for a given model and observational data"""
    rng_key = jr.key(rng_key_int)
    # set up inference
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
    )
    # get model args and used correct arg for
    # provided model
    model_args = model.__code__.co_varnames
    kwargs = {}
    if "y" in model_args:
        kwargs["y"] = obs_data
    if "obs" in model_args:
        kwargs["obs"] = obs_data
    if "observations" in model_args:
        kwargs["observations"] = obs_data
    # run mcmcms
    mcmc.run(rng_key, **kwargs)
    return mcmc


def get_idata_object(model, mcmc, rng_key_int):
    # get posterior samples
    posterior_samples = mcmc.get_samples()
    # get posterior predictive forecast
    posterior_pred_samples = numpyro.infer.Predictive(
        model, posterior_samples=posterior_samples
    )(rng_key=jr.key(rng_key_int))
    # create InferenceData object
    idata_wo_dates = az.from_numpyro(
        posterior=mcmc, posterior_predictive=posterior_pred_samples
    )
    return idata_wo_dates


# %% ARTIFICIAL DATA GENERATORS


def generate_weekly_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("weekly_data", dist.Poisson(4.5).expand([12]))


def generate_biweekly_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("biweekly_data", dist.Poisson(6.0).expand([10]))


def generate_daily_data(rng_key_int: int):
    with numpyro.handlers.seed(rng_seed=rng_key_int):
        return numpyro.sample("daily_data", dist.Poisson(8.0).expand([28]))


# %% GETTING ARTIFICIAL DATA

# generate artificial data
weekly_data = generate_weekly_data(rng_key_int=47)
biweekly_data = generate_biweekly_data(rng_key_int=47)
daily_data = generate_daily_data(rng_key_int=47)


# %% RUNNING MODELS, GETTING MCMC OBJECTS

mcmc_weekly = run_mcmc(
    weekly_rt_nowcast, obs_data=weekly_data, rng_key_int=532, num_chains=2
)
mcmc_biweekly = run_mcmc(
    biweekly_hospitalizations,
    obs_data=biweekly_data,
    rng_key_int=532,
    num_chains=3,
)
mcmc_daily = run_mcmc(
    daily_hospitalizations, obs_data=daily_data, rng_key_int=532, num_chains=4
)

# %% CONVERTING TO INFERENCE DATA WITHOUT DATES

idata_weekly = get_idata_object(
    model=weekly_rt_nowcast, mcmc=mcmc_weekly, rng_key_int=67
)
idata_biweekly = get_idata_object(
    model=biweekly_hospitalizations, mcmc=mcmc_biweekly, rng_key_int=67
)
idata_daily = get_idata_object(
    model=daily_hospitalizations, mcmc=mcmc_daily, rng_key_int=67
)
