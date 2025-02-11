"""
Test the SBC class using a simple model.

```math
\begin{aligned}
\\mu &\\sim \text{Normal}(0, 1), \\
z &\\sim \text{Normal}(\\mu, 1).
\\end{aligned}
```
"""

import numpyro
import pytest
from jax import random
from numpyro.infer import NUTS

from forecasttools.sbc import SBC


@pytest.fixture
def simple_model():
    def model(y=None):
        mu = numpyro.sample("mu", numpyro.distributions.Normal(0, 1))
        numpyro.sample("z", numpyro.distributions.Normal(mu, 1), obs=y)

    return model


@pytest.fixture
def mcmc_kernel(simple_model):
    return NUTS(simple_model)


@pytest.fixture
def observed_vars():
    return {"y": "z"}


@pytest.fixture
def sbc_instance(mcmc_kernel, observed_vars):
    return SBC(mcmc_kernel, y=None, observed_vars=observed_vars)


def test_sbc_initialization(sbc_instance, mcmc_kernel, observed_vars):
    """
    Test that the SBC class is initialized correctly.
    """
    assert sbc_instance.mcmc_kernel == mcmc_kernel
    assert sbc_instance.observed_vars == observed_vars
    assert sbc_instance.num_simulations == 10
    assert sbc_instance.sample_kwargs == dict(
        num_warmup=500, num_samples=100, progress_bar=False
    )
    assert sbc_instance._simulations_complete == 0


def test_get_prior_predictive_samples(sbc_instance):
    """
    Test that the prior and prior predictive samples are generated correctly.
    """
    prior, prior_pred = sbc_instance._get_prior_predictive_samples()
    assert "y" in prior_pred
    assert "mu" in prior


def test_get_posterior_samples(sbc_instance):
    """
    Test that the posterior samples are generated correctly.
    """
    prior, prior_pred = sbc_instance._get_prior_predictive_samples()
    seed = random.PRNGKey(0)
    idata, num_samples = sbc_instance._get_posterior_samples(seed, prior_pred)
    assert "posterior" in idata
    assert num_samples == 100


def test_run_simulations(sbc_instance):
    """
    Test that the simulations for SBC are run correctly.
    """
    sbc_instance.run_simulations()
    assert sbc_instance._simulations_complete == sbc_instance.num_simulations
    assert "mu" in sbc_instance.simulations


def test_plot_results(sbc_instance):
    """
    Test that the results are plotted.
    """
    sbc_instance.run_simulations()
    fig, axes = sbc_instance.plot_results()
    assert fig is not None
    assert axes is not None
