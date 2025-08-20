import arviz as az
import jax.numpy as jnp
import numpyro
from jax import random
from numpyro.infer import MCMC
from numpyro.infer.mcmc import MCMCKernel
from tqdm import tqdm

from forecasttools.sbc_plots import plot_results


class SBC:
    def __init__(
        self,
        mcmc_kernel: MCMCKernel,
        *args,
        observed_vars: dict[str, str],
        num_simulations=10,
        sample_kwargs=None,
        seed=None,
        inspection_mode=False,
        **kwargs,
    ) -> None:
        """
        Set up class for doing SBC.
        Based on simulation based calibration (Talts et. al. 2018) in PyMC.

        Parameters
        ----------
        mcmc_kernel : numpyro.infer.mcmc.MCMCKernel
            An instance of a numpyo MCMC kernel object.
        observed_vars : dict[str, str]
            A dictionary mapping observed/response variable name as a kwarg to
            the numpyro model to the corresponding variable name sampled using
            `numpyro.sample`.
        args : tuple
            Positional arguments passed to `numpyro.sample`.
        num_simulations : int
            How many simulations to run for SBC.
        sample_kwargs : dict[str, Any]
            Arguments passed to `numpyro.sample`. Defaults to
            `dict(num_warmup=500, num_samples=100, progress_bar = False)`.
            Which assumes a MCMC sampler e.g. NUTS.
        seed : random.PRNGKey
            Random seed.
        kwargs : dict[str, Any]
            Keyword arguments passed to `numpyro` models.
        """
        if sample_kwargs is None:
            sample_kwargs = dict(num_warmup=500, num_samples=100, progress_bar=False)
        if seed is None:
            seed = random.PRNGKey(1234)
        self.mcmc_kernel = mcmc_kernel
        if not hasattr(mcmc_kernel, "model"):
            raise ValueError("The `mcmc_kernel` must have a 'model' attribute.")

        self.model = mcmc_kernel.model
        self.args = args
        self.kwargs = kwargs
        self.observed_vars = observed_vars

        for key in self.observed_vars:
            if key in self.kwargs and self.kwargs[key] is not None:
                raise ValueError(
                    f"The value for '{key}' in kwargs must be None for this to"
                    " be a prior predictive check."
                )

        self.num_simulations = num_simulations
        self.sample_kwargs = sample_kwargs
        # Initialize the simulations and random seeds
        self.simulations = {}
        self._simulations_complete = 0
        prior_pred_rng, sampler_rng = random.split(seed)
        self._prior_pred_rng = prior_pred_rng
        self._sampler_rng = sampler_rng
        self.num_samples = None
        # Set the inspection mode
        # if in inspection mode, store all idata objects from fitting
        self.inspection_mode = inspection_mode
        if inspection_mode:
            self.idatas = []

    def _get_prior_predictive_samples(
        self,
    ) -> tuple[dict[str, any], dict[str, any]]:
        """
        Generate samples to use for the simulations by prior predictive
        sampling. Then splits between observed and unobserved variables based
        on the `observed_vars` attribute.

        Returns
        -------
        tuple[dict[str, any], dict[str, any]]
            The prior and prior predictive samples.
        """
        prior_predictive_fn = numpyro.infer.Predictive(
            self.mcmc_kernel.model, num_samples=self.num_simulations
        )
        prior_predictions = prior_predictive_fn(
            self._prior_pred_rng, *self.args, **self.kwargs
        )
        prior_pred = {k: prior_predictions[v] for k, v in self.observed_vars.items()}
        prior = {
            k: v
            for k, v in prior_predictions.items()
            if k not in self.observed_vars.values()
        }
        return prior, prior_pred

    def _get_posterior_samples(
        self, seed: random.PRNGKey, prior_predictive_draw: dict[str, any]
    ) -> tuple[az.InferenceData, int]:
        """
        Generate posterior samples conditioned to a prior predictive sample.
        This returns the posterior samples and the number of samples. The
        number of samples are used in scaling plotting and checking that each
        inference draw has the same number of samples.

        Parameters
        ----------
        seed : random.PRNGKey
            Random seed for MCMC sampling.
        prior_predictive_draw : dict[str, any]
            Prior predictive samples.

        Returns
        -------
        tuple[az.InferenceData, int]
            Posterior samples as an arviz InferenceData object, with the count
            of posterior samples.
        """
        mcmc = MCMC(self.mcmc_kernel, **self.sample_kwargs)
        obs_vars = {**self.kwargs, **prior_predictive_draw}
        mcmc.run(seed, *self.args, **obs_vars)
        num_samples = mcmc.num_samples
        # Check that the number of samples is consistent
        if self.num_samples is None:
            self.num_samples = num_samples
        if self.num_samples != num_samples:
            raise ValueError(
                "The number of samples from the posterior is not consistent."
            )
        idata = az.from_numpyro(mcmc)
        return idata

    def _increment_rank_statistics(self, prior_draw, posterior) -> None:
        """
        Increment the rank statistics for each parameter in the prior draw.

        This method updates the `self.simulations` dictionary with the rank
        statistics for each parameter in the `prior_draw` compared to the
        `posterior`.

        Returns:
            None
        """
        for name in prior_draw:
            num_dims = jnp.ndim(prior_draw[name])
            if num_dims == 0:
                rank_statistics = (
                    (posterior[name].sel(chain=0) < prior_draw[name]).sum().values
                )
                self.simulations[name].append(rank_statistics)
            else:
                rank_statistics = (
                    (posterior[name].sel(chain=0) < prior_draw[name]).sum(axis=0).values
                )
                self.simulations[name].append(rank_statistics)

    def run_simulations(self) -> None:
        """
        The main method of `SBC` class that runs the simulations for
        simulation based calibration and fills the `simulations` attribute
        with the results.
        """
        prior, prior_pred = self._get_prior_predictive_samples()
        sampler_seeds = random.split(self._sampler_rng, self.num_simulations)
        self.simulations = {name: [] for name in prior}
        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )
        if self.inspection_mode:
            self.prior = prior
            self.prior_pred = prior_pred
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_draw = {k: v[idx] for k, v in prior.items()}
                prior_predictive_draw = {k: v[idx] for k, v in prior_pred.items()}
                idata = self._get_posterior_samples(
                    sampler_seeds[idx], prior_predictive_draw
                )
                if self.inspection_mode:
                    self.idatas.append(idata)
                self._increment_rank_statistics(prior_draw, idata["posterior"])
                self._simulations_complete += 1
                progress.update()
        finally:
            self.simulations = {
                k: v[: self._simulations_complete] for k, v in self.simulations.items()
            }
            progress.close()

    def plot_results(self, kind="ecdf", var_names=None, color="C0"):
        """
        Visual diagnostic for SBC.

        Currently it support two options: `ecdf` for the empirical CDF plots
        of the difference between prior and posterior. `hist` for the rank
        histogram.

        Parameters
        ----------
        simulations
            The SBC.simulations dictionary.
        kind : str
            What kind of plot to make. Supported values are 'ecdf' (default)
            and 'hist'
        var_names : list[str]
            Variables to plot (defaults to all)
        figsize : tuple
            Figure size for the plot. If None, it will be defined
            automatically.
        color : str
            Color to use for the eCDF or histogram

        Returns
        -------
        fig, axes
            matplotlib figure and axes
        """
        return plot_results(
            self.simulations,
            self.num_samples,
            kind=kind,
            var_names=var_names,
            color=color,
        )
