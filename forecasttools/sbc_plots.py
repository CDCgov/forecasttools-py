"""
Plots for the simulation based calibration
"""

import itertools

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import bdtrik


def plot_results(
    simulations, ndraws, kind="ecdf", var_names=None, figsize=None, color="C0"
):
    """
    Visual diagnostic for SBC.

    Currently it support two options: `ecdf` for the empirical CDF plots
    of the difference between prior and posterior. `hist` for the rank
    histogram.

    Parameters
    ----------
    simulations : dict[str, Any]
        The SBC.simulations dictionary.
    ndraws : int
        Number of draws in each posterior predictive sample
    kind : str
        What kind of plot to make. Supported values are 'ecdf' (default)
        and 'hist'
    var_names : list[str]
        Variables to plot (defaults to all)
    figsize : tuple
        Figure size for the plot. If None, it will be defined automatically.
    color : str
        Color to use for the eCDF or histogram

    Returns
    -------
    fig, axes
        matplotlib figure and axes
    """

    if kind not in ["ecdf", "hist"]:
        raise ValueError(f"kind must be 'ecdf' or 'hist', not {kind}")

    if var_names is None:
        var_names = list(simulations.keys())

    sims = {}
    for k in var_names:
        ary = np.array(simulations[k])
        while ary.ndim < 2:
            ary = np.expand_dims(ary, -1)
        sims[k] = ary

    n_plots = sum(np.prod(v.shape[1:]) for v in sims.values())

    if n_plots > 1:
        if figsize is None:
            figsize = (8, n_plots * 1.0)

        fig, axes = plt.subplots(
            nrows=(n_plots + 1) // 2, ncols=2, figsize=figsize, sharex=True
        )
        axes = axes.flatten()
    else:
        if figsize is None:
            figsize = (8, 1.5)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        axes = [axes]

    if kind == "ecdf":
        cdf = UniformCDF(ndraws)

    idx = 0
    for var_name, var_data in sims.items():
        plot_idxs = list(
            itertools.product(*(np.arange(s) for s in var_data.shape[1:]))
        )

        for indices in plot_idxs:
            if len(plot_idxs) > 1:  # has dims
                dim_label = f"{var_name}[{']['.join(map(str, indices))}]"
            else:
                dim_label = var_name
            ax = axes[idx]
            ary = var_data[(...,) + indices]
            if kind == "ecdf":
                az.plot_ecdf(
                    ary,
                    cdf=cdf,
                    difference=True,
                    pit=True,
                    confidence_bands="auto",
                    plot_kwargs={"color": color},
                    fill_kwargs={"color": color},
                    ax=ax,
                )
            else:
                hist(ary, color=color, ax=ax)
            ax.set_title(dim_label)
            ax.set_yticks([])
            idx += 1

    for extra_ax in range(n_plots, len(axes)):
        fig.delaxes(axes[extra_ax])

    return fig, axes


def hist(ary, color, ax):
    hist, bins = np.histogram(ary, bins="auto")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    max_rank = np.ceil(bins[-1])
    len_bins = len(bins)
    n_sims = len(ary)

    band = np.ceil(bdtrik([0.025, 0.5, 0.975], n_sims, 1 / len_bins))
    ax.bar(
        bin_centers,
        hist,
        width=bins[1] - bins[0],
        color=color,
        edgecolor="black",
    )
    ax.axhline(band[1], color="0.5", ls="--")
    ax.fill_between(
        np.linspace(0, max_rank, len_bins),
        band[0],
        band[2],
        color="0.5",
        alpha=0.5,
    )


class UniformCDF:
    """
    A class to represent the cumulative distribution function (CDF) of a
    uniform distribution.
    """

    def __init__(self, upper_bound):
        """
        Constructs all the necessary attributes for the UniformCDF object.

        Parameters
        ----------
        upper_bound : float
            The upper bound of the uniform distribution. Must be positive.

        Raises
        ------
        ValueError
            If upper_bound is not positive.
        """
        if upper_bound <= 0:
            raise ValueError(
                f"Upper bound must be positive; got {upper_bound}."
            )
        self.upper_bound = upper_bound

    def __call__(self, x):
        """
        Evaluates the CDF at a given value x.

        Parameters
        ----------
        x : array-like
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        array-like
            The CDF evaluated at x. Returns 0 if x < 0, 1 if x > upper_bound,
            and x / upper_bound otherwise.
        """
        return np.where(
            x < 0, 0, np.where(x > self.upper_bound, 1, x / self.upper_bound)
        )
