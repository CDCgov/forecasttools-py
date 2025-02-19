"""
Plots for the simulation based calibration.
"""

import itertools

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
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
        cdf = dist.DiscreteUniform(high=ndraws).cdf

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
