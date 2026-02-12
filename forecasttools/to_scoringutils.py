"""
Takes differently formatted forecast objects
and converts them into a format ready for R's
`scoringutils`. A FluSight submission has all
necessary content already present save for
the observation (both for the fitting and
forecast period). An InferenceData object can
have the observations available
"""

import arviz as az
import polars as pl


def from_flusight_to_scoringutils(
    submission: pl.DataFrame, observations: list[float]
) -> None:
    """ """

    pass


def from_idata_to_scoringutils(
    save_path: str,
    idata: az.InferenceData,
    observations: list[float],
    idata_samp_param: str,
    idata_obs_param: str,
    overwrite: str = False,
) -> None:
    """
    Take a InferenceData object generated
    from NumPyro posterior and posterior
    predictive samples along with observations
    and generate a parquet file with the
    correct columns for use in `scoringutils`.

    Parameters
    ----------
    save_path
        The save path for the generated
        parquet file.
    idata
        An InferenceData object containing
        a forecast made using NumPyro.
    observations
        An array of observations corresponding
        to the
    overwrite
        Whether to overwrite an existing file
        of the same name.

    Returns
    -------
    None
        A parquet file.
    """
