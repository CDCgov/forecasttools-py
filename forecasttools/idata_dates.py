import datetime as dt
from copy import deepcopy
from typing import Literal

import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs


def get_all_dims(idata: az.InferenceData) -> set[str]:
    """
    Get all unique dimension names from an ArviZ InferenceData object.
    This function iterates through all groups in the InferenceData object and
    collects all unique dimension names across all groups.
    Parameters
    ----------
    idata : az.InferenceData
        The ArviZ InferenceData object to extract dimensions from.
    Returns
    -------
    set[str]
        A set containing all unique dimension names found across all groups
        in the InferenceData object.
    """

    dims = set()
    set(dim for group in idata.values() for dim in group.dims)
    return dims


def replace_all_suffix(
    idata: az.InferenceData,
    new_suffixes: list[str],
    dim_prefix: str = "dim_",
    groups: str | list[str] | None = None,
    filter_groups: Literal["like", "regex"] | None = None,
    inplace: bool = False,
) -> az.InferenceData | None:
    """
    Replace dimension suffixes in an InferenceData object.

    This function renames dimensions by replacing their suffixes with new ones.
    Dimensions are expected to end with patterns like 'dim_0', 'dim_1', etc.,
    which will be replaced with the corresponding new suffixes.

    Parameters
    ----------
    idata : arviz.InferenceData
        The InferenceData object containing dimensions to rename.
    new_suffixes : list of str
        List of new suffixes to replace the existing 'dim_i' suffixes.
        The order corresponds to dim_0, dim_1, dim_2, etc.
    dim_prefix : str, default "dim_"
        The prefix used for dimension names. This is used to identify
        the dimensions to rename.
    groups : str | list of str, optional
        Groups where the selection is to be applied. Can either be group names or metagroup names.
    filter_groups : Literal["like", "regex"] | None = None
        If `None` (default), interpret groups as the real group or metagroup names.
        If "like", interpret groups as substrings of the real group or metagroup names.
        If "regex", interpret groups as regular expressions on the real group or
        metagroup names. A la `pandas.filter`.
    inplace : bool, default False
        If ``True``, modify the InferenceData object inplace, otherwise, return the modified copy.

    Returns
    -------
    InferenceData
        A new InferenceData object by default.
        When `inplace==True` perform renaming in-place and return `None`
    """

    original_dim_names = get_all_dims(idata)

    suffix_dict = {f"{dim_prefix}{i}": new for i, new in enumerate(new_suffixes)}

    name_dict = {
        original_dim_name: original_dim_name.replace(suffix, new)
        for original_dim_name in original_dim_names
        for suffix, new in suffix_dict.items()
        if original_dim_name.endswith(suffix)
    }
    new_idata = idata.rename(
        name_dict=name_dict,
        groups=groups,
        filter_groups=filter_groups,
        inplace=inplace,
    )
    return new_idata


def assign_coords_from_start_step(
    idata: az.InferenceData,
    dim_name: str,
    start_date: dt.date,
    interval: dt.timedelta = dt.timedelta(days=1),
    inplace: bool = False,
) -> az.InferenceData | None:
    """
    Assign coordinates to a dimension based on a start date and step interval.

    This function updates the coordinates of a specified dimension across all groups
    in an ArviZ InferenceData object by creating a sequence of dates starting from
    a given start date and incrementing by a specified time interval.

    Parameters
    ----------
    idata : az.InferenceData
        The ArviZ InferenceData object to update.
    dim_name : str
        The name of the dimension to assign new coordinates to.
    start_date : dt.date
        The starting date for the coordinate sequence.
    interval : dt.timedelta, optional
        The time interval between consecutive coordinate values. Default is 1 day.
    inplace : bool, optional
        If True, modify the InferenceData object in place and return None.
        If False, return a deep copy with updated coordinates. Default is False.

    Returns
    -------
    az.InferenceData or None
        If inplace is False, returns a new InferenceData object with updated
        coordinates. If inplace is True, returns None and modifies the input
        object directly.

    Notes
    -----
    The function iterates through all groups in the InferenceData object and
    updates the specified dimension coordinates only in groups where the
    dimension exists. The new coordinates are generated as a sequence of
    dates starting from `start_date` and incrementing by `interval` for
    each position in the dimension.
    """
    out = idata if inplace else deepcopy(idata)
    for group in list(out.groups()):
        ds = getattr(out, group)
        if dim_name in ds.dims:
            n = ds.sizes[dim_name]
            coords = start_date + interval * np.arange(n)
            new_ds = ds.assign_coords({dim_name: coords})
            setattr(out, group, new_ds)
    if inplace:
        return None
    else:
        return out


def coalesce_common_columns(
    df: pl.DataFrame, suffix: str, new_colname: str | None = None
) -> pl.DataFrame:
    """
    Coalesce multiple columns with a common suffix into a single column.
    This function finds all columns in the DataFrame that end with the specified
    suffix, coalesces them (takes the first non-null value across the columns),
    and creates a new column with the coalesced values. The original columns
    with the suffix are then removed from the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame to process.
    suffix : str
        The suffix to match column names for coalescing.
    new_colname : str | None, optional
        The name for the new coalesced column.
        If None, defaults to the suffix with leading underscores stripped.

    Returns:
        pl.DataFrame: A new DataFrame with the coalesced column and original suffix columns removed.
    """
    if new_colname is None:
        new_colname = suffix.lstrip("_")
    coalesced_df = df.with_columns(
        pl.coalesce(cs.ends_with(suffix)).alias(new_colname)
    ).select(cs.exclude(cs.ends_with(suffix)))

    return coalesced_df
