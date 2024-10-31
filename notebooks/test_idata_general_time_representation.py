"""
Experiment file for testing
general representation of timepoints
in idata object (intervals not just
1D).
"""

# %% LIBRARY IMPORTS

from datetime import datetime, timedelta

import arviz as az
import polars as pl

import forecasttools

# %% LOAD IDATA WO DATES

idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates

# print(idata_wo_dates["observed_data"]["obs_dim_0"])

# print(idata_wo_dates["posterior_predictive"])


# %% FUNCTION FOR MODIFYING SINGLE VARIABLE'S DIMENSIONS


def add_time_coords_to_idata_variable(
    idata: az.InferenceData,
    group: str,
    variable: str,
    dimensions: list[str],
    start_date_iso: str,
    time_step: timedelta,
) -> az.InferenceData:
    """
    Adds time coordinates to a specified variable
    within a group in an ArviZ InferenceData object.
    This function assigns a range of time coordinates
    to a specified dimension in a variable
    within a group in an InferenceData object.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object containing
        the group and variable to modify.
    group : str
        The name of the group within the InferenceData
        object (e.g., "posterior_predictive").
    variable : str
        The name of the variable within the
        specified group to assign time coordinates to.
    dimensions : list[str]
        A list of dimension names to which time
        coordinates should be assigned.
    start_date_iso : str
        The start date for the time coordinates
        in ISO format (e.g., "2022-08-08").
    time_step : timedelta
        The time interval between each coordinate
        (e.g., `timedelta(days=1)` for daily intervals).

    Returns
    -------
    az.InferenceData
        The InferenceData object with updated time
        coordinates for the specified variable.

    Raises
    ------
    ValueError
        If the specified group or variable is not
        found in the InferenceData object or
        if none of the specified dimensions
        exist in the variable.
    """
    # retrieve the specified group from the idata object
    idata_group = getattr(idata, group, None)
    if idata_group is None:
        raise ValueError(f"Group '{group}' not found in idata object.")
    # check if the specified variable exists in the group
    if variable not in idata_group.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in group '{group}'."
        )
    # retrieve the variable's data array
    variable_data = idata_group[variable]
    # check and apply time coordinates only to specified dimensions that exist in the variable
    applicable_dims = [dim for dim in dimensions if dim in variable_data.dims]
    if not applicable_dims:
        raise ValueError(
            f"No specified dimensions found in variable '{variable}' within group '{group}'."
        )
    # convert start date to a datetime object
    start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
    # iterate over applicable dimensions to assign time coordinates
    for dim_name in applicable_dims:
        # determine the interval size for the current dimension
        interval_size = variable_data.sizes[dim_name]
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
        # update the dimension's coordinates (corresponding to the passed variable)
        idata_group = idata_group.assign_coords({dim_name: interval_dates})
    setattr(idata, group, idata_group)
    return idata


# %% USAGE OF (FUNCTION FOR MODIFYING SINGLE VARIABLE'S DIMENSIONS)

# create idata copy for testing
testing_copy_idata_wo_dates = idata_wo_dates.copy()

# examine coords before adding dates
print(testing_copy_idata_wo_dates["posterior_predictive"]["obs"].coords)

# add dates to the
testing_copy_idata_wo_dates = add_time_coords_to_idata_variable(
    idata=testing_copy_idata_wo_dates,
    group="posterior_predictive",
    variable="obs",
    dimensions=["obs_dim_0"],
    start_date_iso="2022-08-08",
    time_step=timedelta(days=1),
)

# examine coords before adding dates
print(testing_copy_idata_wo_dates["posterior_predictive"]["obs"].coords)

# %% FUNCTION FOR MODIFYING SINGLE GROUP'S VARIABLE(S)

# NOTE: each tuple is expected to have the same
# time coordinates (with the only difference
# being the length of the coordinates)


def add_time_coords_to_idata_variables(
    idata: az.InferenceData,
    group_var_dim_tuples: list[tuple[str, str, str]],
    start_date_iso: str,
    time_step: timedelta,
) -> az.InferenceData:
    """
    Adds time-based coordinates to multiple
    (group, variable, dim) tuples within an
    InferenceData object. Each tuple will
    receive the same date range based on the
    provided start date and time step, with
    the only difference possibly being the
    length of the coordinates by variable.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object to modify.
    group_var_dim_tuples : list of tuples
        A list of (group, variable, dim) tuples
        specifying the groups, variables, and
        dimensions to modify.
    start_date_iso : str
        The start date in ISO format (YYYY-MM-DD)
        from which to begin the date range.
    time_step : timedelta
        The time interval between consecutive
        dates.

    Returns
    -------
    az.InferenceData
        The modified InferenceData object with
        updated time coordinates for the specified tuples.
    """
    # iterate over each (group, variable, dimension) tuple
    # and apply the date range using the helper function
    for group, variable, dim in group_var_dim_tuples:
        try:
            # call helper function to assign date coordinates
            # to each specified variable
            idata = add_time_coords_to_idata_variable(
                idata=idata,
                group=group,
                variable=variable,
                dimensions=[dim],  # pass dim as a single-item list
                start_date_iso=start_date_iso,
                time_step=time_step,
            )
        except ValueError as e:
            print(
                f"Error for (group={group}, variable={variable}, dim={dim}): {e}"
            )
    return idata


# %% USAGE OF (FUNCTION FOR MODIFYING SINGLE GROUP'S VARIABLES)

# create idata copy for testing
testing_copy_idata_wo_dates_v2 = idata_wo_dates.copy()

# examine coords before adding dates
print(f"{'observed_data coordinates (before):'.upper()}")
print(testing_copy_idata_wo_dates_v2["observed_data"]["obs"].coords)
print(f"{'posterior_predictive coordinates (before):'.upper()}")
print(testing_copy_idata_wo_dates_v2["posterior_predictive"]["obs"].coords)

# call the function to apply the same date range to all specified tuples
# NOTE: length of coordinates different between groups here
testing_copy_idata_wo_dates_v2 = add_time_coords_to_idata_variables(
    idata=testing_copy_idata_wo_dates,
    group_var_dim_tuples=[
        ("posterior_predictive", "obs", "obs_dim_0"),
        ("observed_data", "obs", "obs_dim_0"),
    ],
    start_date_iso="2022-08-08",
    time_step=timedelta(weeks=1),
)

# examine coords before adding dates
print(f"{'observed_data coordinates (after):'.upper()}")
print(testing_copy_idata_wo_dates_v2["observed_data"]["obs"].coords)
print(f"{'posterior_predictive coordinates (after):'.upper()}")
print(testing_copy_idata_wo_dates_v2["posterior_predictive"]["obs"].coords)

# examine size of coordinates
print(testing_copy_idata_wo_dates_v2["observed_data"]["obs_dim_0"].size)
print(testing_copy_idata_wo_dates_v2["posterior_predictive"]["obs_dim_0"].size)
# %%
