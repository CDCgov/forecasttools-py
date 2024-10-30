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

print(idata_wo_dates["observed_data"]["obs_dim_0"])

print(idata_wo_dates["posterior_predictive"])

# NOTE: assumptions about different chains
# ought to be taken into account, not currently
# taken into account; might matter.

# %% OPTION 1 FOR ADDING DATES


def option_1_add_dates_as_coords_to_idata(
    idata_wo_dates: az.InferenceData,
    dim_name: str,
    group_date_mapping: dict[str, tuple[str, timedelta]],
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
    for group_name, (start_date_iso, time_step) in group_date_mapping.items():
        # get idata group
        idata_group = getattr(idata_w_dates, group_name, None)
        # if group exists and contains the specified dimension, update its coordinates
        if idata_group is not None and dim_name in idata_group.dims:
            # convert start date to a datetime object
            start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
            # get the interval size for dimension
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
                .astype(str)
            )
            # update coordinates of group for specified dimension
            idata_group_with_dates = idata_group.assign_coords(
                {dim_name: interval_dates}
            )
            # set the modified group back to the idata object
            setattr(idata_w_dates, group_name, idata_group_with_dates)
        else:
            print(
                f"Warning: '{group_name}' not found or '{dim_name}' not in group.."
            )
    return idata_w_dates


option_1_idata_w_dates = option_1_add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates,
    dim_name="obs_dim_0",
    group_date_mapping={
        "observed_data": ("2022-08-08", timedelta(days=1)),
        "posterior_predictive": ("2022-08-08", timedelta(weeks=1)),
    },
)

print(option_1_idata_w_dates["observed_data"])
print(option_1_idata_w_dates["posterior_predictive"])


# %% OPTION 2 FOR ADDING DATES

# NOTE: this option will likely not work. The idea was to find the
# correct dimension without having to explicitly write the dim name
# out. We know that the dim is not chain or draw, but there may
# be other dims remaining.


def option_2_add_dates_as_coords_to_idata(
    idata_wo_dates: az.InferenceData,
    group_date_mapping: dict[str, tuple[str, timedelta]],
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
    for group_name, (start_date_iso, time_step) in group_date_mapping.items():
        # get idata group
        idata_group = getattr(idata_w_dates, group_name, None)
        # proceed even if group is not located
        if idata_group is None:
            print(f"Warning: Group '{group_name}' not found in idata.")
            continue
        # convert start date to a datetime object
        start_date_as_dt = datetime.strptime(start_date_iso, "%Y-%m-%d")
        # identify dim to replace with dates based on length time series
        target_dim = None
        interval_size = None
        for dim, size in idata_group.sizes.items():
            if dim not in ["chain", "draw"]:
                target_dim = dim
                interval_size = size
                break  # gets first non-chain non-draw dim, failure mode
        # uses the previous determined dimensions in date-setting
        if target_dim:
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
                {target_dim: interval_dates}
            )
            # set the modified group back to the idata object
            setattr(idata_w_dates, group_name, idata_group_with_dates)
        else:
            print(f"Warning: '{group_name}' not found or no target dim found.")
    return idata_w_dates


option_2_idata_w_dates = option_2_add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates,
    group_date_mapping={
        "observed_data": ("2022-08-08", timedelta(days=1)),
        "posterior_predictive": ("2022-08-08", timedelta(weeks=1)),
    },
)
print(option_2_idata_w_dates["observed_data"])
print(option_2_idata_w_dates["posterior_predictive"])

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
            idata_group[variable_name] = idata_group[
                variable_name
            ].assign_coords({dim_name: interval_dates})
    return idata_w_dates


option_4_idata_w_dates = option_4_add_dates_as_coords_to_idata(
    idata_wo_dates=idata_wo_dates,
    group_variable_date_mapping={
        "observed_data": {
            "obs": (
                "2022-08-08",
                timedelta(days=14),
                "obs_dim_0",
            ),  # bi-weekly intervals
        },
        "posterior_predictive": {
            "obs": (
                "2022-08-08",
                timedelta(weeks=2),
                "obs_dim_0",
            ),  # bi-weekly intervals
            "obs2": (
                "2022-08-08",
                timedelta(days=7),
                "obs2_dim_0",
            ),  # weekly intervals
            "rt": (
                "2022-08-08",
                timedelta(weeks=1),
                "rt_dim_0",
            ),  # weekly intervals
        },
    },
)
