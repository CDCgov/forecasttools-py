"""
Testing the historical recode locations
utilities and possibly experimenting with
additional utilities.
"""

# %% LIBRARY IMPORTS

import xarray as xr

import forecasttools

xr.set_options(display_expand_data=False, display_expand_attrs=False)


## %% LOCATION MODIFICATION FUNCTIONS

# %% LOADING FLUSIGHT SUBMISSION

# load example FluSight submission
submission = forecasttools.example_flusight_submission
# display structure of submission
print(submission)
