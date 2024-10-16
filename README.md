# CFA Forecast Tools (Python)

The following repository, `forecasttools-py` is a _Python package for common pre- and post-processing operations done by CFA Predict for short term forecasting, nowcasting, and scenario modeling._

NOTE: This repository is a WORK IN PROGRESS.

---

# Installation

Install `forecasttools` via:

```
pip3 install git+https://github.com/CDCgov/forecasttools-py@main
```

# Vignettes

* Format Arviz Forecast Output For FluSight Submission (In Progress)

# Datasets

`forecasttools` contains several datasets. These datasets aid with experimentation or are directly necessary to some of `forecasttools` utilities.

## Location Table

The location table contains abbreviations, codes, and extended names for the US jurisdictions for which the FluSight and COVID forecasting hubs require users to generate forecasts.

Shape: (58, 3)

| location_code | short_name | long_name                   |
| ---           | ---        | ---                         |
| str           | str        | str                         |
|---------------|------------|-----------------------------|
| US            | US         | United States               |
| 1             | AL         | Alabama                     |
| 2             | AK         | Alaska                      |
| 4             | AZ         | Arizona                     |
| 5             | AR         | Arkansas                    |
| 6             | CA         | California                  |
| 8             | CO         | Colorado                    |
| 9             | CT         | Connecticut                 |
| …            | …         | …                          |
| 56            | WY         | Wyoming                     |
| 60            | AS         | American Samoa              |
| 66            | GU         | Guam                        |
| 69            | MP         | Northern Mariana Islands    |
| 72            | PR         | Puerto Rico                 |
| 74            | UM         | U.S. Minor Outlying Islands |
| 78            | VI         | U.S. Virgin Islands         |

The location table is stored in `forecasttools` as a `polars` dataframe and is accessed via:

```python
import forecasttools
loc_table = forecasttools.location_table
```

Using `data.py`, the location table was created by running the following:

```python
make_census_dataset(
    file_save_path=os.path.join(os.getcwd(), "location_table.csv"),
)
```

## Example FluSight Hub Submission

The example FluSight submission comes from the [following 2023-24 submission](https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/cfa-flumech/2023-10-14-cfa-flumech.csv).

Shape: (4_876, 8)

| reference_date | target     | horizon | target_end_date | location | output_type | output_type_id | value |
|------------|------------|---------|------------|----------|------------|------------|------------|
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.01       | 7.670286   |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.025      | 9.968043   |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.05       | 12.022354  |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.1        | 14.497646  |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.15       | 16.119813  |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.2        | 17.670122  |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.25       | 19.125462  |
|            | hosp       |         |            |          |            |            |            |
| 2023-10-14 | wk inc flu | -1      | 2023-10-07 | 01       | quantile   | 0.3        | 20.443282  |
|            | hosp       |         |            |          |            |            |            |
| …         | …         | …      | …         | …       | …         | …         | …          |
| 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.75       | 1995.98533 |
|            | hosp       |         |            |          |            |            | 6          |
| 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.99       | 4761.75738 |
|            | hosp       |         |            |          |            |            | 5          |

The example FluSight submission is stored in `forecasttools` as a `polars` dataframe and is accessed via:

```python
import forecasttools
submission = forecasttools.example_flusight_submission
```

Using `data.py`, the example FluSight submission was created by running the following:

```python
get_and_save_flusight_submission(
    file_save_path=os.path.join(os.getcwd(), "example_flusight_submission.csv"),
)
```


## NHSN COVID And Flu Hospital Admissions

Hospital admissions data for fitting from NHSN for COVID and Flu is included in `forecasttools` as well, for user experimentation. This data is current as of `2024-04-27` and comes from the website [HealthData.gov COVID-19 Reported Patient Impact and Hospital Capacity by State Timeseries](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh). For influenza, the `previous_day_admission_influenza_confirmed` column is retained and for COVID the `previous_day_admission_adult_covid_confirmed` column is retained. As can be seen in the example below, some early dates for each jurisdiction do not have data.

Shape: (81_713, 3)

| state | date       | hosp |
| ---   | ---        | ---  |
| str   | str        | str  |
|-------|------------|------|
| AK    | 2020-03-23 | null |
| AK    | 2020-03-24 | null |
| AK    | 2020-03-25 | null |
| AK    | 2020-03-26 | null |
| AK    | 2020-03-27 | null |
| AK    | 2020-03-28 | null |
| AK    | 2020-03-29 | null |
| AK    | 2020-03-30 | null |
| …    | …         | …   |
| WY    | 2024-04-21 | 0    |
| WY    | 2024-04-22 | 2    |
| WY    | 2024-04-23 | 1    |
| WY    | 2024-04-24 | 1    |
| WY    | 2024-04-25 | 0    |
| WY    | 2024-04-26 | 0    |
| WY    | 2024-04-27 | 0    |


The fitting data is stored in `forecasttools` as a `polars` dataframe and is accessed via:

```python
import forecasttools


# access COVID data
covid_nhsn_data = forecasttools.nhsn_hosp_COVID

# access flu data
flu_nhsn_data = forecasttools.nhsn_hosp_flu
```

The data was created by placing a csv file called `NHSN_RAW_20240926.csv` (the full NHSN dataset) into `./forecasttools` and running, in `data.py`, the following:

```python
# generate COVID dataset
make_nshn_fitting_dataset(
    dataset="COVID",
    nhsn_dataset_path="NHSN_RAW_20240926.csv",
    file_save_path=os.path.join(os.getcwd(),"nhsn_hosp_COVID.csv")
)

# generate flu dataset
make_nshn_fitting_dataset(
    dataset="flu",
    nhsn_dataset_path="NHSN_RAW_20240926.csv",
    file_save_path=os.path.join(os.getcwd(),"nhsn_hosp_flu.csv")
)
```

## Influenza Hospitalizations Forecast

An example forecast stored in an Arviz `InferenceData` object is included for vignettes and user experimentation. This 28 day forecast for Texas was made using a spline regression model fitted to NHSN influenza data between 2022-08-08 and 2022-12-08. The `idata` object which includes the observed data and posterior predictive samples is given below:

```
Inference data with groups:
	> posterior
	> posterior_predictive
	> log_likelihood
	> sample_stats
	> prior
	> prior_predictive
	> observed_data
```

The forecast `idata` is accessed via:

```python
import forecasttools


idata = forecasttools.nhsn_flu_forecast
```

The forecast was generated following the creation of `nhsn_hosp_flu.csv` (see previous section) by running `data.py` with the following added:

```python
make_nhsn_fitted_forecast_idata(
    nhsn_dataset_path="nhsn_hosp_flu.csv",
    file_save_path=os.path.join(os.getcwd(), "example_flu_forecast.nc"),
    start_date"2022/08/08",
    end_date="2023/12/08",
    forecast_days=28,
    juris_subset=["TX"],
    create_save_directory=False,
    show_plot=True,
    save_idata=True
)
```

---

# CDC Open Source Considerations

**General disclaimer** This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Rules, Policy, And Collaboration

* [Open Practices](./rules-and-policies/open_practices.md)
* [Rules of Behavior](./rules-and-policies/rules_of_behavior.md)
* [Thanks and Acknowledgements](./rules-and-policies/thanks.md)
* [Disclaimer](DISCLAIMER.md)
* [Contribution Notice](CONTRIBUTING.md)
* [Code of Conduct](./rules-and-policies/code-of-conduct.md)


## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
