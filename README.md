# CFA Forecast Tools (Python)


<!-- To learn more about using Quarto for
render a GitHub README, see
<https://quarto.org/docs/output-formats/gfm.html>
-->

    polars.config.Config

Summary of `forecasttools-py`:

- A Python package.
- Primarily supports the Short Term Forecast’s team.
- Intended to support wider Real Time Monitoring branch operations.
- Has tools for pre- and post-processing.
  - Conversion of `az.InferenceData` forecast to Hubverse format.
  - Addition of time and or dates to `az.InferenceData`.

Notes:

- This repository is a WORK IN PROGRESS.

# Installation

Install `forecasttools` via:

    pip3 install git+https://github.com/CDCgov/forecasttools-py@main

# Vignettes

- [Format Arviz Forecast Output For FluSight
  Submission](https://github.com/CDCgov/forecasttools-py/blob/main/notebooks/flusight_from_idata.qmd)
- [Community Meeting Utilities Demonstration
  (2024-11-19)](https://github.com/CDCgov/forecasttools-py/blob/main/notebooks/forecasttools_community_demo_2024-11-19.qmd)

*Coming soon as webpages, once [Issue
26](https://github.com/CDCgov/forecasttools-py/issues/26) is completed*.

# Datasets

Within `forecasttools-py`, one finds several packaged datasets. These
datasets can aid with experimentation; some are directly necessary to
other utilities provided by `forecasttools-py`.

``` python
import forecasttools
```

Summary of datasets:

- `forecasttools.location_table`
  - A Polars dataframe of location abbreviations, codes, and names for
    Hubverse formatted forecast submissions.
- `forecasttools.example_flusight_submission`
  - An example Hubverse formatted influenza forecast submission (as a
    Polars dataframe) submitted to the FluSight Hub.
- `forecasttools.nhsn_hosp_COVID`
  - A Polars dataframe of NHSN COVID hospital admissions data.
- `forecasttools.nhsn_hosp_flu`
  - A Polars dataframe of NHSN influenza hospital admissions data.
- `forecasttools.nhsn_flu_forecast_wo_dates`
  - An `az.InferenceData` object containing a forecast made using NSHN
    influenza data for Texas.
- `forecasttools.nhsn_flu_forecast_w_dates`
  - An modified (with dates as coordinates) `az.InferenceData` object
    containing a forecast made using NSHN influenza data for Texas.

See below for more information on the datasets.

## Location Table

The location table contains abbreviations, codes, and extended names for
the US jurisdictions for which the FluSight and COVID forecasting hubs
require users to generate forecasts.

The location table is stored in `forecasttools` as a `polars` dataframe
and is accessed via:

``` python
loc_table = forecasttools.location_table
print(loc_table)
```

shape: (58, 3) \| location_code \| short_name \| long_name \| \| — \| —
\| — \| \| str \| str \| str \| \|—————\|————\|—————————–\| \| US \| US
\| United States \| \| 01 \| AL \| Alabama \| \| 02 \| AK \| Alaska \|
\| 04 \| AZ \| Arizona \| \| 05 \| AR \| Arkansas \| \| … \| … \| … \|
\| 66 \| GU \| Guam \| \| 69 \| MP \| Northern Mariana Islands \| \| 72
\| PR \| Puerto Rico \| \| 74 \| UM \| U.S. Minor Outlying Islands \| \|
78 \| VI \| U.S. Virgin Islands \|

Using `./forecasttools/data.py`, the location table was created by
running the following:

``` python
make_census_dataset(
    file_save_path=os.path.join(
        os.getcwd(),
        "location_table.csv"
    ),
)
```

## Example FluSight Hub Submission

The example FluSight submission comes from the [following 2023-24
submission](https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/cfa-flumech/2023-10-14-cfa-flumech.csv).

The example FluSight submission is stored in `forecasttools` as a
`polars` dataframe and is accessed via:

``` python
submission = forecasttools.example_flusight_submission
print(submission)
```

    shape: (4_876, 8)
    | reference_ | target     | horizon | target_end | location | output_typ | output_typ | value      |
    | date       | ---        | ---     | _date      | ---      | e          | e_id       | ---        |
    | ---        | str        | i64     | ---        | str      | ---        | ---        | f64        |
    | str        |            |         | str        |          | str        | f64        |            |
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
    | ...        | ...        | ...     | ...        | ...      | ...        | ...        | ...        |
    | 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.85       | 2451.87489 |
    |            | hosp       |         |            |          |            |            | 9          |
    | 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.9        | 2806.92858 |
    |            | hosp       |         |            |          |            |            | 8          |
    | 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.95       | 3383.74799 |
    |            | hosp       |         |            |          |            |            |            |
    | 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.975      | 3940.39253 |
    |            | hosp       |         |            |          |            |            | 6          |
    | 2023-10-14 | wk inc flu | 2       | 2023-10-28 | US       | quantile   | 0.99       | 4761.75738 |
    |            | hosp       |         |            |          |            |            | 5          |

Using `data.py`, the example FluSight submission was created by running
the following:

``` python
get_and_save_flusight_submission(
    file_save_path=os.path.join(
        os.getcwd(),
        "example_flusight_submission.csv"
    ),
)
```

## NHSN COVID And Flu Hospital Admissions

NHSN hospital admissions fitting data for COVID and Flu is included in
`forecasttools-py` as well, for user experimentation.

This data:

- Is current as of `2024-04-27`
- Comes from the website [HealthData.gov COVID-19 Reported Patient
  Impact and Hospital Capacity by State
  Timeseries](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh).

For influenza, the `previous_day_admission_influenza_confirmed` column
is retained and for COVID the
`previous_day_admission_adult_covid_confirmed` column is retained. As
can be seen in the example below, some early dates for each jurisdiction
do not have data.

The fitting data is stored in `forecasttools` as a `polars` dataframe
and is accessed via:

``` python
# access COVID data
covid_nhsn_data = forecasttools.nhsn_hosp_COVID

# access flu data
flu_nhsn_data = forecasttools.nhsn_hosp_flu

# display flu data
flu_nhsn_data
```

<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (81_713, 3)</small>

| state | date         | hosp |
|-------|--------------|------|
| str   | str          | str  |
| "AK"  | "2020-03-23" | null |
| "AK"  | "2020-03-24" | null |
| "AK"  | "2020-03-25" | null |
| "AK"  | "2020-03-26" | null |
| "AK"  | "2020-03-27" | null |
| …     | …            | …    |
| "WY"  | "2024-04-23" | "1"  |
| "WY"  | "2024-04-24" | "1"  |
| "WY"  | "2024-04-25" | "0"  |
| "WY"  | "2024-04-26" | "0"  |
| "WY"  | "2024-04-27" | "0"  |

</div>

The data was created by placing a csv file called
`NHSN_RAW_20240926.csv` (the full NHSN dataset) into `./forecasttools/`
and running, in `data.py`, the following:

``` python
# generate COVID dataset
make_nshn_fitting_dataset(
    dataset="COVID",
    nhsn_dataset_path="NHSN_RAW_20240926.csv",
    file_save_path=os.path.join(
        os.getcwd(),
        "nhsn_hosp_COVID.csv"
    )
)

# generate flu dataset
make_nshn_fitting_dataset(
    dataset="flu",
    nhsn_dataset_path="NHSN_RAW_20240926.csv",
    file_save_path=os.path.join(
        os.getcwd(),
        "nhsn_hosp_flu.csv"
    )
)
```

## Influenza Hospitalizations Forecast(s)

Two example forecasts stored in Arviz `InferenceData` objects are
included for vignettes and user experimentation. Both are 28 day
influenza hospital admissions forecasts for Texas made using a spline
regression model fitted to NHSN data between 2022-08-08 and 2022-12-08.
The only difference between the forecasts is that
`example_flu_forecast_w_dates.nc` has had dates added as its coordinates
(this is not a native Arviz feature). The `idata` objects which includes
the observed data and posterior predictive samples is given below:

    Inference data with groups:
        > posterior
        > posterior_predictive
        > log_likelihood
        > sample_stats
        > prior
        > prior_predictive
        > observed_data

The forecast `idata`s are accessed via:

``` python
# idata with dates as coordinates
idata_w_dates = forecasttools.nhsn_flu_forecast_w_dates
idata_w_dates
```

            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">
              &#10;            <li class = "xr-section-item">
                  <input id="idata_posteriora38a1ccc-c36a-42f4-b332-8a4023b54619" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriora38a1ccc-c36a-42f4-b332-8a4023b54619" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 48kB
Dimensions:            (chain: 1, draw: 1000, beta_coeffs_dim_0: 8)
Coordinates:
  * chain              (chain) int64 8B 0
  * draw               (draw) int64 8kB 0 1 2 3 4 5 ... 994 995 996 997 998 999
  * beta_coeffs_dim_0  (beta_coeffs_dim_0) int64 64B 0 1 2 3 4 5 6 7
Data variables:
    alpha              (chain, draw) float32 4kB ...
    beta_coeffs        (chain, draw, beta_coeffs_dim_0) float32 32kB ...
    shift              (chain, draw) float32 4kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.119636+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-3f93ae6c-9495-4f2b-8cfb-52ccc5919b83' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3f93ae6c-9495-4f2b-8cfb-52ccc5919b83' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-58720433-e367-4e00-8351-34c3900a14b4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-58720433-e367-4e00-8351-34c3900a14b4' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-05a53f36-83e2-4601-bbca-bda076b02d86' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-05a53f36-83e2-4601-bbca-bda076b02d86' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f19426aa-ac13-4274-95b6-f10712758096' class='xr-var-data-in' type='checkbox'><label for='data-f19426aa-ac13-4274-95b6-f10712758096' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-86c51aee-ca4b-4b34-81fd-1453cb10aec0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-86c51aee-ca4b-4b34-81fd-1453cb10aec0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-82833f2d-1aad-4589-aa8d-07c0eb9eb700' class='xr-var-data-in' type='checkbox'><label for='data-82833f2d-1aad-4589-aa8d-07c0eb9eb700' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-7ade699d-7ac8-48e9-8770-9f77a47e415a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7ade699d-7ac8-48e9-8770-9f77a47e415a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80098cf3-535f-4231-ba69-f2e401d190e8' class='xr-var-data-in' type='checkbox'><label for='data-80098cf3-535f-4231-ba69-f2e401d190e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-30066697-0ab4-4433-9c53-503e28f422c8' class='xr-section-summary-in' type='checkbox'  checked><label for='section-30066697-0ab4-4433-9c53-503e28f422c8' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-03e52ff0-5e17-4c5d-9f77-ecaebff5f790' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-03e52ff0-5e17-4c5d-9f77-ecaebff5f790' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c1b9eff7-b18f-4df8-8769-f917e3e7b658' class='xr-var-data-in' type='checkbox'><label for='data-c1b9eff7-b18f-4df8-8769-f917e3e7b658' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-cbbb3ea4-beb2-4455-9bba-6274f0cbc7cf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cbbb3ea4-beb2-4455-9bba-6274f0cbc7cf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ab8ecb79-a1fe-4c8a-ba8e-2157e346345a' class='xr-var-data-in' type='checkbox'><label for='data-ab8ecb79-a1fe-4c8a-ba8e-2157e346345a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-e9fd4a08-a8d1-469d-886a-57a582fe48fd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9fd4a08-a8d1-469d-886a-57a582fe48fd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8cb67079-0816-4274-954c-d8b192f7482e' class='xr-var-data-in' type='checkbox'><label for='data-8cb67079-0816-4274-954c-d8b192f7482e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-83eade80-3c4a-4ec0-b15b-cfeda553c7a0' class='xr-section-summary-in' type='checkbox'  ><label for='section-83eade80-3c4a-4ec0-b15b-cfeda553c7a0' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0d1bf60d-dbf5-4aec-a88b-6eb1c87c5395' class='xr-index-data-in' type='checkbox'/><label for='index-0d1bf60d-dbf5-4aec-a88b-6eb1c87c5395' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-57c3d0dd-f0cb-4354-979b-680e6b1e84a6' class='xr-index-data-in' type='checkbox'/><label for='index-57c3d0dd-f0cb-4354-979b-680e6b1e84a6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-488df5a1-735f-488c-8c1c-08a37060b3da' class='xr-index-data-in' type='checkbox'/><label for='index-488df5a1-735f-488c-8c1c-08a37060b3da' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d80f2a02-5643-4632-8ffa-92ef5903626a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d80f2a02-5643-4632-8ffa-92ef5903626a' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.119636+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_posterior_predictivee665f728-f6e8-42f0-b79e-d9d6cf3e19ea" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior_predictivee665f728-f6e8-42f0-b79e-d9d6cf3e19ea" class = "xr-section-summary">posterior_predictive</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 613kB
Dimensions:    (chain: 1, draw: 1000, obs_dim_0: 151)
Coordinates:
  * chain      (chain) int64 8B 0
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * obs_dim_0  (obs_dim_0) datetime64[ns] 1kB 2022-08-08 ... 2023-01-05
Data variables:
    obs        (chain, draw, obs_dim_0) int32 604kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.236298+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-efbde647-4224-4670-a1ee-dba1d2fed106' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-efbde647-4224-4670-a1ee-dba1d2fed106' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 151</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-a9c7520a-4a3b-4bd3-93b7-8efc215d8ff9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a9c7520a-4a3b-4bd3-93b7-8efc215d8ff9' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-81f09d98-67db-4a6d-94a3-371a6e627df5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-81f09d98-67db-4a6d-94a3-371a6e627df5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1e1da83b-caad-49a9-9009-1da18099441c' class='xr-var-data-in' type='checkbox'><label for='data-1e1da83b-caad-49a9-9009-1da18099441c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-5d24532a-f89d-4b97-8292-708028cb23d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5d24532a-f89d-4b97-8292-708028cb23d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c54fd975-fc08-434d-8812-bfc6f38fd1a6' class='xr-var-data-in' type='checkbox'><label for='data-c54fd975-fc08-434d-8812-bfc6f38fd1a6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2023-01-05</div><input id='attrs-0f43a788-825b-45ab-8304-e9b84ebb9888' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0f43a788-825b-45ab-8304-e9b84ebb9888' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-33ca8b9a-10b4-459e-9750-11236632244b' class='xr-var-data-in' type='checkbox'><label for='data-33ca8b9a-10b4-459e-9750-11236632244b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
       &#x27;2022-08-10T00:00:00.000000000&#x27;, &#x27;2022-08-11T00:00:00.000000000&#x27;,
       &#x27;2022-08-12T00:00:00.000000000&#x27;, &#x27;2022-08-13T00:00:00.000000000&#x27;,
       &#x27;2022-08-14T00:00:00.000000000&#x27;, &#x27;2022-08-15T00:00:00.000000000&#x27;,
       &#x27;2022-08-16T00:00:00.000000000&#x27;, &#x27;2022-08-17T00:00:00.000000000&#x27;,
       &#x27;2022-08-18T00:00:00.000000000&#x27;, &#x27;2022-08-19T00:00:00.000000000&#x27;,
       &#x27;2022-08-20T00:00:00.000000000&#x27;, &#x27;2022-08-21T00:00:00.000000000&#x27;,
       &#x27;2022-08-22T00:00:00.000000000&#x27;, &#x27;2022-08-23T00:00:00.000000000&#x27;,
       &#x27;2022-08-24T00:00:00.000000000&#x27;, &#x27;2022-08-25T00:00:00.000000000&#x27;,
       &#x27;2022-08-26T00:00:00.000000000&#x27;, &#x27;2022-08-27T00:00:00.000000000&#x27;,
       &#x27;2022-08-28T00:00:00.000000000&#x27;, &#x27;2022-08-29T00:00:00.000000000&#x27;,
       &#x27;2022-08-30T00:00:00.000000000&#x27;, &#x27;2022-08-31T00:00:00.000000000&#x27;,
       &#x27;2022-09-01T00:00:00.000000000&#x27;, &#x27;2022-09-02T00:00:00.000000000&#x27;,
       &#x27;2022-09-03T00:00:00.000000000&#x27;, &#x27;2022-09-04T00:00:00.000000000&#x27;,
       &#x27;2022-09-05T00:00:00.000000000&#x27;, &#x27;2022-09-06T00:00:00.000000000&#x27;,
       &#x27;2022-09-07T00:00:00.000000000&#x27;, &#x27;2022-09-08T00:00:00.000000000&#x27;,
       &#x27;2022-09-09T00:00:00.000000000&#x27;, &#x27;2022-09-10T00:00:00.000000000&#x27;,
       &#x27;2022-09-11T00:00:00.000000000&#x27;, &#x27;2022-09-12T00:00:00.000000000&#x27;,
       &#x27;2022-09-13T00:00:00.000000000&#x27;, &#x27;2022-09-14T00:00:00.000000000&#x27;,
       &#x27;2022-09-15T00:00:00.000000000&#x27;, &#x27;2022-09-16T00:00:00.000000000&#x27;,
       &#x27;2022-09-17T00:00:00.000000000&#x27;, &#x27;2022-09-18T00:00:00.000000000&#x27;,
       &#x27;2022-09-19T00:00:00.000000000&#x27;, &#x27;2022-09-20T00:00:00.000000000&#x27;,
       &#x27;2022-09-21T00:00:00.000000000&#x27;, &#x27;2022-09-22T00:00:00.000000000&#x27;,
       &#x27;2022-09-23T00:00:00.000000000&#x27;, &#x27;2022-09-24T00:00:00.000000000&#x27;,
       &#x27;2022-09-25T00:00:00.000000000&#x27;, &#x27;2022-09-26T00:00:00.000000000&#x27;,
       &#x27;2022-09-27T00:00:00.000000000&#x27;, &#x27;2022-09-28T00:00:00.000000000&#x27;,
       &#x27;2022-09-29T00:00:00.000000000&#x27;, &#x27;2022-09-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-01T00:00:00.000000000&#x27;, &#x27;2022-10-02T00:00:00.000000000&#x27;,
       &#x27;2022-10-03T00:00:00.000000000&#x27;, &#x27;2022-10-04T00:00:00.000000000&#x27;,
       &#x27;2022-10-05T00:00:00.000000000&#x27;, &#x27;2022-10-06T00:00:00.000000000&#x27;,
       &#x27;2022-10-07T00:00:00.000000000&#x27;, &#x27;2022-10-08T00:00:00.000000000&#x27;,
       &#x27;2022-10-09T00:00:00.000000000&#x27;, &#x27;2022-10-10T00:00:00.000000000&#x27;,
       &#x27;2022-10-11T00:00:00.000000000&#x27;, &#x27;2022-10-12T00:00:00.000000000&#x27;,
       &#x27;2022-10-13T00:00:00.000000000&#x27;, &#x27;2022-10-14T00:00:00.000000000&#x27;,
       &#x27;2022-10-15T00:00:00.000000000&#x27;, &#x27;2022-10-16T00:00:00.000000000&#x27;,
       &#x27;2022-10-17T00:00:00.000000000&#x27;, &#x27;2022-10-18T00:00:00.000000000&#x27;,
       &#x27;2022-10-19T00:00:00.000000000&#x27;, &#x27;2022-10-20T00:00:00.000000000&#x27;,
       &#x27;2022-10-21T00:00:00.000000000&#x27;, &#x27;2022-10-22T00:00:00.000000000&#x27;,
       &#x27;2022-10-23T00:00:00.000000000&#x27;, &#x27;2022-10-24T00:00:00.000000000&#x27;,
       &#x27;2022-10-25T00:00:00.000000000&#x27;, &#x27;2022-10-26T00:00:00.000000000&#x27;,
       &#x27;2022-10-27T00:00:00.000000000&#x27;, &#x27;2022-10-28T00:00:00.000000000&#x27;,
       &#x27;2022-10-29T00:00:00.000000000&#x27;, &#x27;2022-10-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-31T00:00:00.000000000&#x27;, &#x27;2022-11-01T00:00:00.000000000&#x27;,
       &#x27;2022-11-02T00:00:00.000000000&#x27;, &#x27;2022-11-03T00:00:00.000000000&#x27;,
       &#x27;2022-11-04T00:00:00.000000000&#x27;, &#x27;2022-11-05T00:00:00.000000000&#x27;,
       &#x27;2022-11-06T00:00:00.000000000&#x27;, &#x27;2022-11-07T00:00:00.000000000&#x27;,
       &#x27;2022-11-08T00:00:00.000000000&#x27;, &#x27;2022-11-09T00:00:00.000000000&#x27;,
       &#x27;2022-11-10T00:00:00.000000000&#x27;, &#x27;2022-11-11T00:00:00.000000000&#x27;,
       &#x27;2022-11-12T00:00:00.000000000&#x27;, &#x27;2022-11-13T00:00:00.000000000&#x27;,
       &#x27;2022-11-14T00:00:00.000000000&#x27;, &#x27;2022-11-15T00:00:00.000000000&#x27;,
       &#x27;2022-11-16T00:00:00.000000000&#x27;, &#x27;2022-11-17T00:00:00.000000000&#x27;,
       &#x27;2022-11-18T00:00:00.000000000&#x27;, &#x27;2022-11-19T00:00:00.000000000&#x27;,
       &#x27;2022-11-20T00:00:00.000000000&#x27;, &#x27;2022-11-21T00:00:00.000000000&#x27;,
       &#x27;2022-11-22T00:00:00.000000000&#x27;, &#x27;2022-11-23T00:00:00.000000000&#x27;,
       &#x27;2022-11-24T00:00:00.000000000&#x27;, &#x27;2022-11-25T00:00:00.000000000&#x27;,
       &#x27;2022-11-26T00:00:00.000000000&#x27;, &#x27;2022-11-27T00:00:00.000000000&#x27;,
       &#x27;2022-11-28T00:00:00.000000000&#x27;, &#x27;2022-11-29T00:00:00.000000000&#x27;,
       &#x27;2022-11-30T00:00:00.000000000&#x27;, &#x27;2022-12-01T00:00:00.000000000&#x27;,
       &#x27;2022-12-02T00:00:00.000000000&#x27;, &#x27;2022-12-03T00:00:00.000000000&#x27;,
       &#x27;2022-12-04T00:00:00.000000000&#x27;, &#x27;2022-12-05T00:00:00.000000000&#x27;,
       &#x27;2022-12-06T00:00:00.000000000&#x27;, &#x27;2022-12-07T00:00:00.000000000&#x27;,
       &#x27;2022-12-08T00:00:00.000000000&#x27;, &#x27;2022-12-09T00:00:00.000000000&#x27;,
       &#x27;2022-12-10T00:00:00.000000000&#x27;, &#x27;2022-12-11T00:00:00.000000000&#x27;,
       &#x27;2022-12-12T00:00:00.000000000&#x27;, &#x27;2022-12-13T00:00:00.000000000&#x27;,
       &#x27;2022-12-14T00:00:00.000000000&#x27;, &#x27;2022-12-15T00:00:00.000000000&#x27;,
       &#x27;2022-12-16T00:00:00.000000000&#x27;, &#x27;2022-12-17T00:00:00.000000000&#x27;,
       &#x27;2022-12-18T00:00:00.000000000&#x27;, &#x27;2022-12-19T00:00:00.000000000&#x27;,
       &#x27;2022-12-20T00:00:00.000000000&#x27;, &#x27;2022-12-21T00:00:00.000000000&#x27;,
       &#x27;2022-12-22T00:00:00.000000000&#x27;, &#x27;2022-12-23T00:00:00.000000000&#x27;,
       &#x27;2022-12-24T00:00:00.000000000&#x27;, &#x27;2022-12-25T00:00:00.000000000&#x27;,
       &#x27;2022-12-26T00:00:00.000000000&#x27;, &#x27;2022-12-27T00:00:00.000000000&#x27;,
       &#x27;2022-12-28T00:00:00.000000000&#x27;, &#x27;2022-12-29T00:00:00.000000000&#x27;,
       &#x27;2022-12-30T00:00:00.000000000&#x27;, &#x27;2022-12-31T00:00:00.000000000&#x27;,
       &#x27;2023-01-01T00:00:00.000000000&#x27;, &#x27;2023-01-02T00:00:00.000000000&#x27;,
       &#x27;2023-01-03T00:00:00.000000000&#x27;, &#x27;2023-01-04T00:00:00.000000000&#x27;,
       &#x27;2023-01-05T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-959b8e1a-13af-4baa-b000-b3b34a2289fc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-959b8e1a-13af-4baa-b000-b3b34a2289fc' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-e7185a93-303c-4bbf-a007-5fcc1f31c345' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e7185a93-303c-4bbf-a007-5fcc1f31c345' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d0780173-56b3-4276-b336-8fa6e4b69449' class='xr-var-data-in' type='checkbox'><label for='data-d0780173-56b3-4276-b336-8fa6e4b69449' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[151000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b604b2a0-87e3-4480-ada0-6c8b462256aa' class='xr-section-summary-in' type='checkbox'  ><label for='section-b604b2a0-87e3-4480-ada0-6c8b462256aa' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-539c95d9-9131-479f-b378-1b703da1dca5' class='xr-index-data-in' type='checkbox'/><label for='index-539c95d9-9131-479f-b378-1b703da1dca5' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-7ea2d3b3-a5b3-460f-85a3-83f261dec764' class='xr-index-data-in' type='checkbox'/><label for='index-7ea2d3b3-a5b3-460f-85a3-83f261dec764' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9268f664-ff67-412f-ae35-9e1fa4c68697' class='xr-index-data-in' type='checkbox'/><label for='index-9268f664-ff67-412f-ae35-9e1fa4c68697' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-12-27&#x27;, &#x27;2022-12-28&#x27;, &#x27;2022-12-29&#x27;, &#x27;2022-12-30&#x27;,
               &#x27;2022-12-31&#x27;, &#x27;2023-01-01&#x27;, &#x27;2023-01-02&#x27;, &#x27;2023-01-03&#x27;,
               &#x27;2023-01-04&#x27;, &#x27;2023-01-05&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=151, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e34aa635-dad3-415a-90e9-20c2cff041c0' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e34aa635-dad3-415a-90e9-20c2cff041c0' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.236298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_log_likelihood99facd9a-657e-41fb-afa7-cdd04b18312e" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_log_likelihood99facd9a-657e-41fb-afa7-cdd04b18312e" class = "xr-section-summary">log_likelihood</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 501kB
Dimensions:    (chain: 1, draw: 1000, obs_dim_0: 123)
Coordinates:
  * chain      (chain) int64 8B 0
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * obs_dim_0  (obs_dim_0) int64 984B 0 1 2 3 4 5 6 ... 117 118 119 120 121 122
Data variables:
    obs        (chain, draw, obs_dim_0) float32 492kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.235298+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-a39551e3-4009-4b87-9163-27d92682a1da' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-a39551e3-4009-4b87-9163-27d92682a1da' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-44f9b009-aaf9-46a4-9be4-28e5b1a85f7b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-44f9b009-aaf9-46a4-9be4-28e5b1a85f7b' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-2ec8cfed-a12a-4071-81ea-84ee902751ab' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2ec8cfed-a12a-4071-81ea-84ee902751ab' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-43ff71cc-5b0f-4546-aac5-43ce8a1baa1b' class='xr-var-data-in' type='checkbox'><label for='data-43ff71cc-5b0f-4546-aac5-43ce8a1baa1b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-b7b81bcb-a3a5-45e5-89fe-77572d0afc08' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7b81bcb-a3a5-45e5-89fe-77572d0afc08' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7e3798e5-416d-40c6-8d96-d25205125047' class='xr-var-data-in' type='checkbox'><label for='data-7e3798e5-416d-40c6-8d96-d25205125047' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 118 119 120 121 122</div><input id='attrs-8ab784a2-18cd-4f85-b56a-ea34e5692aa7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8ab784a2-18cd-4f85-b56a-ea34e5692aa7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bb72c246-a03c-42db-9c5b-a18fbe92506a' class='xr-var-data-in' type='checkbox'><label for='data-bb72c246-a03c-42db-9c5b-a18fbe92506a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5a8046cd-7232-41b7-a0fc-4659872ae3c0' class='xr-section-summary-in' type='checkbox'  checked><label for='section-5a8046cd-7232-41b7-a0fc-4659872ae3c0' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-9551d41b-7eef-42c5-beb4-42ce011ca16a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9551d41b-7eef-42c5-beb4-42ce011ca16a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb89ec45-a536-4c05-bdf1-4d237b006c6b' class='xr-var-data-in' type='checkbox'><label for='data-eb89ec45-a536-4c05-bdf1-4d237b006c6b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-0f1c344d-a324-444a-94f1-cd8ccd3ed576' class='xr-section-summary-in' type='checkbox'  ><label for='section-0f1c344d-a324-444a-94f1-cd8ccd3ed576' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9ef75c23-5de3-4ee8-9d88-2e14ce888feb' class='xr-index-data-in' type='checkbox'/><label for='index-9ef75c23-5de3-4ee8-9d88-2e14ce888feb' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c44e3b4d-3556-4577-b850-1f2f8bbd46de' class='xr-index-data-in' type='checkbox'/><label for='index-c44e3b4d-3556-4577-b850-1f2f8bbd46de' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c6013f00-8ea5-455b-a302-2c69afc5fe02' class='xr-index-data-in' type='checkbox'/><label for='index-c6013f00-8ea5-455b-a302-2c69afc5fe02' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
      dtype=&#x27;int64&#x27;, name=&#x27;obs_dim_0&#x27;, length=123))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-15491aae-f9de-44fa-aebf-22b16ef65f91' class='xr-section-summary-in' type='checkbox'  checked><label for='section-15491aae-f9de-44fa-aebf-22b16ef65f91' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.235298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_sample_stats372e5076-01af-4722-a7a4-460b40d94cd3" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_stats372e5076-01af-4722-a7a4-460b40d94cd3" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 9kB
Dimensions:    (chain: 1, draw: 1000)
Coordinates:
  * chain      (chain) int64 8B 0
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
Data variables:
    diverging  (chain, draw) bool 1kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.122620+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-375816a9-c07e-4dfe-8394-af94a9e412c0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-375816a9-c07e-4dfe-8394-af94a9e412c0' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-496be1c8-574a-4c88-b1e8-60a329128578' class='xr-section-summary-in' type='checkbox'  checked><label for='section-496be1c8-574a-4c88-b1e8-60a329128578' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-b156cffc-42bd-44c1-95a2-fd11908a222b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b156cffc-42bd-44c1-95a2-fd11908a222b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c2f6e400-ec16-42a5-a5c5-f1a7c8a06e21' class='xr-var-data-in' type='checkbox'><label for='data-c2f6e400-ec16-42a5-a5c5-f1a7c8a06e21' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-ec8f1a55-a404-44b2-8ca2-2488a5199822' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ec8f1a55-a404-44b2-8ca2-2488a5199822' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0649ae5f-211a-4aea-9acd-35f69eeecd0f' class='xr-var-data-in' type='checkbox'><label for='data-0649ae5f-211a-4aea-9acd-35f69eeecd0f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-57ece44a-66f7-4311-9f90-78671c5dbeb3' class='xr-section-summary-in' type='checkbox'  checked><label for='section-57ece44a-66f7-4311-9f90-78671c5dbeb3' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-b0708a61-483b-4bf3-931b-90b9797098c6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b0708a61-483b-4bf3-931b-90b9797098c6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7a8f8bbe-4446-4f49-b108-98e1c74d1198' class='xr-var-data-in' type='checkbox'><label for='data-7a8f8bbe-4446-4f49-b108-98e1c74d1198' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=bool]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ec01e8a5-3fc0-4330-a451-b5fd3dc66525' class='xr-section-summary-in' type='checkbox'  ><label for='section-ec01e8a5-3fc0-4330-a451-b5fd3dc66525' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4bd08830-5969-4140-af77-0a8392076ed7' class='xr-index-data-in' type='checkbox'/><label for='index-4bd08830-5969-4140-af77-0a8392076ed7' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-ee9a33c5-f6cf-47b4-918d-930c5c3c786d' class='xr-index-data-in' type='checkbox'/><label for='index-ee9a33c5-f6cf-47b4-918d-930c5c3c786d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-740024d4-8083-46d7-8998-72c246bf20e3' class='xr-section-summary-in' type='checkbox'  checked><label for='section-740024d4-8083-46d7-8998-72c246bf20e3' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.122620+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_priore0e8647a-572e-4ca9-9f1d-1be95cb403ee" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_priore0e8647a-572e-4ca9-9f1d-1be95cb403ee" class = "xr-section-summary">prior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 48kB
Dimensions:            (chain: 1, draw: 1000, beta_coeffs_dim_0: 8)
Coordinates:
  * chain              (chain) int64 8B 0
  * draw               (draw) int64 8kB 0 1 2 3 4 5 ... 994 995 996 997 998 999
  * beta_coeffs_dim_0  (beta_coeffs_dim_0) int64 64B 0 1 2 3 4 5 6 7
Data variables:
    alpha              (chain, draw) float32 4kB ...
    beta_coeffs        (chain, draw, beta_coeffs_dim_0) float32 32kB ...
    shift              (chain, draw) float32 4kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.237407+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-692eff48-03bd-4815-aaf2-aab0710fcd11' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-692eff48-03bd-4815-aaf2-aab0710fcd11' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-11470455-a3b6-4bb7-aa3a-62eaf35b3ecc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-11470455-a3b6-4bb7-aa3a-62eaf35b3ecc' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-6bdb9427-f3cd-4bc4-a700-10f467e48ee2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6bdb9427-f3cd-4bc4-a700-10f467e48ee2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ebfdce1a-f5f4-4b06-95a1-9c39aa2de4da' class='xr-var-data-in' type='checkbox'><label for='data-ebfdce1a-f5f4-4b06-95a1-9c39aa2de4da' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-c1e98eb2-c3eb-4229-a703-c130aa9bdb22' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c1e98eb2-c3eb-4229-a703-c130aa9bdb22' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bcd44277-1d69-43fa-b432-c9695350ab4b' class='xr-var-data-in' type='checkbox'><label for='data-bcd44277-1d69-43fa-b432-c9695350ab4b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-e8ae5740-17da-4c5c-898f-04e7b2d25e15' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e8ae5740-17da-4c5c-898f-04e7b2d25e15' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-013ce5da-e66c-4eb8-bcd8-cbaff898e4f3' class='xr-var-data-in' type='checkbox'><label for='data-013ce5da-e66c-4eb8-bcd8-cbaff898e4f3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d39b95d0-9070-400b-9779-b6fd9a929b0c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d39b95d0-9070-400b-9779-b6fd9a929b0c' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-c0ef6459-4a2a-4fd6-a437-299f4805c4a8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c0ef6459-4a2a-4fd6-a437-299f4805c4a8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c487de80-544b-4712-b71f-944fec647d20' class='xr-var-data-in' type='checkbox'><label for='data-c487de80-544b-4712-b71f-944fec647d20' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-d3acdee6-c0cc-48f8-89db-e8927b971024' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d3acdee6-c0cc-48f8-89db-e8927b971024' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d0c542f0-d3ae-4abe-a132-7a23ce982e02' class='xr-var-data-in' type='checkbox'><label for='data-d0c542f0-d3ae-4abe-a132-7a23ce982e02' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-d63525b5-e53c-4b2c-8f69-042c100569d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d63525b5-e53c-4b2c-8f69-042c100569d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a09f38f9-cf1b-476a-9006-1537ef1349e3' class='xr-var-data-in' type='checkbox'><label for='data-a09f38f9-cf1b-476a-9006-1537ef1349e3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-faccc653-78a6-4439-8b0e-9ba38a6c9c02' class='xr-section-summary-in' type='checkbox'  ><label for='section-faccc653-78a6-4439-8b0e-9ba38a6c9c02' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-206d784c-405f-46c5-8d74-380bcde5374d' class='xr-index-data-in' type='checkbox'/><label for='index-206d784c-405f-46c5-8d74-380bcde5374d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-32a01f86-4517-449d-9302-e469c8ded8b5' class='xr-index-data-in' type='checkbox'/><label for='index-32a01f86-4517-449d-9302-e469c8ded8b5' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-71879c3f-25e2-4a7f-9cbd-2ef2322cceb6' class='xr-index-data-in' type='checkbox'/><label for='index-71879c3f-25e2-4a7f-9cbd-2ef2322cceb6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f64e7341-6929-4a09-b770-6ab30e4294a7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f64e7341-6929-4a09-b770-6ab30e4294a7' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.237407+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_prior_predictive4a851fb8-955b-474b-b93c-a2066577aecc" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_prior_predictive4a851fb8-955b-474b-b93c-a2066577aecc" class = "xr-section-summary">prior_predictive</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 501kB
Dimensions:    (chain: 1, draw: 1000, obs_dim_0: 123)
Coordinates:
  * chain      (chain) int64 8B 0
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * obs_dim_0  (obs_dim_0) datetime64[ns] 984B 2022-08-08 ... 2022-12-08
Data variables:
    obs        (chain, draw, obs_dim_0) int32 492kB ...
Attributes:
    created_at:                 2024-10-24T16:45:20.238442+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-ac7b0eb5-9ceb-43d7-9caf-9792e91edaf7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ac7b0eb5-9ceb-43d7-9caf-9792e91edaf7' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-35d83cb2-31e8-4ebd-b846-1c32fd5696c2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-35d83cb2-31e8-4ebd-b846-1c32fd5696c2' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-2947a1a4-6ad3-4562-87d2-e38ce8498c9f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2947a1a4-6ad3-4562-87d2-e38ce8498c9f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9c51646d-1455-4144-a14d-f4b3be0f8323' class='xr-var-data-in' type='checkbox'><label for='data-9c51646d-1455-4144-a14d-f4b3be0f8323' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-16fb6877-bf3e-416f-b0fb-982cbfc00d23' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-16fb6877-bf3e-416f-b0fb-982cbfc00d23' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b74fe634-1b19-4c78-9407-c4de1a7c51ea' class='xr-var-data-in' type='checkbox'><label for='data-b74fe634-1b19-4c78-9407-c4de1a7c51ea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-53684a90-21e8-485e-b677-43eafbeab5c3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-53684a90-21e8-485e-b677-43eafbeab5c3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a9b3d88a-c26a-45b5-a79e-793859029f24' class='xr-var-data-in' type='checkbox'><label for='data-a9b3d88a-c26a-45b5-a79e-793859029f24' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
       &#x27;2022-08-10T00:00:00.000000000&#x27;, &#x27;2022-08-11T00:00:00.000000000&#x27;,
       &#x27;2022-08-12T00:00:00.000000000&#x27;, &#x27;2022-08-13T00:00:00.000000000&#x27;,
       &#x27;2022-08-14T00:00:00.000000000&#x27;, &#x27;2022-08-15T00:00:00.000000000&#x27;,
       &#x27;2022-08-16T00:00:00.000000000&#x27;, &#x27;2022-08-17T00:00:00.000000000&#x27;,
       &#x27;2022-08-18T00:00:00.000000000&#x27;, &#x27;2022-08-19T00:00:00.000000000&#x27;,
       &#x27;2022-08-20T00:00:00.000000000&#x27;, &#x27;2022-08-21T00:00:00.000000000&#x27;,
       &#x27;2022-08-22T00:00:00.000000000&#x27;, &#x27;2022-08-23T00:00:00.000000000&#x27;,
       &#x27;2022-08-24T00:00:00.000000000&#x27;, &#x27;2022-08-25T00:00:00.000000000&#x27;,
       &#x27;2022-08-26T00:00:00.000000000&#x27;, &#x27;2022-08-27T00:00:00.000000000&#x27;,
       &#x27;2022-08-28T00:00:00.000000000&#x27;, &#x27;2022-08-29T00:00:00.000000000&#x27;,
       &#x27;2022-08-30T00:00:00.000000000&#x27;, &#x27;2022-08-31T00:00:00.000000000&#x27;,
       &#x27;2022-09-01T00:00:00.000000000&#x27;, &#x27;2022-09-02T00:00:00.000000000&#x27;,
       &#x27;2022-09-03T00:00:00.000000000&#x27;, &#x27;2022-09-04T00:00:00.000000000&#x27;,
       &#x27;2022-09-05T00:00:00.000000000&#x27;, &#x27;2022-09-06T00:00:00.000000000&#x27;,
       &#x27;2022-09-07T00:00:00.000000000&#x27;, &#x27;2022-09-08T00:00:00.000000000&#x27;,
       &#x27;2022-09-09T00:00:00.000000000&#x27;, &#x27;2022-09-10T00:00:00.000000000&#x27;,
       &#x27;2022-09-11T00:00:00.000000000&#x27;, &#x27;2022-09-12T00:00:00.000000000&#x27;,
       &#x27;2022-09-13T00:00:00.000000000&#x27;, &#x27;2022-09-14T00:00:00.000000000&#x27;,
       &#x27;2022-09-15T00:00:00.000000000&#x27;, &#x27;2022-09-16T00:00:00.000000000&#x27;,
       &#x27;2022-09-17T00:00:00.000000000&#x27;, &#x27;2022-09-18T00:00:00.000000000&#x27;,
       &#x27;2022-09-19T00:00:00.000000000&#x27;, &#x27;2022-09-20T00:00:00.000000000&#x27;,
       &#x27;2022-09-21T00:00:00.000000000&#x27;, &#x27;2022-09-22T00:00:00.000000000&#x27;,
       &#x27;2022-09-23T00:00:00.000000000&#x27;, &#x27;2022-09-24T00:00:00.000000000&#x27;,
       &#x27;2022-09-25T00:00:00.000000000&#x27;, &#x27;2022-09-26T00:00:00.000000000&#x27;,
       &#x27;2022-09-27T00:00:00.000000000&#x27;, &#x27;2022-09-28T00:00:00.000000000&#x27;,
       &#x27;2022-09-29T00:00:00.000000000&#x27;, &#x27;2022-09-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-01T00:00:00.000000000&#x27;, &#x27;2022-10-02T00:00:00.000000000&#x27;,
       &#x27;2022-10-03T00:00:00.000000000&#x27;, &#x27;2022-10-04T00:00:00.000000000&#x27;,
       &#x27;2022-10-05T00:00:00.000000000&#x27;, &#x27;2022-10-06T00:00:00.000000000&#x27;,
       &#x27;2022-10-07T00:00:00.000000000&#x27;, &#x27;2022-10-08T00:00:00.000000000&#x27;,
       &#x27;2022-10-09T00:00:00.000000000&#x27;, &#x27;2022-10-10T00:00:00.000000000&#x27;,
       &#x27;2022-10-11T00:00:00.000000000&#x27;, &#x27;2022-10-12T00:00:00.000000000&#x27;,
       &#x27;2022-10-13T00:00:00.000000000&#x27;, &#x27;2022-10-14T00:00:00.000000000&#x27;,
       &#x27;2022-10-15T00:00:00.000000000&#x27;, &#x27;2022-10-16T00:00:00.000000000&#x27;,
       &#x27;2022-10-17T00:00:00.000000000&#x27;, &#x27;2022-10-18T00:00:00.000000000&#x27;,
       &#x27;2022-10-19T00:00:00.000000000&#x27;, &#x27;2022-10-20T00:00:00.000000000&#x27;,
       &#x27;2022-10-21T00:00:00.000000000&#x27;, &#x27;2022-10-22T00:00:00.000000000&#x27;,
       &#x27;2022-10-23T00:00:00.000000000&#x27;, &#x27;2022-10-24T00:00:00.000000000&#x27;,
       &#x27;2022-10-25T00:00:00.000000000&#x27;, &#x27;2022-10-26T00:00:00.000000000&#x27;,
       &#x27;2022-10-27T00:00:00.000000000&#x27;, &#x27;2022-10-28T00:00:00.000000000&#x27;,
       &#x27;2022-10-29T00:00:00.000000000&#x27;, &#x27;2022-10-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-31T00:00:00.000000000&#x27;, &#x27;2022-11-01T00:00:00.000000000&#x27;,
       &#x27;2022-11-02T00:00:00.000000000&#x27;, &#x27;2022-11-03T00:00:00.000000000&#x27;,
       &#x27;2022-11-04T00:00:00.000000000&#x27;, &#x27;2022-11-05T00:00:00.000000000&#x27;,
       &#x27;2022-11-06T00:00:00.000000000&#x27;, &#x27;2022-11-07T00:00:00.000000000&#x27;,
       &#x27;2022-11-08T00:00:00.000000000&#x27;, &#x27;2022-11-09T00:00:00.000000000&#x27;,
       &#x27;2022-11-10T00:00:00.000000000&#x27;, &#x27;2022-11-11T00:00:00.000000000&#x27;,
       &#x27;2022-11-12T00:00:00.000000000&#x27;, &#x27;2022-11-13T00:00:00.000000000&#x27;,
       &#x27;2022-11-14T00:00:00.000000000&#x27;, &#x27;2022-11-15T00:00:00.000000000&#x27;,
       &#x27;2022-11-16T00:00:00.000000000&#x27;, &#x27;2022-11-17T00:00:00.000000000&#x27;,
       &#x27;2022-11-18T00:00:00.000000000&#x27;, &#x27;2022-11-19T00:00:00.000000000&#x27;,
       &#x27;2022-11-20T00:00:00.000000000&#x27;, &#x27;2022-11-21T00:00:00.000000000&#x27;,
       &#x27;2022-11-22T00:00:00.000000000&#x27;, &#x27;2022-11-23T00:00:00.000000000&#x27;,
       &#x27;2022-11-24T00:00:00.000000000&#x27;, &#x27;2022-11-25T00:00:00.000000000&#x27;,
       &#x27;2022-11-26T00:00:00.000000000&#x27;, &#x27;2022-11-27T00:00:00.000000000&#x27;,
       &#x27;2022-11-28T00:00:00.000000000&#x27;, &#x27;2022-11-29T00:00:00.000000000&#x27;,
       &#x27;2022-11-30T00:00:00.000000000&#x27;, &#x27;2022-12-01T00:00:00.000000000&#x27;,
       &#x27;2022-12-02T00:00:00.000000000&#x27;, &#x27;2022-12-03T00:00:00.000000000&#x27;,
       &#x27;2022-12-04T00:00:00.000000000&#x27;, &#x27;2022-12-05T00:00:00.000000000&#x27;,
       &#x27;2022-12-06T00:00:00.000000000&#x27;, &#x27;2022-12-07T00:00:00.000000000&#x27;,
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-0447b317-b694-4110-a02c-658bff25d5b7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-0447b317-b694-4110-a02c-658bff25d5b7' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-a6a936b6-31d8-480a-8b28-156cb84471bb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a6a936b6-31d8-480a-8b28-156cb84471bb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ee240de0-0ddc-4d65-856c-89d218b361dc' class='xr-var-data-in' type='checkbox'><label for='data-ee240de0-0ddc-4d65-856c-89d218b361dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-68e886d0-0504-415c-bf4c-7e7ffb480325' class='xr-section-summary-in' type='checkbox'  ><label for='section-68e886d0-0504-415c-bf4c-7e7ffb480325' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d7c52252-0b6d-492b-a3c7-6e23b84c3a60' class='xr-index-data-in' type='checkbox'/><label for='index-d7c52252-0b6d-492b-a3c7-6e23b84c3a60' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-adca0268-7309-4b21-866b-0b7d1b6c4d40' class='xr-index-data-in' type='checkbox'/><label for='index-adca0268-7309-4b21-866b-0b7d1b6c4d40' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-604f9c07-125e-4ad0-bd92-6ead1ea2eae8' class='xr-index-data-in' type='checkbox'/><label for='index-604f9c07-125e-4ad0-bd92-6ead1ea2eae8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-dc31cb18-ad4c-4c29-aa10-16c326e1627e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-dc31cb18-ad4c-4c29-aa10-16c326e1627e' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.238442+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_observed_data31b7d8d5-5cc7-4e28-a7f6-45967f05f8c9" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_observed_data31b7d8d5-5cc7-4e28-a7f6-45967f05f8c9" class = "xr-section-summary">observed_data</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 2kB
Dimensions:    (obs_dim_0: 123)
Coordinates:
  * obs_dim_0  (obs_dim_0) datetime64[ns] 984B 2022-08-08 ... 2022-12-08
Data variables:
    obs        (obs_dim_0) int64 984B ...
Attributes:
    created_at:                 2024-10-24T16:45:20.239093+00:00
    arviz_version:              0.19.0
    inference_library:          numpyro
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-89f6a3dd-8643-4355-8fae-96bb7f3758a7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-89f6a3dd-8643-4355-8fae-96bb7f3758a7' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-7804514b-a234-4814-a09b-4211d6521956' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7804514b-a234-4814-a09b-4211d6521956' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-7cb625ff-f566-4924-bc94-a6c055390b0b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7cb625ff-f566-4924-bc94-a6c055390b0b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e75dbd3b-6572-4c21-b64e-8a120c8bdc97' class='xr-var-data-in' type='checkbox'><label for='data-e75dbd3b-6572-4c21-b64e-8a120c8bdc97' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
       &#x27;2022-08-10T00:00:00.000000000&#x27;, &#x27;2022-08-11T00:00:00.000000000&#x27;,
       &#x27;2022-08-12T00:00:00.000000000&#x27;, &#x27;2022-08-13T00:00:00.000000000&#x27;,
       &#x27;2022-08-14T00:00:00.000000000&#x27;, &#x27;2022-08-15T00:00:00.000000000&#x27;,
       &#x27;2022-08-16T00:00:00.000000000&#x27;, &#x27;2022-08-17T00:00:00.000000000&#x27;,
       &#x27;2022-08-18T00:00:00.000000000&#x27;, &#x27;2022-08-19T00:00:00.000000000&#x27;,
       &#x27;2022-08-20T00:00:00.000000000&#x27;, &#x27;2022-08-21T00:00:00.000000000&#x27;,
       &#x27;2022-08-22T00:00:00.000000000&#x27;, &#x27;2022-08-23T00:00:00.000000000&#x27;,
       &#x27;2022-08-24T00:00:00.000000000&#x27;, &#x27;2022-08-25T00:00:00.000000000&#x27;,
       &#x27;2022-08-26T00:00:00.000000000&#x27;, &#x27;2022-08-27T00:00:00.000000000&#x27;,
       &#x27;2022-08-28T00:00:00.000000000&#x27;, &#x27;2022-08-29T00:00:00.000000000&#x27;,
       &#x27;2022-08-30T00:00:00.000000000&#x27;, &#x27;2022-08-31T00:00:00.000000000&#x27;,
       &#x27;2022-09-01T00:00:00.000000000&#x27;, &#x27;2022-09-02T00:00:00.000000000&#x27;,
       &#x27;2022-09-03T00:00:00.000000000&#x27;, &#x27;2022-09-04T00:00:00.000000000&#x27;,
       &#x27;2022-09-05T00:00:00.000000000&#x27;, &#x27;2022-09-06T00:00:00.000000000&#x27;,
       &#x27;2022-09-07T00:00:00.000000000&#x27;, &#x27;2022-09-08T00:00:00.000000000&#x27;,
       &#x27;2022-09-09T00:00:00.000000000&#x27;, &#x27;2022-09-10T00:00:00.000000000&#x27;,
       &#x27;2022-09-11T00:00:00.000000000&#x27;, &#x27;2022-09-12T00:00:00.000000000&#x27;,
       &#x27;2022-09-13T00:00:00.000000000&#x27;, &#x27;2022-09-14T00:00:00.000000000&#x27;,
       &#x27;2022-09-15T00:00:00.000000000&#x27;, &#x27;2022-09-16T00:00:00.000000000&#x27;,
       &#x27;2022-09-17T00:00:00.000000000&#x27;, &#x27;2022-09-18T00:00:00.000000000&#x27;,
       &#x27;2022-09-19T00:00:00.000000000&#x27;, &#x27;2022-09-20T00:00:00.000000000&#x27;,
       &#x27;2022-09-21T00:00:00.000000000&#x27;, &#x27;2022-09-22T00:00:00.000000000&#x27;,
       &#x27;2022-09-23T00:00:00.000000000&#x27;, &#x27;2022-09-24T00:00:00.000000000&#x27;,
       &#x27;2022-09-25T00:00:00.000000000&#x27;, &#x27;2022-09-26T00:00:00.000000000&#x27;,
       &#x27;2022-09-27T00:00:00.000000000&#x27;, &#x27;2022-09-28T00:00:00.000000000&#x27;,
       &#x27;2022-09-29T00:00:00.000000000&#x27;, &#x27;2022-09-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-01T00:00:00.000000000&#x27;, &#x27;2022-10-02T00:00:00.000000000&#x27;,
       &#x27;2022-10-03T00:00:00.000000000&#x27;, &#x27;2022-10-04T00:00:00.000000000&#x27;,
       &#x27;2022-10-05T00:00:00.000000000&#x27;, &#x27;2022-10-06T00:00:00.000000000&#x27;,
       &#x27;2022-10-07T00:00:00.000000000&#x27;, &#x27;2022-10-08T00:00:00.000000000&#x27;,
       &#x27;2022-10-09T00:00:00.000000000&#x27;, &#x27;2022-10-10T00:00:00.000000000&#x27;,
       &#x27;2022-10-11T00:00:00.000000000&#x27;, &#x27;2022-10-12T00:00:00.000000000&#x27;,
       &#x27;2022-10-13T00:00:00.000000000&#x27;, &#x27;2022-10-14T00:00:00.000000000&#x27;,
       &#x27;2022-10-15T00:00:00.000000000&#x27;, &#x27;2022-10-16T00:00:00.000000000&#x27;,
       &#x27;2022-10-17T00:00:00.000000000&#x27;, &#x27;2022-10-18T00:00:00.000000000&#x27;,
       &#x27;2022-10-19T00:00:00.000000000&#x27;, &#x27;2022-10-20T00:00:00.000000000&#x27;,
       &#x27;2022-10-21T00:00:00.000000000&#x27;, &#x27;2022-10-22T00:00:00.000000000&#x27;,
       &#x27;2022-10-23T00:00:00.000000000&#x27;, &#x27;2022-10-24T00:00:00.000000000&#x27;,
       &#x27;2022-10-25T00:00:00.000000000&#x27;, &#x27;2022-10-26T00:00:00.000000000&#x27;,
       &#x27;2022-10-27T00:00:00.000000000&#x27;, &#x27;2022-10-28T00:00:00.000000000&#x27;,
       &#x27;2022-10-29T00:00:00.000000000&#x27;, &#x27;2022-10-30T00:00:00.000000000&#x27;,
       &#x27;2022-10-31T00:00:00.000000000&#x27;, &#x27;2022-11-01T00:00:00.000000000&#x27;,
       &#x27;2022-11-02T00:00:00.000000000&#x27;, &#x27;2022-11-03T00:00:00.000000000&#x27;,
       &#x27;2022-11-04T00:00:00.000000000&#x27;, &#x27;2022-11-05T00:00:00.000000000&#x27;,
       &#x27;2022-11-06T00:00:00.000000000&#x27;, &#x27;2022-11-07T00:00:00.000000000&#x27;,
       &#x27;2022-11-08T00:00:00.000000000&#x27;, &#x27;2022-11-09T00:00:00.000000000&#x27;,
       &#x27;2022-11-10T00:00:00.000000000&#x27;, &#x27;2022-11-11T00:00:00.000000000&#x27;,
       &#x27;2022-11-12T00:00:00.000000000&#x27;, &#x27;2022-11-13T00:00:00.000000000&#x27;,
       &#x27;2022-11-14T00:00:00.000000000&#x27;, &#x27;2022-11-15T00:00:00.000000000&#x27;,
       &#x27;2022-11-16T00:00:00.000000000&#x27;, &#x27;2022-11-17T00:00:00.000000000&#x27;,
       &#x27;2022-11-18T00:00:00.000000000&#x27;, &#x27;2022-11-19T00:00:00.000000000&#x27;,
       &#x27;2022-11-20T00:00:00.000000000&#x27;, &#x27;2022-11-21T00:00:00.000000000&#x27;,
       &#x27;2022-11-22T00:00:00.000000000&#x27;, &#x27;2022-11-23T00:00:00.000000000&#x27;,
       &#x27;2022-11-24T00:00:00.000000000&#x27;, &#x27;2022-11-25T00:00:00.000000000&#x27;,
       &#x27;2022-11-26T00:00:00.000000000&#x27;, &#x27;2022-11-27T00:00:00.000000000&#x27;,
       &#x27;2022-11-28T00:00:00.000000000&#x27;, &#x27;2022-11-29T00:00:00.000000000&#x27;,
       &#x27;2022-11-30T00:00:00.000000000&#x27;, &#x27;2022-12-01T00:00:00.000000000&#x27;,
       &#x27;2022-12-02T00:00:00.000000000&#x27;, &#x27;2022-12-03T00:00:00.000000000&#x27;,
       &#x27;2022-12-04T00:00:00.000000000&#x27;, &#x27;2022-12-05T00:00:00.000000000&#x27;,
       &#x27;2022-12-06T00:00:00.000000000&#x27;, &#x27;2022-12-07T00:00:00.000000000&#x27;,
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-87dea63a-8012-4c35-99a6-fc13ee25e48a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-87dea63a-8012-4c35-99a6-fc13ee25e48a' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-2d23f99a-e0ae-4602-a149-1c6eed768851' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2d23f99a-e0ae-4602-a149-1c6eed768851' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bc7f1eab-23aa-471e-9a2a-bed90a24a8c5' class='xr-var-data-in' type='checkbox'><label for='data-bc7f1eab-23aa-471e-9a2a-bed90a24a8c5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123 values with dtype=int64]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-47cdc56e-c34f-44ad-9b21-452c1aa6a8c5' class='xr-section-summary-in' type='checkbox'  ><label for='section-47cdc56e-c34f-44ad-9b21-452c1aa6a8c5' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0801479d-7468-4212-9e77-4bf89330d06b' class='xr-index-data-in' type='checkbox'/><label for='index-0801479d-7468-4212-9e77-4bf89330d06b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-37932c0b-f0b1-4506-8bc5-efaf3d512da9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-37932c0b-f0b1-4506-8bc5-efaf3d512da9' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.239093+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */
&#10;:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
&#10;html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}
&#10;.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}
&#10;.xr-sections.group-sections {
  grid-template-columns: auto;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: none;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: '▼';
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: '(';
}
&#10;.xr-dim-list:after {
  content: ')';
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>

``` python
# show dates
print(idata_w_dates["observed_data"]["obs"]["obs_dim_0"][:15])
```

    <xarray.DataArray 'obs_dim_0' (obs_dim_0: 15)> Size: 120B
    array(['2022-08-08T00:00:00.000000000', '2022-08-09T00:00:00.000000000',
           '2022-08-10T00:00:00.000000000', '2022-08-11T00:00:00.000000000',
           '2022-08-12T00:00:00.000000000', '2022-08-13T00:00:00.000000000',
           '2022-08-14T00:00:00.000000000', '2022-08-15T00:00:00.000000000',
           '2022-08-16T00:00:00.000000000', '2022-08-17T00:00:00.000000000',
           '2022-08-18T00:00:00.000000000', '2022-08-19T00:00:00.000000000',
           '2022-08-20T00:00:00.000000000', '2022-08-21T00:00:00.000000000',
           '2022-08-22T00:00:00.000000000'], dtype='datetime64[ns]')
    Coordinates:
      * obs_dim_0  (obs_dim_0) datetime64[ns] 120B 2022-08-08 ... 2022-08-22

``` python
# idata without dates as coordinates
idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates
print(idata_wo_dates["observed_data"]["obs"]["obs_dim_0"][:15])
```

    <xarray.DataArray 'obs_dim_0' (obs_dim_0: 15)> Size: 120B
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
    Coordinates:
      * obs_dim_0  (obs_dim_0) int64 120B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

The forecast was generated following the creation of `nhsn_hosp_flu.csv`
(see previous section) by running `data.py` with the following added:

``` python
make_forecast(
    nhsn_data=forecasttools.nhsn_hosp_flu,
    start_date="2022-08-08",
    end_date="2022-12-08",
    juris_subset=["TX"],
    forecast_days=28,
    save_path="../forecasttools/example_flu_forecast_w_dates.nc",
    save_idata=True,
    use_log=False,
)
```

(note: `make_forecast` is no longer included in `forecasttools-py`,
given the expectation that no one would ever call it; however, for
reproducibility’s sake, the following is included here)

<details>

<summary>

Some Of The Forecast Code
</summary>

``` python
"""
Creating a new idata object with
dates to change the functionality
of idata_w_dates_to_df.
"""

# %% IMPORTS

import os
from datetime import datetime, timedelta

import arviz as az
import forecasttools
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import patsy
import polars as pl
from numpy.typing import NDArray

# %% CHECK FILE PATH


def check_file_save_path(
    file_save_path: str,
) -> None:
    """
    Checks whether a file path is valid.

    file_save_path
        The file path to be checked.
    """
    directory = os.path.dirname(file_save_path)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory is not writable: {directory}")
    if os.path.exists(file_save_path):
        raise FileExistsError(f"File already exists at: {file_save_path}")


# %% SPLINE REGRESSION MODEL


def model(basis_matrix, y=None):
    # priors
    shift = numpyro.sample("shift", dist.Normal(0.0, 2.0))
    beta_coeffs = numpyro.sample(
        "beta_coeffs",
        dist.Normal(jnp.zeros(basis_matrix.shape[1]), 2.0),
    )
    shift_mu = jnp.dot(basis_matrix, beta_coeffs) + shift
    mu_exp = jnp.exp(shift_mu)
    alpha = numpyro.sample("alpha", dist.Exponential(1.0))
    # likelihood
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mu_exp, alpha),
        obs=y,
    )


# %% SPLINE BASIS MATRIX


def spline_basis(X, degree: int = 4, df: int = 8) -> NDArray:
    basis = patsy.dmatrix(
        "bs(x, df=df, degree=degree, include_intercept=True) - 1",
        {"x": X, "df": df, "degree": degree},
        return_type="matrix",
    )
    return np.array(basis)


# %% PLOT AND OR SAVE FORECAST


def plot_and_or_save_forecast(
    idata: az.InferenceData,
    X: NDArray,
    y: NDArray,
    title: str,
    start_date: str,
    end_date: str,
    last_fit: int,
    X_act: NDArray,
    y_act: NDArray,
    save_to_pdf: bool = False,
    use_log: bool = False,
):
    """
    Includes hard-coded variables. For the
    author's testing and no more.
    """
    x_data = idata.posterior_predictive["obs_dim_0"]
    y_data = idata.posterior_predictive["obs"]
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.95,
        color="skyblue",
        smooth=False,
        fill_kwargs={
            "alpha": 0.2,
            "label": "95% Credible",
        },
        ax=axes,
    )
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.75,
        color="skyblue",
        smooth=False,
        fill_kwargs={
            "alpha": 0.4,
            "label": "75% Credible",
        },
        ax=axes,
    )
    az.plot_hdi(
        x_data,
        y_data,
        hdi_prob=0.5,
        color="C0",
        smooth=False,
        fill_kwargs={
            "alpha": 0.6,
            "label": "50% Credible",
        },
        ax=axes,
    )
    axes.plot(
        X,
        y,
        marker="o",
        color="black",
        linewidth=1.0,
        markersize=3.0,
        label="Observed",
    )
    if (X_act is not None) and (y_act is not None):
        axes.plot(
            X_act,
            y_act,
            marker="o",
            color="red",
            linewidth=1.0,
            markersize=3.0,
            label="Actual",
        )
    if use_log:
        axes.set_yscale("log")
        axes.set_ylabel(
            "(Log) Hospital Admissions",
            fontsize=17.5,
        )
    if not use_log:
        axes.set_ylabel("Hospital Admissions", fontsize=17.5)
    median_ts = y_data.median(dim=["chain", "draw"])
    axes.plot(
        x_data,
        median_ts,
        color="blue",
        label="Median",
    )
    axes.legend()
    axes.axvline(last_fit, color="black", linestyle="--")
    axes.set_title(
        f"{title}",
        fontsize=20,
    )
    axes.set_xlabel("Time", fontsize=17.5)

    plt.show()


# %% ADD DATES TO AN INFERENCE DATA OBJECT


def add_dates_to_idata_object(
    idata: az.InferenceData,
    start_date: str,
) -> az.InferenceData:
    """
    Takes an InferenceData object w/
    observed_data and posterior_predictive
    groups and adds date indexing
    """
    pass


# %% MAKE A FORECAST


def make_forecast(
    nhsn_data: str,
    start_date: str,
    end_date: str,
    juris_subset: list[str],
    forecast_days: int,
    save_path: str = os.path.join(os.getcwd(), "forecast.nc"),
    show_plot: bool = True,
    save_idata: bool = False,
    use_log: bool = False,
) -> None:
    """
    Generates a forecast for specified
    dates using a spline regression model.
    """
    # check dataset path
    check_file_save_path(save_path)
    # clean data and organize data, cleaning null values
    nhsn_data = nhsn_data.with_columns(
        pl.col("hosp").cast(pl.Int64),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    ).filter(
        pl.col("hosp").is_not_null(),
        pl.col("state").is_in(juris_subset),
    )
    nhsn_data_ready = nhsn_data.filter(
        (
            pl.col("date")
            >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
        & (
            pl.col("date")
            <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
    )
    # get the actual values, if they exist
    try:
        forecast_end_date = datetime.strptime(
            end_date, "%Y-%m-%d"
        ) + timedelta(days=forecast_days)
        nhsn_data_actual = nhsn_data.filter(
            (
                pl.col("date")
                >= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
            & (pl.col("date") <= pl.lit(forecast_end_date))
        )
    except Exception as e:
        nhsn_data_actual = None
        print(f"The following error occurred: {e}")
    # define some shared inference values
    random_seed = 2134312
    num_samples = 1000
    num_warmup = 500
    # get posterior samples and make forecasts for each selected state
    for state in juris_subset:
        # get the state data
        state_nhsn = nhsn_data_ready.filter(pl.col("state") == state)
        # get observation (fitting) data y, X
        y = state_nhsn["hosp"].to_numpy()
        X = np.arange(y.shape[0])
        # set up inference, NUTS/MCMC
        kernel = numpyro.infer.NUTS(
            model=model,
            max_tree_depth=12,
            target_accept_prob=0.85,
            init_strategy=numpyro.infer.init_to_uniform(),
        )
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        # create spline basis for obs period and forecast period
        last = X[-1]
        X_future = np.hstack(
            (
                X,
                np.arange(
                    last + 1,
                    last + 1 + forecast_days,
                ),
            )
        )
        sbm = spline_basis(X_future)
        # get posterior samples
        mcmc.run(
            rng_key=jr.key(random_seed),
            basis_matrix=sbm[: len(X)],
            y=y,
        )
        posterior_samples = mcmc.get_samples()
        # get prior predictive
        prior_pred = numpyro.infer.Predictive(model, num_samples=num_samples)(
            rng_key=jr.key(random_seed),
            basis_matrix=sbm[: len(X)],
        )
        # get posterior predictive forecast
        posterior_pred_for = numpyro.infer.Predictive(
            model,
            posterior_samples=posterior_samples,
        )(
            rng_key=jr.key(random_seed),
            basis_matrix=sbm,
        )
        # create initial inference data object(s) and store
        idata = az.from_numpyro(
            posterior=mcmc,
            posterior_predictive=posterior_pred_for,
            prior=prior_pred,
        )
        # get actual data, if it exists
        if isinstance(nhsn_data_actual, pl.DataFrame):
            actual_data = nhsn_data_actual.filter(pl.col("state") == state)
            y_act = actual_data["hosp"].to_numpy()
            X_act = np.arange(last - 1, last + forecast_days)
        if not isinstance(nhsn_data_actual, pl.DataFrame):
            y_act = None
            X_act = None
        # add dates to idata object

        # save idata object(s)
        if save_idata:
            idata.to_netcdf(save_path)
        # plot forecast (if desired) from idata light
        if show_plot:
            plot_and_or_save_forecast(
                idata=idata,
                X=X,
                y=y,
                title=f"Hospital Admissions ({state}, {start_date}-{end_date})",
                start_date=start_date,
                end_date=end_date,
                last_fit=last,
                X_act=X_act,
                y_act=y_act,
                use_log=use_log,
            )


# %% EXECUTE MODE

make_forecast(
    nhsn_data=forecasttools.nhsn_hosp_flu,
    start_date="2022-08-08",
    end_date="2022-12-08",
    juris_subset=["TX"],
    forecast_days=28,
    save_path="../forecasttools/example_flu_forecast_w_dates.nc",
    save_idata=False,
    use_log=True,
)
```

</details>

The forecast looks like:

<img src="./assets/example_forecast_w_dates.png" style="width:75.0%"
alt="Example NHSN-based Influenza forecast" />

# CDC Open Source Considerations

**General disclaimer** This repository was created for use by CDC
programs to collaborate on public health related projects in support of
the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).
GitHub is not hosted by the CDC, but is a third party website used by
CDC and its partners to share information and collaborate on software.
CDC use of GitHub does not imply an endorsement of any one particular
service, product, or enterprise.

<details>

<summary>

Rules, Policy, And Collaboration
</summary>

- [Open Practices](./rules-and-policies/open_practices.md)
- [Rules of Behavior](./rules-and-policies/rules_of_behavior.md)
- [Thanks and Acknowledgements](./rules-and-policies/thanks.md)
- [Disclaimer](DISCLAIMER.md)
- [Contribution Notice](CONTRIBUTING.md)
- [Code of Conduct](./rules-and-policies/code-of-conduct.md)

</details>

<details>

<summary>

Public Domain Standard Notice
</summary>

This repository constitutes a work of the United States Government and
is not subject to domestic copyright protection under 17 USC § 105. This
repository is in the public domain within the United States, and
copyright and related rights in the work worldwide are waived through
the [CC0 1.0 Universal public domain
dedication](https://creativecommons.org/publicdomain/zero/1.0/). All
contributions to this repository will be released under the CC0
dedication. By submitting a pull request you are agreeing to comply with
this waiver of copyright interest.
</details>

<details>

<summary>

License Standard Notice
</summary>

The repository utilizes code licensed under the terms of the Apache
Software License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it
and/or modify it under the terms of the Apache Software License version
2, or (at your option) any later version.

This source code in this repository is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache Software License for more details.

You should have received a copy of the Apache Software License along
with this program. If not, see
http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its
license.
</details>

<details>

<summary>

Privacy Standard Notice
</summary>

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md) and [Code of Conduct](code-of-conduct.md).
For more information about CDC’s privacy policy, please visit
[http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).
</details>

<details>

<summary>

Contributing Standard Notice
</summary>

Anyone is encouraged to contribute to the repository by
[forking](https://help.github.com/articles/fork-a-repo) and submitting a
pull request. (If you are new to GitHub, you might start with a [basic
tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual,
irrevocable, non-exclusive, transferable license to all users under the
terms of the [Apache Software License
v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

All comments, messages, pull requests, and other submissions received
through CDC including this GitHub page may be subject to applicable
federal law, including but not limited to the Federal Records Act, and
may be archived. Learn more at <http://www.cdc.gov/other/privacy.html>.
</details>

<details>

<summary>

Records Management Standard Notice
</summary>

This repository is not a source of government records, but is a copy to
increase collaboration and collaborative potential. All government
records will be published through the [CDC web
site](http://www.cdc.gov).
</details>

<details>

<summary>

Additional Standard Notices
</summary>

Please refer to [CDC’s Template
Repository](https://github.com/CDCgov/template) for more information
about [contributing to this
repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md),
[public domain notices and
disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md),
and [code of
conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
</details>
