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
loc_table
```

<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>

| location_code | short_name | long_name                     |
|---------------|------------|-------------------------------|
| str           | str        | str                           |
| "US"          | "US"       | "United States"               |
| "01"          | "AL"       | "Alabama"                     |
| "02"          | "AK"       | "Alaska"                      |
| "04"          | "AZ"       | "Arizona"                     |
| "05"          | "AR"       | "Arkansas"                    |
| …             | …          | …                             |
| "66"          | "GU"       | "Guam"                        |
| "69"          | "MP"       | "Northern Mariana Islands"    |
| "72"          | "PR"       | "Puerto Rico"                 |
| "74"          | "UM"       | "U.S. Minor Outlying Islands" |
| "78"          | "VI"       | "U.S. Virgin Islands"         |

</div>

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
submission
```

<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>

| reference_date | target | horizon | target_end_date | location | output_type | output_type_id | value |
|----|----|----|----|----|----|----|----|
| str | str | i64 | str | str | str | f64 | f64 |
| "2023-10-14" | "wk inc flu hosp" | -1 | "2023-10-07" | "01" | "quantile" | 0.01 | 7.670286 |
| "2023-10-14" | "wk inc flu hosp" | -1 | "2023-10-07" | "01" | "quantile" | 0.025 | 9.968043 |
| "2023-10-14" | "wk inc flu hosp" | -1 | "2023-10-07" | "01" | "quantile" | 0.05 | 12.022354 |
| "2023-10-14" | "wk inc flu hosp" | -1 | "2023-10-07" | "01" | "quantile" | 0.1 | 14.497646 |
| "2023-10-14" | "wk inc flu hosp" | -1 | "2023-10-07" | "01" | "quantile" | 0.15 | 16.119813 |
| … | … | … | … | … | … | … | … |
| "2023-10-14" | "wk inc flu hosp" | 2 | "2023-10-28" | "US" | "quantile" | 0.85 | 2451.874899 |
| "2023-10-14" | "wk inc flu hosp" | 2 | "2023-10-28" | "US" | "quantile" | 0.9 | 2806.928588 |
| "2023-10-14" | "wk inc flu hosp" | 2 | "2023-10-28" | "US" | "quantile" | 0.95 | 3383.74799 |
| "2023-10-14" | "wk inc flu hosp" | 2 | "2023-10-28" | "US" | "quantile" | 0.975 | 3940.392536 |
| "2023-10-14" | "wk inc flu hosp" | 2 | "2023-10-28" | "US" | "quantile" | 0.99 | 4761.757385 |

</div>

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
                  <input id="idata_posterior1d3f4de4-d616-47cc-a15b-741f94fbef64" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior1d3f4de4-d616-47cc-a15b-741f94fbef64" class = "xr-section-summary">posterior</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-49b56610-9573-488b-87f7-f3fe24893117' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-49b56610-9573-488b-87f7-f3fe24893117' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-83322a20-65fb-4934-9522-87dc88abe9c7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-83322a20-65fb-4934-9522-87dc88abe9c7' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-2de728b0-0b3b-4827-8dfa-56e204b2975d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2de728b0-0b3b-4827-8dfa-56e204b2975d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-37f09484-4538-40b0-a9db-9c4c00320bc9' class='xr-var-data-in' type='checkbox'><label for='data-37f09484-4538-40b0-a9db-9c4c00320bc9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-70b102c3-dff8-4cbd-9eff-83023c3bce24' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-70b102c3-dff8-4cbd-9eff-83023c3bce24' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-84b2d829-2add-4879-8b56-236f66903f32' class='xr-var-data-in' type='checkbox'><label for='data-84b2d829-2add-4879-8b56-236f66903f32' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-dc9665b9-8174-49b1-8f74-f8bf271222cc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dc9665b9-8174-49b1-8f74-f8bf271222cc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c18f41a8-8e12-4160-a161-4b588688147b' class='xr-var-data-in' type='checkbox'><label for='data-c18f41a8-8e12-4160-a161-4b588688147b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-41f083b0-8b31-4aa3-ab00-9337fda0794c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-41f083b0-8b31-4aa3-ab00-9337fda0794c' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-a14a864e-a510-4e37-a0d5-feebfc7bead4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a14a864e-a510-4e37-a0d5-feebfc7bead4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-90d799c9-9534-4511-8466-04758b868aab' class='xr-var-data-in' type='checkbox'><label for='data-90d799c9-9534-4511-8466-04758b868aab' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-198b219c-ee68-4de9-b9e9-7ffeac24a85a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-198b219c-ee68-4de9-b9e9-7ffeac24a85a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f92c4e9f-771c-4da4-b6cc-2076d4578ba4' class='xr-var-data-in' type='checkbox'><label for='data-f92c4e9f-771c-4da4-b6cc-2076d4578ba4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-90228d18-daf7-4fee-acfa-3fb4a6de3271' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-90228d18-daf7-4fee-acfa-3fb4a6de3271' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d7cc6517-629b-45d5-a361-6ef9cc8928e2' class='xr-var-data-in' type='checkbox'><label for='data-d7cc6517-629b-45d5-a361-6ef9cc8928e2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-396df943-b715-4118-8e24-6550119d8402' class='xr-section-summary-in' type='checkbox'  ><label for='section-396df943-b715-4118-8e24-6550119d8402' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c662459c-ed4d-45ea-9416-f6591ec47417' class='xr-index-data-in' type='checkbox'/><label for='index-c662459c-ed4d-45ea-9416-f6591ec47417' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-932fd7d2-ff90-4678-984a-6575f3ebca86' class='xr-index-data-in' type='checkbox'/><label for='index-932fd7d2-ff90-4678-984a-6575f3ebca86' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-a7b2de8d-14fb-4a5b-8a0a-b90e8430d155' class='xr-index-data-in' type='checkbox'/><label for='index-a7b2de8d-14fb-4a5b-8a0a-b90e8430d155' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c3a155a5-3f3f-4820-8a3f-ed6b91ea160f' class='xr-section-summary-in' type='checkbox'  ><label for='section-c3a155a5-3f3f-4820-8a3f-ed6b91ea160f' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.119636+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_posterior_predictive51277758-d852-4be2-9443-1aca19c6df2f" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior_predictive51277758-d852-4be2-9443-1aca19c6df2f" class = "xr-section-summary">posterior_predictive</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-740c41df-6652-4fa5-b1be-d21fee367381' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-740c41df-6652-4fa5-b1be-d21fee367381' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 151</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-884b28d1-8c2d-4418-8573-df697713a2b3' class='xr-section-summary-in' type='checkbox'  checked><label for='section-884b28d1-8c2d-4418-8573-df697713a2b3' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-66a57d7d-302c-45b5-98c1-2ea8e789d51d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-66a57d7d-302c-45b5-98c1-2ea8e789d51d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e0b94142-3dab-4233-a342-b6f8132672ae' class='xr-var-data-in' type='checkbox'><label for='data-e0b94142-3dab-4233-a342-b6f8132672ae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-9a81d52b-6204-4584-b15b-4f496f1e67a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9a81d52b-6204-4584-b15b-4f496f1e67a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-93ab6d6c-c336-4c8e-bf01-2d3c8c83d411' class='xr-var-data-in' type='checkbox'><label for='data-93ab6d6c-c336-4c8e-bf01-2d3c8c83d411' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2023-01-05</div><input id='attrs-de9eebb6-dee6-446a-86b0-fbdb58b15a53' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de9eebb6-dee6-446a-86b0-fbdb58b15a53' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1f42adee-ee0f-48b5-a31a-cf79bb87d3c5' class='xr-var-data-in' type='checkbox'><label for='data-1f42adee-ee0f-48b5-a31a-cf79bb87d3c5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2023-01-05T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-27b2dea5-dde7-4308-a798-9f01b08d73d1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-27b2dea5-dde7-4308-a798-9f01b08d73d1' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-800f36c6-2396-4651-958b-afcf877dca51' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-800f36c6-2396-4651-958b-afcf877dca51' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e055036a-79c6-4b55-bfea-8e2d3417d50e' class='xr-var-data-in' type='checkbox'><label for='data-e055036a-79c6-4b55-bfea-8e2d3417d50e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[151000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ab4ec24d-98ca-4a1e-89c5-cf4204abe3a8' class='xr-section-summary-in' type='checkbox'  ><label for='section-ab4ec24d-98ca-4a1e-89c5-cf4204abe3a8' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c2718b58-af72-41c4-9b0c-a5b15794f29c' class='xr-index-data-in' type='checkbox'/><label for='index-c2718b58-af72-41c4-9b0c-a5b15794f29c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-22374769-6837-4ea5-8b11-7040d4b128ca' class='xr-index-data-in' type='checkbox'/><label for='index-22374769-6837-4ea5-8b11-7040d4b128ca' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-37e0bd53-6ef5-4335-9d1c-5a95a46522e8' class='xr-index-data-in' type='checkbox'/><label for='index-37e0bd53-6ef5-4335-9d1c-5a95a46522e8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-12-27&#x27;, &#x27;2022-12-28&#x27;, &#x27;2022-12-29&#x27;, &#x27;2022-12-30&#x27;,
               &#x27;2022-12-31&#x27;, &#x27;2023-01-01&#x27;, &#x27;2023-01-02&#x27;, &#x27;2023-01-03&#x27;,
               &#x27;2023-01-04&#x27;, &#x27;2023-01-05&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=151, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-efb9912a-ddda-4e69-a7b5-eea30cc451b8' class='xr-section-summary-in' type='checkbox'  ><label for='section-efb9912a-ddda-4e69-a7b5-eea30cc451b8' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.236298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_log_likelihood6323c28d-2cad-4eb7-b7b0-15a71df556ff" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_log_likelihood6323c28d-2cad-4eb7-b7b0-15a71df556ff" class = "xr-section-summary">log_likelihood</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-18f61e03-cd9e-42bd-8c11-4a6866c88c77' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-18f61e03-cd9e-42bd-8c11-4a6866c88c77' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-972d90ef-e6ce-4d79-8f28-632fe62a58ef' class='xr-section-summary-in' type='checkbox'  checked><label for='section-972d90ef-e6ce-4d79-8f28-632fe62a58ef' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-5e885690-1a81-4a31-8512-e8bb6385da34' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5e885690-1a81-4a31-8512-e8bb6385da34' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4ad2fe8f-a294-4f7d-a254-36160e28a324' class='xr-var-data-in' type='checkbox'><label for='data-4ad2fe8f-a294-4f7d-a254-36160e28a324' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-d945c519-b23c-44ee-8cae-8a46039e9cf7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d945c519-b23c-44ee-8cae-8a46039e9cf7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-10ff1d2e-da37-4f34-a306-c5cf9d453a32' class='xr-var-data-in' type='checkbox'><label for='data-10ff1d2e-da37-4f34-a306-c5cf9d453a32' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 118 119 120 121 122</div><input id='attrs-a0296e38-c51a-4fe4-9cad-0eb9f17844f3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a0296e38-c51a-4fe4-9cad-0eb9f17844f3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-55e4e82b-1ad6-4336-be08-7e400790ea00' class='xr-var-data-in' type='checkbox'><label for='data-55e4e82b-1ad6-4336-be08-7e400790ea00' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e936bdad-1273-4ee8-8a02-5eb990a77c96' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e936bdad-1273-4ee8-8a02-5eb990a77c96' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-d898fc40-470a-49e9-b664-5de4e51fbf07' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d898fc40-470a-49e9-b664-5de4e51fbf07' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a7703e68-6488-4933-a292-4be4b46e4569' class='xr-var-data-in' type='checkbox'><label for='data-a7703e68-6488-4933-a292-4be4b46e4569' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bedd6654-4d9d-463a-934a-fb0ab8ddf33d' class='xr-section-summary-in' type='checkbox'  ><label for='section-bedd6654-4d9d-463a-934a-fb0ab8ddf33d' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-7e09642b-4a58-4dcb-a7bd-123f1aa657e6' class='xr-index-data-in' type='checkbox'/><label for='index-7e09642b-4a58-4dcb-a7bd-123f1aa657e6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-592e8341-4c2c-40f2-b7d4-897675cb3025' class='xr-index-data-in' type='checkbox'/><label for='index-592e8341-4c2c-40f2-b7d4-897675cb3025' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e43aeaad-b5bd-4672-9a01-97253b2d6a72' class='xr-index-data-in' type='checkbox'/><label for='index-e43aeaad-b5bd-4672-9a01-97253b2d6a72' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
      dtype=&#x27;int64&#x27;, name=&#x27;obs_dim_0&#x27;, length=123))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-795d10b7-5154-478e-a59c-110994ad987e' class='xr-section-summary-in' type='checkbox'  ><label for='section-795d10b7-5154-478e-a59c-110994ad987e' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.235298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_sample_stats306e422a-6b85-4c0a-97d6-52325b9dc517" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_stats306e422a-6b85-4c0a-97d6-52325b9dc517" class = "xr-section-summary">sample_stats</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-d2ecc932-56a1-4988-b310-a1ac1a54e31e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-d2ecc932-56a1-4988-b310-a1ac1a54e31e' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-d0fe9093-bddc-4a93-aaf4-30a95b73a52b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d0fe9093-bddc-4a93-aaf4-30a95b73a52b' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-0ae6d99e-769c-4360-b90c-8a7a9899de68' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0ae6d99e-769c-4360-b90c-8a7a9899de68' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d1f7e6a1-2d36-4e1f-abb3-4d13bc9885fb' class='xr-var-data-in' type='checkbox'><label for='data-d1f7e6a1-2d36-4e1f-abb3-4d13bc9885fb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-246db676-8913-444a-b882-6ab9d94e93bb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-246db676-8913-444a-b882-6ab9d94e93bb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7ad5f3cc-56c9-40fc-af42-914dbb45be8b' class='xr-var-data-in' type='checkbox'><label for='data-7ad5f3cc-56c9-40fc-af42-914dbb45be8b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e9912c20-7cfc-467a-b41b-22c7c6e9458f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e9912c20-7cfc-467a-b41b-22c7c6e9458f' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-ad7bd943-a1c2-4be9-baaf-521329828fdf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ad7bd943-a1c2-4be9-baaf-521329828fdf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2062452c-8ec1-4168-b463-b9c2367fb759' class='xr-var-data-in' type='checkbox'><label for='data-2062452c-8ec1-4168-b463-b9c2367fb759' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=bool]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f9fdd51e-e3f1-47af-bb94-e07e783cad88' class='xr-section-summary-in' type='checkbox'  ><label for='section-f9fdd51e-e3f1-47af-bb94-e07e783cad88' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3c9e4a3b-4306-474f-b343-07833459c1b1' class='xr-index-data-in' type='checkbox'/><label for='index-3c9e4a3b-4306-474f-b343-07833459c1b1' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3c4b480a-39d7-4ce1-a251-f54eec6f61e0' class='xr-index-data-in' type='checkbox'/><label for='index-3c4b480a-39d7-4ce1-a251-f54eec6f61e0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e589129f-272a-4cad-9a68-365022ab9e9f' class='xr-section-summary-in' type='checkbox'  ><label for='section-e589129f-272a-4cad-9a68-365022ab9e9f' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.122620+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_prioraf617cf9-d70d-4ba8-8d06-9a4b4fd24461" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_prioraf617cf9-d70d-4ba8-8d06-9a4b4fd24461" class = "xr-section-summary">prior</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-9e1344b8-72f9-4d6e-b758-f329f3920536' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-9e1344b8-72f9-4d6e-b758-f329f3920536' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-b8473569-6abf-4b04-9ddd-30940b08f97e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-b8473569-6abf-4b04-9ddd-30940b08f97e' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-2311d6d1-1045-48b1-83d8-1acbb8c58a27' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2311d6d1-1045-48b1-83d8-1acbb8c58a27' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b56b79b4-0b99-40e4-a1fd-53f8a384cf10' class='xr-var-data-in' type='checkbox'><label for='data-b56b79b4-0b99-40e4-a1fd-53f8a384cf10' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-824ec788-a87c-46b3-82aa-6d86fdab3f09' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-824ec788-a87c-46b3-82aa-6d86fdab3f09' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ee88b9b6-948e-4408-9032-ba170b84a0a9' class='xr-var-data-in' type='checkbox'><label for='data-ee88b9b6-948e-4408-9032-ba170b84a0a9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-11c78c2a-5fa7-41de-9c88-3447d3e5afb3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-11c78c2a-5fa7-41de-9c88-3447d3e5afb3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bbe1b1e4-940e-4a59-9469-e26ea2be54ca' class='xr-var-data-in' type='checkbox'><label for='data-bbe1b1e4-940e-4a59-9469-e26ea2be54ca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e234a80d-c067-4564-932c-b4cdb57ede6d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e234a80d-c067-4564-932c-b4cdb57ede6d' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-52f6440b-7b29-47c5-b0c2-14c50253d164' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-52f6440b-7b29-47c5-b0c2-14c50253d164' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-549f271e-5166-4d2f-a18e-335b377b1eae' class='xr-var-data-in' type='checkbox'><label for='data-549f271e-5166-4d2f-a18e-335b377b1eae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-fea8b358-a198-4c7a-9273-d891896e42fc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fea8b358-a198-4c7a-9273-d891896e42fc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-99284e24-4c2b-451c-8ffe-e58796ea96c2' class='xr-var-data-in' type='checkbox'><label for='data-99284e24-4c2b-451c-8ffe-e58796ea96c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-5fe58077-fe9b-4662-bd24-1f99042610f5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5fe58077-fe9b-4662-bd24-1f99042610f5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-17bd86db-b078-4d5a-8e08-418b7ac51a89' class='xr-var-data-in' type='checkbox'><label for='data-17bd86db-b078-4d5a-8e08-418b7ac51a89' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8493e293-364f-4011-a279-310d728b65d0' class='xr-section-summary-in' type='checkbox'  ><label for='section-8493e293-364f-4011-a279-310d728b65d0' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-778993a5-1eb9-479f-be1c-a322763d1189' class='xr-index-data-in' type='checkbox'/><label for='index-778993a5-1eb9-479f-be1c-a322763d1189' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d6f73a29-3335-4964-a4ce-45502aeb430a' class='xr-index-data-in' type='checkbox'/><label for='index-d6f73a29-3335-4964-a4ce-45502aeb430a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3a1272c4-4d18-4683-83a2-d2caa078ba33' class='xr-index-data-in' type='checkbox'/><label for='index-3a1272c4-4d18-4683-83a2-d2caa078ba33' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-771a01a9-0346-4292-aa37-6f56cb7bead8' class='xr-section-summary-in' type='checkbox'  ><label for='section-771a01a9-0346-4292-aa37-6f56cb7bead8' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.237407+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_prior_predictiveecb40ebb-3507-461f-a4e9-b278ae410e48" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_prior_predictiveecb40ebb-3507-461f-a4e9-b278ae410e48" class = "xr-section-summary">prior_predictive</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-2a3d0595-2d70-4c1c-929c-24c09ad74087' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-2a3d0595-2d70-4c1c-929c-24c09ad74087' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-02a5bdd2-7095-49b2-a63f-e68b7dd26e5e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-02a5bdd2-7095-49b2-a63f-e68b7dd26e5e' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-aba0f4a8-287d-4cdb-8149-44a37e2fc4db' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-aba0f4a8-287d-4cdb-8149-44a37e2fc4db' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1099bc8f-451c-403c-ab72-30d7cfa8e14e' class='xr-var-data-in' type='checkbox'><label for='data-1099bc8f-451c-403c-ab72-30d7cfa8e14e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-1c21c586-1c90-48df-992f-5c86e7ebfb4c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c21c586-1c90-48df-992f-5c86e7ebfb4c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e657481e-79dc-48e1-b67f-a0bcdc14898c' class='xr-var-data-in' type='checkbox'><label for='data-e657481e-79dc-48e1-b67f-a0bcdc14898c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-eea77e9e-c631-408c-a0f0-aa17d71a8976' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eea77e9e-c631-408c-a0f0-aa17d71a8976' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-34f047e2-97f4-4de0-94b4-590e8e7f8a8f' class='xr-var-data-in' type='checkbox'><label for='data-34f047e2-97f4-4de0-94b4-590e8e7f8a8f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-99286668-13a2-469b-b9a3-071caa2e973d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-99286668-13a2-469b-b9a3-071caa2e973d' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-570bd10b-56ba-4516-b59f-e12306297177' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-570bd10b-56ba-4516-b59f-e12306297177' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-844ebaf9-24f8-4224-b643-9e3f259f727d' class='xr-var-data-in' type='checkbox'><label for='data-844ebaf9-24f8-4224-b643-9e3f259f727d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c1e1dc6e-8c34-4ffd-a037-217b395aac03' class='xr-section-summary-in' type='checkbox'  ><label for='section-c1e1dc6e-8c34-4ffd-a037-217b395aac03' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-dc694a75-1828-4ac5-87e9-b4109f31980a' class='xr-index-data-in' type='checkbox'/><label for='index-dc694a75-1828-4ac5-87e9-b4109f31980a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-38638eac-2b03-408d-96b5-c5cb9dd9ff30' class='xr-index-data-in' type='checkbox'/><label for='index-38638eac-2b03-408d-96b5-c5cb9dd9ff30' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-b7c2e287-b4ab-4bf6-8c45-f370e1d19ccf' class='xr-index-data-in' type='checkbox'/><label for='index-b7c2e287-b4ab-4bf6-8c45-f370e1d19ccf' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e3928e36-b494-48bf-af36-a66e24178fdd' class='xr-section-summary-in' type='checkbox'  ><label for='section-e3928e36-b494-48bf-af36-a66e24178fdd' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.238442+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_observed_data09d1054d-8cdf-4fff-ab39-8f97b12d034a" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_observed_data09d1054d-8cdf-4fff-ab39-8f97b12d034a" class = "xr-section-summary">observed_data</label>
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
Attributes: (4)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-99b472fc-7824-4a63-a40b-ecc3a7d7c429' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-99b472fc-7824-4a63-a40b-ecc3a7d7c429' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-f4193609-a772-41d3-ba64-437fe129243e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f4193609-a772-41d3-ba64-437fe129243e' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-33ac3c6a-032f-4143-8aba-b1ca9890fced' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-33ac3c6a-032f-4143-8aba-b1ca9890fced' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-26835826-b440-4cfd-81c6-7e39f9beea8c' class='xr-var-data-in' type='checkbox'><label for='data-26835826-b440-4cfd-81c6-7e39f9beea8c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-743f596a-199c-4a09-a545-c3c9ec9c66e2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-743f596a-199c-4a09-a545-c3c9ec9c66e2' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-b659ff2b-16cc-4f37-bfda-c0d03e9daeac' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b659ff2b-16cc-4f37-bfda-c0d03e9daeac' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7015e34d-18f0-4a37-aef2-0169e1d59316' class='xr-var-data-in' type='checkbox'><label for='data-7015e34d-18f0-4a37-aef2-0169e1d59316' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123 values with dtype=int64]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1efad5b6-9413-4f29-a177-56fd9abaf6b1' class='xr-section-summary-in' type='checkbox'  ><label for='section-1efad5b6-9413-4f29-a177-56fd9abaf6b1' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-b6da5990-b37f-4676-87cd-68b977342ee8' class='xr-index-data-in' type='checkbox'/><label for='index-b6da5990-b37f-4676-87cd-68b977342ee8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2d65965f-ce96-4d1a-8e30-e18c8bdc9a0c' class='xr-section-summary-in' type='checkbox'  ><label for='section-2d65965f-ce96-4d1a-8e30-e18c8bdc9a0c' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.239093+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
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
    2022-08-08 2022-08-09 2022-08-10 2022-08-11 ... 2022-08-20 2022-08-21 2022-08-22
    Coordinates:
      * obs_dim_0  (obs_dim_0) datetime64[ns] 120B 2022-08-08 ... 2022-08-22

``` python
# idata without dates as coordinates
idata_wo_dates = forecasttools.nhsn_flu_forecast_wo_dates
print(idata_wo_dates["observed_data"]["obs"]["obs_dim_0"][:15])
```

    <xarray.DataArray 'obs_dim_0' (obs_dim_0: 15)> Size: 120B
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
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

\`\`\`{details markdown=1}
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
this waiver of copyright interest. \`\`\`

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
