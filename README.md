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
<small>shape: (58, 3)</small>

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
<small>shape: (4_876, 8)</small>

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
                  <input id="idata_posteriorc5241ff3-4aea-4fc5-8dcf-61bcbce11c1a" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriorc5241ff3-4aea-4fc5-8dcf-61bcbce11c1a" class = "xr-section-summary">posterior</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-c3ae67ec-16c3-4007-86bf-9140b3747ce3' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c3ae67ec-16c3-4007-86bf-9140b3747ce3' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-3da39cae-ea22-454d-9026-303d0e0a58a6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-3da39cae-ea22-454d-9026-303d0e0a58a6' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-b4ddb66e-482a-40dc-8ce5-06175db04178' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b4ddb66e-482a-40dc-8ce5-06175db04178' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0d3c9e85-cf0b-4e5c-a34e-9575e444acc3' class='xr-var-data-in' type='checkbox'><label for='data-0d3c9e85-cf0b-4e5c-a34e-9575e444acc3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-ae9254f0-109d-4eb7-a76f-2fe00bf22071' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ae9254f0-109d-4eb7-a76f-2fe00bf22071' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-03d02336-f0f6-4f27-a843-a2ffa5baba25' class='xr-var-data-in' type='checkbox'><label for='data-03d02336-f0f6-4f27-a843-a2ffa5baba25' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-9132c500-c137-44a3-a278-671b3020ebc2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9132c500-c137-44a3-a278-671b3020ebc2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4a07643a-f5c8-4b4e-8a32-1a5bae02b908' class='xr-var-data-in' type='checkbox'><label for='data-4a07643a-f5c8-4b4e-8a32-1a5bae02b908' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bd945e4a-68ff-4811-9abb-939597512935' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bd945e4a-68ff-4811-9abb-939597512935' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-73337223-4abe-4a77-b234-001c82297e02' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-73337223-4abe-4a77-b234-001c82297e02' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3a4cdd3e-2dfb-4667-9fdc-726fb6b16344' class='xr-var-data-in' type='checkbox'><label for='data-3a4cdd3e-2dfb-4667-9fdc-726fb6b16344' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-2060ddf7-2b4a-4881-b7ba-d30f28354e85' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2060ddf7-2b4a-4881-b7ba-d30f28354e85' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7a71fa95-999d-47b5-ac2e-1cf51a87aaa1' class='xr-var-data-in' type='checkbox'><label for='data-7a71fa95-999d-47b5-ac2e-1cf51a87aaa1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-ce7e0197-ff98-4186-9fb4-1df249ad0290' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ce7e0197-ff98-4186-9fb4-1df249ad0290' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c192e1bb-3c61-4595-ab26-83eccf1296aa' class='xr-var-data-in' type='checkbox'><label for='data-c192e1bb-3c61-4595-ab26-83eccf1296aa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-968e07f1-3a3c-4d67-a478-af30d5df0258' class='xr-section-summary-in' type='checkbox'  ><label for='section-968e07f1-3a3c-4d67-a478-af30d5df0258' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d4f4185e-7669-4eb2-858f-c53483323821' class='xr-index-data-in' type='checkbox'/><label for='index-d4f4185e-7669-4eb2-858f-c53483323821' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fc618216-2a36-4afb-a768-6d15ad5d8f4f' class='xr-index-data-in' type='checkbox'/><label for='index-fc618216-2a36-4afb-a768-6d15ad5d8f4f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-a9c98d2a-1ff5-41a3-b98b-fde46460c9af' class='xr-index-data-in' type='checkbox'/><label for='index-a9c98d2a-1ff5-41a3-b98b-fde46460c9af' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-57bf0ce3-7c30-4926-8880-46cc0941825f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-57bf0ce3-7c30-4926-8880-46cc0941825f' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.119636+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_posterior_predictive957aed10-854a-40b5-8656-7e8413b2ab1b" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior_predictive957aed10-854a-40b5-8656-7e8413b2ab1b" class = "xr-section-summary">posterior_predictive</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cdb24841-8936-4624-acea-c9e611a7d465' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cdb24841-8936-4624-acea-c9e611a7d465' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 151</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-006252c6-0ea2-41f9-9ba6-9ea46ddb23cd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-006252c6-0ea2-41f9-9ba6-9ea46ddb23cd' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-434197c8-6d41-4c7a-b6e3-ae88ba146930' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-434197c8-6d41-4c7a-b6e3-ae88ba146930' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-beda975a-9674-478d-a431-0a34a9e132d8' class='xr-var-data-in' type='checkbox'><label for='data-beda975a-9674-478d-a431-0a34a9e132d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-217f4673-c916-4898-bccd-317be794d0dd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-217f4673-c916-4898-bccd-317be794d0dd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c0daa612-c12e-4c57-8c7d-ba5567ae019c' class='xr-var-data-in' type='checkbox'><label for='data-c0daa612-c12e-4c57-8c7d-ba5567ae019c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2023-01-05</div><input id='attrs-d99192dd-8611-42eb-9d2c-98453189c92b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d99192dd-8611-42eb-9d2c-98453189c92b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f04f4650-e9cc-4ddb-b3df-9520f56a077f' class='xr-var-data-in' type='checkbox'><label for='data-f04f4650-e9cc-4ddb-b3df-9520f56a077f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2023-01-05T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-611ac5ba-4671-4293-9f9b-d7142bb313da' class='xr-section-summary-in' type='checkbox'  checked><label for='section-611ac5ba-4671-4293-9f9b-d7142bb313da' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-0a925a58-7a21-41ed-af1a-11c2a061c851' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0a925a58-7a21-41ed-af1a-11c2a061c851' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bbb76a45-1ce8-492e-8deb-0324a02cb678' class='xr-var-data-in' type='checkbox'><label for='data-bbb76a45-1ce8-492e-8deb-0324a02cb678' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[151000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-89fc1dc7-c436-47dd-a159-30ff260a65a2' class='xr-section-summary-in' type='checkbox'  ><label for='section-89fc1dc7-c436-47dd-a159-30ff260a65a2' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-75098e3a-d5bc-44cf-a445-30a48105155b' class='xr-index-data-in' type='checkbox'/><label for='index-75098e3a-d5bc-44cf-a445-30a48105155b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-d7441360-ab3f-4d85-856e-9d23a9466f69' class='xr-index-data-in' type='checkbox'/><label for='index-d7441360-ab3f-4d85-856e-9d23a9466f69' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-50a1b8d5-9609-40e1-9f99-f0ec2456c4ce' class='xr-index-data-in' type='checkbox'/><label for='index-50a1b8d5-9609-40e1-9f99-f0ec2456c4ce' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-12-27&#x27;, &#x27;2022-12-28&#x27;, &#x27;2022-12-29&#x27;, &#x27;2022-12-30&#x27;,
               &#x27;2022-12-31&#x27;, &#x27;2023-01-01&#x27;, &#x27;2023-01-02&#x27;, &#x27;2023-01-03&#x27;,
               &#x27;2023-01-04&#x27;, &#x27;2023-01-05&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=151, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4367e378-339e-43de-a1db-5e222fb43223' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4367e378-339e-43de-a1db-5e222fb43223' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.236298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_log_likelihood5c65e195-00d7-4d9a-a557-8aadd021c60e" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_log_likelihood5c65e195-00d7-4d9a-a557-8aadd021c60e" class = "xr-section-summary">log_likelihood</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-fdee7d6f-5256-480e-91b2-97e5e24b8642' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-fdee7d6f-5256-480e-91b2-97e5e24b8642' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-10928d8b-92d4-424b-af96-fbdc0ade4afd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-10928d8b-92d4-424b-af96-fbdc0ade4afd' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-44c73ee9-22f4-4643-88c0-84af89cbe9f3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-44c73ee9-22f4-4643-88c0-84af89cbe9f3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0f8876c8-3de4-402f-b3af-318a32d394f0' class='xr-var-data-in' type='checkbox'><label for='data-0f8876c8-3de4-402f-b3af-318a32d394f0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-01cc9fc2-cd70-4fb8-bbcc-618fc7747d19' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-01cc9fc2-cd70-4fb8-bbcc-618fc7747d19' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-27870150-739b-42f6-af56-2924e96a6b1d' class='xr-var-data-in' type='checkbox'><label for='data-27870150-739b-42f6-af56-2924e96a6b1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 118 119 120 121 122</div><input id='attrs-94b836fa-b3ae-424f-867d-e01d914b5313' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-94b836fa-b3ae-424f-867d-e01d914b5313' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-19d5da9a-8f5a-48aa-b169-572f126d00ba' class='xr-var-data-in' type='checkbox'><label for='data-19d5da9a-8f5a-48aa-b169-572f126d00ba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-54df3686-3869-49f5-9597-4d3f9baa1f0a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54df3686-3869-49f5-9597-4d3f9baa1f0a' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-cf4e7dcb-096f-4c9d-be6b-dc9ebcc3fdd4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cf4e7dcb-096f-4c9d-be6b-dc9ebcc3fdd4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4002b8c8-ed09-49e7-9264-89f98e4d178d' class='xr-var-data-in' type='checkbox'><label for='data-4002b8c8-ed09-49e7-9264-89f98e4d178d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-38b8ee73-2199-41cd-a154-acdd00a107d6' class='xr-section-summary-in' type='checkbox'  ><label for='section-38b8ee73-2199-41cd-a154-acdd00a107d6' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-b190581c-34a8-420b-820d-84c2120b038b' class='xr-index-data-in' type='checkbox'/><label for='index-b190581c-34a8-420b-820d-84c2120b038b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-49653a6a-8b44-42eb-a85d-c564d9462155' class='xr-index-data-in' type='checkbox'/><label for='index-49653a6a-8b44-42eb-a85d-c564d9462155' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-fd063101-a6c4-46ac-bae5-3e930813b7eb' class='xr-index-data-in' type='checkbox'/><label for='index-fd063101-a6c4-46ac-bae5-3e930813b7eb' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
      dtype=&#x27;int64&#x27;, name=&#x27;obs_dim_0&#x27;, length=123))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d0ca3910-b7f8-4850-b532-853e955e5595' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d0ca3910-b7f8-4850-b532-853e955e5595' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.235298+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_sample_statsca3f3e3b-bdd7-4716-8787-27c594199285" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_statsca3f3e3b-bdd7-4716-8787-27c594199285" class = "xr-section-summary">sample_stats</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-6a7bcff0-9ce3-43c5-80c6-c50412107b6f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6a7bcff0-9ce3-43c5-80c6-c50412107b6f' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-54c66ec5-d407-48de-b041-42ed1da455fc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54c66ec5-d407-48de-b041-42ed1da455fc' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-f2d48de1-8153-49ae-9d31-d9b2bab10f6a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f2d48de1-8153-49ae-9d31-d9b2bab10f6a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f417ad56-886c-4db4-b6f4-cbb916460b37' class='xr-var-data-in' type='checkbox'><label for='data-f417ad56-886c-4db4-b6f4-cbb916460b37' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-dc9c0c35-8db3-428a-858f-a833820ba3b7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dc9c0c35-8db3-428a-858f-a833820ba3b7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52ae4fd3-eea4-405c-a341-1d5e7811b303' class='xr-var-data-in' type='checkbox'><label for='data-52ae4fd3-eea4-405c-a341-1d5e7811b303' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f2383300-59af-4301-8668-7f5a1e9cdcb7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f2383300-59af-4301-8668-7f5a1e9cdcb7' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-32163e40-20be-4100-9812-380cea644dc9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-32163e40-20be-4100-9812-380cea644dc9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0e418200-c0e1-4189-bf1e-242b992da02c' class='xr-var-data-in' type='checkbox'><label for='data-0e418200-c0e1-4189-bf1e-242b992da02c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=bool]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-77699b86-5220-42a4-9e29-5d9ebe945583' class='xr-section-summary-in' type='checkbox'  ><label for='section-77699b86-5220-42a4-9e29-5d9ebe945583' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-8bf32df9-70d5-464b-8420-966870f1d32e' class='xr-index-data-in' type='checkbox'/><label for='index-8bf32df9-70d5-464b-8420-966870f1d32e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-72cd8fc5-7941-4bbd-b4e5-89310d010b06' class='xr-index-data-in' type='checkbox'/><label for='index-72cd8fc5-7941-4bbd-b4e5-89310d010b06' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1b9ce1c0-588c-465a-8087-b21f27f99566' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1b9ce1c0-588c-465a-8087-b21f27f99566' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.122620+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_priord870637e-e6ec-4750-bc3f-3b330669c38e" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_priord870637e-e6ec-4750-bc3f-3b330669c38e" class = "xr-section-summary">prior</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-a59315d4-edd5-4ebc-80ac-33ccaf39eea8' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-a59315d4-edd5-4ebc-80ac-33ccaf39eea8' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>beta_coeffs_dim_0</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-28e4f630-d31a-4871-9fc8-315ac645d0bd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-28e4f630-d31a-4871-9fc8-315ac645d0bd' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-21f1c5c9-8861-4411-8b5a-21d2b600e6dd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21f1c5c9-8861-4411-8b5a-21d2b600e6dd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-64b55334-f294-4d31-9e5e-c44cd9526dc7' class='xr-var-data-in' type='checkbox'><label for='data-64b55334-f294-4d31-9e5e-c44cd9526dc7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-c2397e98-4062-4d29-8c14-70eec4324745' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c2397e98-4062-4d29-8c14-70eec4324745' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5190fca2-1f5b-4981-8064-857f77ddaf0f' class='xr-var-data-in' type='checkbox'><label for='data-5190fca2-1f5b-4981-8064-857f77ddaf0f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>beta_coeffs_dim_0</span></div><div class='xr-var-dims'>(beta_coeffs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-b7ddf5c2-c0f2-4045-b038-7b3a3828bd6d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7ddf5c2-c0f2-4045-b038-7b3a3828bd6d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-73edc8bf-01ab-4011-8029-412d6149ddd4' class='xr-var-data-in' type='checkbox'><label for='data-73edc8bf-01ab-4011-8029-412d6149ddd4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b1bb42c9-9e6a-4950-aef0-a8c43182d5dd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-b1bb42c9-9e6a-4950-aef0-a8c43182d5dd' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-41a55031-27f9-4bfc-b7de-689bdbce65f2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-41a55031-27f9-4bfc-b7de-689bdbce65f2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b01d532e-8b21-42b8-980f-c4dbbfcfd0c0' class='xr-var-data-in' type='checkbox'><label for='data-b01d532e-8b21-42b8-980f-c4dbbfcfd0c0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_coeffs</span></div><div class='xr-var-dims'>(chain, draw, beta_coeffs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-204227d1-ee78-4dc3-8da7-7886851a4240' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-204227d1-ee78-4dc3-8da7-7886851a4240' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4c6ef2fe-fafd-49da-8a56-4078ddb33fae' class='xr-var-data-in' type='checkbox'><label for='data-4c6ef2fe-fafd-49da-8a56-4078ddb33fae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[8000 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>shift</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-2f5fd1c0-efe0-4094-8a75-e843e64d763e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f5fd1c0-efe0-4094-8a75-e843e64d763e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e5cac1c1-21c4-4306-8203-0ab4d2cdcbec' class='xr-var-data-in' type='checkbox'><label for='data-e5cac1c1-21c4-4306-8203-0ab4d2cdcbec' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[1000 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ddc0670d-fd9b-48e3-ae72-2f5b3bec9951' class='xr-section-summary-in' type='checkbox'  ><label for='section-ddc0670d-fd9b-48e3-ae72-2f5b3bec9951' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-586295a6-c765-4e9d-bb14-894f11ba28e2' class='xr-index-data-in' type='checkbox'/><label for='index-586295a6-c765-4e9d-bb14-894f11ba28e2' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-e7316876-1a8f-4d81-988f-2f7145353756' class='xr-index-data-in' type='checkbox'/><label for='index-e7316876-1a8f-4d81-988f-2f7145353756' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>beta_coeffs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4acb3f76-a09d-45cb-b779-d1a6c66f1c90' class='xr-index-data-in' type='checkbox'/><label for='index-4acb3f76-a09d-45cb-b779-d1a6c66f1c90' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;beta_coeffs_dim_0&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-980d154a-5c37-43a2-b525-8567369e9f72' class='xr-section-summary-in' type='checkbox'  checked><label for='section-980d154a-5c37-43a2-b525-8567369e9f72' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.237407+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_prior_predictive432a0c06-e934-4476-9444-5af779d48cd5" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_prior_predictive432a0c06-e934-4476-9444-5af779d48cd5" class = "xr-section-summary">prior_predictive</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cb23bac1-1df5-4e40-9a72-f6dfe8a308e4' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cb23bac1-1df5-4e40-9a72-f6dfe8a308e4' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-4990f5d9-9fff-4f67-a5dd-3a73542d5874' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4990f5d9-9fff-4f67-a5dd-3a73542d5874' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-7b621e68-f494-4bfc-b856-816ce4790d7e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7b621e68-f494-4bfc-b856-816ce4790d7e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bf8f0c66-7698-49e6-ba7a-6df00f5c32e7' class='xr-var-data-in' type='checkbox'><label for='data-bf8f0c66-7698-49e6-ba7a-6df00f5c32e7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-fb97fd3a-7674-457e-b389-f7837688cbca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fb97fd3a-7674-457e-b389-f7837688cbca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-31b189bf-9e09-419f-85e6-6ebda0575a23' class='xr-var-data-in' type='checkbox'><label for='data-31b189bf-9e09-419f-85e6-6ebda0575a23' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-5f20060d-29af-49e4-b39d-b6214f84b6a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5f20060d-29af-49e4-b39d-b6214f84b6a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2e49406d-7b9d-437f-a396-ccbe5cbea4bd' class='xr-var-data-in' type='checkbox'><label for='data-2e49406d-7b9d-437f-a396-ccbe5cbea4bd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-04276a94-90f5-4a2d-9f6c-e8ef5053b5bb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-04276a94-90f5-4a2d-9f6c-e8ef5053b5bb' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(chain, draw, obs_dim_0)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-855676fb-b201-44d3-919f-5ccd507737f5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-855676fb-b201-44d3-919f-5ccd507737f5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2ccbd48d-78df-4f25-bc39-114e006568ca' class='xr-var-data-in' type='checkbox'><label for='data-2ccbd48d-78df-4f25-bc39-114e006568ca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123000 values with dtype=int32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-29d56370-0faf-46b8-ba55-5174bc46cf7e' class='xr-section-summary-in' type='checkbox'  ><label for='section-29d56370-0faf-46b8-ba55-5174bc46cf7e' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-967ec51d-9348-423d-b4de-ea04043cd865' class='xr-index-data-in' type='checkbox'/><label for='index-967ec51d-9348-423d-b4de-ea04043cd865' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-eff3d184-5ace-4438-89c8-2f2219df4e1c' class='xr-index-data-in' type='checkbox'/><label for='index-eff3d184-5ace-4438-89c8-2f2219df4e1c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-aa6b0197-0764-48eb-a7e9-9db2abe92d7b' class='xr-index-data-in' type='checkbox'/><label for='index-aa6b0197-0764-48eb-a7e9-9db2abe92d7b' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4ec0aef8-5aae-4659-9e66-bc68f66558da' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4ec0aef8-5aae-4659-9e66-bc68f66558da' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.238442+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            &#10;            <li class = "xr-section-item">
                  <input id="idata_observed_datab9f8cb1c-5431-427f-a377-efaeb04e8872" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_observed_datab9f8cb1c-5431-427f-a377-efaeb04e8872" class = "xr-section-summary">observed_data</label>
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
    inference_library_version:  0.15.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-f4cacb26-fe1b-4c31-a24b-c33c8fbf30aa' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f4cacb26-fe1b-4c31-a24b-c33c8fbf30aa' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>obs_dim_0</span>: 123</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6b3f7ad2-2b7c-4426-ab83-567a096c43c5' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6b3f7ad2-2b7c-4426-ab83-567a096c43c5' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>obs_dim_0</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2022-08-08 ... 2022-12-08</div><input id='attrs-c8480e75-673e-410a-9ee0-ba4d6968831a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c8480e75-673e-410a-9ee0-ba4d6968831a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5f2a4fe1-5fa1-444b-a48c-09c2b6135dfd' class='xr-var-data-in' type='checkbox'><label for='data-5f2a4fe1-5fa1-444b-a48c-09c2b6135dfd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2022-08-08T00:00:00.000000000&#x27;, &#x27;2022-08-09T00:00:00.000000000&#x27;,
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
       &#x27;2022-12-08T00:00:00.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f46e4432-3dd3-4ec9-b5ed-4d582a668d89' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f46e4432-3dd3-4ec9-b5ed-4d582a668d89' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>obs</span></div><div class='xr-var-dims'>(obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-9cbb2af4-f2e5-4aff-aa71-31c104bc848d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9cbb2af4-f2e5-4aff-aa71-31c104bc848d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d7c9070b-28a0-4a68-b07e-61a9cdbdb9c4' class='xr-var-data-in' type='checkbox'><label for='data-d7c9070b-28a0-4a68-b07e-61a9cdbdb9c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[123 values with dtype=int64]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9b3200ce-a10b-4dbe-93b7-15f0890236cd' class='xr-section-summary-in' type='checkbox'  ><label for='section-9b3200ce-a10b-4dbe-93b7-15f0890236cd' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>obs_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-ec366d55-a3b1-40a9-97ce-131cffc0f069' class='xr-index-data-in' type='checkbox'/><label for='index-ec366d55-a3b1-40a9-97ce-131cffc0f069' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2022-08-08&#x27;, &#x27;2022-08-09&#x27;, &#x27;2022-08-10&#x27;, &#x27;2022-08-11&#x27;,
               &#x27;2022-08-12&#x27;, &#x27;2022-08-13&#x27;, &#x27;2022-08-14&#x27;, &#x27;2022-08-15&#x27;,
               &#x27;2022-08-16&#x27;, &#x27;2022-08-17&#x27;,
               ...
               &#x27;2022-11-29&#x27;, &#x27;2022-11-30&#x27;, &#x27;2022-12-01&#x27;, &#x27;2022-12-02&#x27;,
               &#x27;2022-12-03&#x27;, &#x27;2022-12-04&#x27;, &#x27;2022-12-05&#x27;, &#x27;2022-12-06&#x27;,
               &#x27;2022-12-07&#x27;, &#x27;2022-12-08&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;obs_dim_0&#x27;, length=123, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-00e4afa1-a8c1-4524-b261-6c89708f335f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-00e4afa1-a8c1-4524-b261-6c89708f335f' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-10-24T16:45:20.239093+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.19.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.15.3</dd></dl></div></li></ul></div></div><br></div>
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
