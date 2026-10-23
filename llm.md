# MIMO-ELM: scientific scope and reproduction contract

## Research purpose and authority

Forecast multiple sources of an electricity mix jointly using a shared random hidden representation and a closed-form output fit. The purpose is to use cross-series information with limited training cost, not to assume that AI or a larger network is intrinsically preferable. The published MIMO approach and the later multi-horizon reconciliation extension are different studies.

Authority reviewed: author manuscript arXiv:2508.12764v2, Sections 4.3–4.4, compared with repository commit `6894ac5a06d20cb56a690cd44b2220250f5c0151`. Article DOI: https://doi.org/10.1016/j.apenergy.2026.127599 . The publisher version has not been checked line by line. Read the discrepancies below before describing the script as reproducing the article.

## Verified differences: manuscript and main.m

| Item | Author manuscript | Distributed MATLAB script |
|---|---|---|
| Input features | 7*48+2 = 338 | 9*48 = 432; sine/cosine repeated within every lag |
| Random initializations | 50 | 2 |
| Selection criterion | In-sample performance | Mean output RMSE on XTest/YTest |

The final item means the nominal test set participates in choosing a random initialization. It is not an untouched final evaluation set. Do not cite a rerun of this script as an independent confirmation of the article's out-of-sample results. These findings concern this distributed script and the author-version description; they do not prove what pipeline generated the published results. Resolving the differences requires a separately reviewed code/protocol change. The original script is preserved.

## MATLAB inputs, outputs and physical units

`main.m` is a script, not a parameterized API. It reads `Data.xlsx` in the current directory. Its seven outputs, in order, are total production, thermal, hydro, solar, wind, bioenergy and imports, all in MW (power, not MWh). The total follows demand approximately; losses and operational conventions matter. The cost column in the source spreadsheet is not among those seven outputs.

Default script settings: `window_size=48`, `prediction_horizon=1`, `numHiddenUnits=4096`, `numInitializations=2`. A horizon is measured in hourly rows; ensure chronological ordering, a regular time grid and a defined treatment of daylight-saving transitions before interpreting it as elapsed time. Local hour is decoded in Europe/Paris and shifted by one before sine/cosine encoding. Do not silently replace these features when claiming exact script reproduction.

The script clips the selected input power columns at zero, then replaces NaN in constructed X/Y with zero. Missing observations are not physically zero generation: review this convention before reuse. Input vectors and target units must remain consistent. Default parameters are not enforced by a general input-validation API.

For each window starting at i, inputs are rows i through i+47 and the target is row i+48+h-1. Constructed samples are split chronologically 80/20. The script reserves `N-window_size-h` windows (one fewer than the maximal inclusive count). The paper's chronological split must not be confused with the separate issue of selecting initializations on test errors.

The hidden matrix is H=ReLU(X*W_in'+bias'), and output weights are pinv(H)*Y. Random input weights are not optimized iteratively. Predictions are clipped nonnegative. There is no guarantee in this MATLAB script that component forecasts sum to the separately predicted total; reconciliation belongs to the distinct extension.

RMSE/MAE retain MW; normalized errors divide by each target series' mean. Zero or very small mean makes normalized errors undefined or unstable. A successful execution is not a scientific replication until input data, feature construction, seed/selection protocol and scoring are reconciled with the article.

## Minimal inspection without launching a large fit

```matlab
window_size=48; source_count=7; time_channels=2;
script_feature_count=window_size*(source_count+time_channels); % 432
article_feature_count=window_size*source_count+time_channels;  % 338
```

To run the preserved implementation, start MATLAB in the repository and execute `main`; this launches a substantial pseudoinverse fit with 4096 hidden units and may consume considerable memory. The documentation audit does not imply that the full fit was rerun. Use `main_SISO` and `main_LSTM` only after separately checking their settings and dependencies.

## Separate Python MIMO-MH extension

`mimo_mh/` contains a multi-horizon pipeline and reconciliation code credited in README to Yanis Jheelan and Elena Magliaro. It is associated with a separate manuscript, not the published MIMO validation. Its broader scientific protocol is outside this contract's paper-to-code review; do not transfer published MIMO scores to it.

The isolated `reconciliation.py` utility expects arrays shaped (samples,horizons,outputs), with total first, followed by all mutually exclusive component series. `create_S_from_outputs` builds the sum matrix; `estimate_W_from_train_residuals` estimates covariance/variance from training residuals; `reconcile_all_horizons` projects forecasts onto the sum constraint. Consistent units and a correct hierarchy are required. A sum-consistency check validates only that algebraic property, not forecast accuracy, nonnegative outputs or physical feasibility.

## Citation, rights and status

Published MIMO authors: Cyril Voyant, Milan Despotovic, Luis Garcia-Gutierrez, Mohammed Asloune, Yves-Marie Saint-Drenan, Jean-Laurent Duchaud, Ghjuvan Antone Faggianelli and Elena Magliaro. See CITATION.cff for full article citation. Preserve separate credits/status for MIMO-MH from README.

Software concept DOI: https://doi.org/10.5281/zenodo.23562559 ; v2.0.0: https://doi.org/10.5281/zenodo.23562560 . Code GPL-3.0; EDF Corse data under Licence Ouverte, with source attribution. Later documentation is not retroactively part of an existing archive. These descriptors do not deploy an executable WebSemantic adapter.
