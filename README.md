# Data_Viz_2024

MATLAB scripts and figures for visualising and benchmarking hourly energy production data for Corsica
(thermal, hydro, solar, wind, biomass, imports) based on EDF Open Data (hourly).  
Outputs include pre-generated MATLAB `.fig` files for multiple horizons (e.g., Forecast_1h.fig … Forecast_10h.fig).

Data source: https://opendata-corse.edf.fr/

## Repository contents
- **main.m** — entry script to load `Data.xlsx`, prepare series, and generate forecast visualisations.
- **Data.xlsx** — hourly dataset (production by source), used by `main.m`.
- **Forecast_*h.fig** — saved MATLAB figures for horizons 1–10 h.
- **LICENSE** — GPL-3.0

## Quick start (MATLAB)
```matlab
% 1) Open MATLAB in this repository folder
% 2) Run:
run('main.m')

% main.m expects Data.xlsx in the same folder.
% It generates/updates Forecast_*h.fig figures.
