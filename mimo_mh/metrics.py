# ==============================================================================
# metrics.py
# ==============================================================================
# Functions for computing and handling performance metrics.

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import re
from statsmodels.tsa.stattools import pacf

# COMPUTE ALL METRICS
# -----------------------------------------------------------------------------
# What this does:
# • Groups the long results table `df` by (model, horizon, output).
# • For each group, computes:
#     - RMSE, MAE, MBE (error stats) and R² (goodness of fit).
#     - Normalized versions nRMSE, nMAE, nMBE using mean(y_true) as the scale.
# • Returns a tidy `metrics_df` with one row per (model, horizon, output).
#
# Notes:
# • If mean(y_true) == 0, normalized metrics are set to NaN to avoid division by zero.
# • R² requires at least 2 points; otherwise it’s NaN.
# • Expects columns: ['model','horizon','output','y_true','y_pred'] in `df`.
def compute_all_metrics(df):
    """
    Computes a standard set of metrics (nRMSE, nMAE, nMBE, R2, etc.) from a results DataFrame.

    Utility: Aggregates and calculates metrics per model, horizon, output.

    Arguments:
    - df: DataFrame, results with y_true and y_pred.

    Returns:
    - metrics_df: DataFrame with computed metrics.
    """
    print(">>> Computing performance metrics...")
    if df.empty:
        return pd.DataFrame(columns=["model", "horizon", "output", "nRMSE", "nMAE", "nMBE", "R2", "RMSE", "MAE", "MBE"])
    metrics_list = []
    # Group by model, horizon, and output variable to calculate metrics for each segment
    grouped = df.groupby(['model', 'horizon', 'output'])
    for (model, horizon, output), group in grouped:
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        
        if len(y_true) == 0: continue

        error = y_pred - y_true
        mae = np.mean(np.abs(error))
        mbe = np.mean(error)
        rmse = np.sqrt(np.mean(error ** 2))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        
        mean_true = np.mean(y_true)
        # Calculate normalized metrics, handling division by zero
        nmae = mae / mean_true if mean_true != 0 else np.nan
        nmbe = mbe / mean_true if mean_true != 0 else np.nan
        nrmse = rmse / mean_true if mean_true != 0 else np.nan
        
        metrics_list.append([
            model, horizon, output,
            nrmse, nmae, nmbe, r2, rmse, mae, mbe
        ])
        
    metrics_df = pd.DataFrame(
        metrics_list,
        columns=["model", "horizon", "output", "nRMSE", "nMAE", "nMBE", "R2", "RMSE", "MAE", "MBE"]
    )
    print(f"    - Metrics computed for {len(metrics_df)} combinations.")
    print(">>> Metrics computation complete.\n")
    return metrics_df

def stack_external_metrics(forecast_dfs: dict[str, pd.DataFrame], model_name: str) -> pd.DataFrame:
    """
    Converts forecast DataFrames (like from Prophet) into a standardized metrics DataFrame.

    Utility: Computes metrics from external forecast files.

    Arguments:
    - forecast_dfs: dict {target: DataFrame}, forecasts.
    - model_name: str, name of the model (e.g., "Prophet").

    Returns:
    - pd.DataFrame with metrics.
    """
    print(f"    - Computing metrics for {model_name}...")
    frames = []
    _H_RE = re.compile(r"\d+") # Regex to find horizon number in column names like 'yhat24'
    
    for out_name, df in forecast_dfs.items():
        yhat_cols = [c for c in df.columns if str(c).lower().startswith("yhat")]
        
        for col in yhat_cols:
            m = _H_RE.findall(str(col))
            if not m: continue
            h = int(m[-1])

            sub = df[["y", col]].dropna()
            if sub.empty: continue
            
            y_true = sub["y"].to_numpy(float)
            y_pred = sub[col].to_numpy(float)
            
            # Re-use the metric calculation logic
            error = y_pred - y_true
            mae, mbe, rmse = np.mean(np.abs(error)), np.mean(error), np.sqrt(np.mean(error**2))
            r2 = r2_score(y_true, y_pred)
            mean_true = np.mean(y_true)
            
            nrmse = rmse / mean_true if mean_true != 0 else np.nan
            nmae = mae / mean_true if mean_true != 0 else np.nan
            nmbe = mbe / mean_true if mean_true != 0 else np.nan
            
            row = {
                "model": model_name, "horizon": h, "output": out_name,
                "nRMSE": nrmse, "nMAE": nmae, "nMBE": nmbe, "R2": r2, 
                "RMSE": rmse, "MAE": mae, "MBE": mbe
            }
            frames.append(row)
            
    print(f"    - {model_name} metrics finished.")
    return pd.DataFrame(frames)


def compute_correlations(df_results, by='output'):
    """
    Compute average Spearman correlation matrices.

    Utility: Calculates correlations between outputs (averaged over horizons) or horizons (averaged over outputs).

    Arguments:
    - df_results: DataFrame, results data.
    - by: str, 'output' or 'horizon'.

    Returns:
    - results: dict {model: DataFrame}, average correlation matrices.
    """
    if df_results.empty:
        return None
    
    models = df_results['model'].unique()
    results = {}
    
    for model in models:
        sub_df = df_results[df_results['model'] == model]
        
        if by == 'output':
            horizons = sub_df['horizon'].unique()
            corr_mats = []
            for h in horizons:
                pivot = sub_df[sub_df['horizon'] == h].pivot(index='datetime', columns='output', values='y_pred').dropna()
                if not pivot.empty:
                    corr_mats.append(pivot.corr(method='spearman'))  # CHANGEMENT: Spearman au lieu de Pearson
            if corr_mats:
                avg_corr = pd.concat(corr_mats).groupby(level=0).mean()  # Moyenne sur horizons
                results[model] = avg_corr
        elif by == 'horizon':
            outputs = sub_df['output'].unique()
            corr_mats = []
            for out in outputs:
                pivot = sub_df[sub_df['output'] == out].pivot(index='datetime', columns='horizon', values='y_pred').dropna()
                if not pivot.empty:
                    corr_mats.append(pivot.corr(method='spearman'))  # CHANGEMENT: Spearman au lieu de Pearson
            if corr_mats:
                avg_corr = pd.concat(corr_mats).groupby(level=0).mean()  # Moyenne sur outputs
                results[model] = avg_corr
    
    return results

def compute_pacf_matrix(df_results, output="solar_photovoltaic_mw", max_horizon=24):
    """
    Compute PACF matrix between horizons for a given output.

    Utility: Treats prediction errors per horizon as series and computes PACF.

    Arguments:
    - df_results: DataFrame, results.
    - output: str, target output.
    - max_horizon: int, max horizon.

    Returns:
    - pacf_matrices: dict {model: DataFrame}, PACF matrices.
    """
    pacf_matrices = {}

    for model in df_results["model"].unique():
        df_model = df_results[(df_results["model"] == model) & (df_results["output"] == output)]
        if df_model.empty:
            continue

        horizons = sorted(df_model["horizon"].unique())
        horizons = [h for h in horizons if h <= max_horizon]
        pacf_mat = np.zeros((len(horizons), len(horizons)))

        for i, h1 in enumerate(horizons):
            series_h1 = df_model[df_model["horizon"] == h1]["y_true"] - df_model[df_model["horizon"] == h1]["y_pred"]
            series_h1 = series_h1.dropna().values
            if len(series_h1) < 10:
                continue

            pacf_vals = pacf(series_h1, nlags=max_horizon, method="ywm")

            for j, h2 in enumerate(horizons):
                if h2 < len(pacf_vals):
                    pacf_mat[i, j] = pacf_vals[h2]
                else:
                    pacf_mat[i, j] = np.nan

        pacf_matrices[model] = pd.DataFrame(pacf_mat, index=horizons, columns=horizons)

    return pacf_matrices
    
def get_metrics(y_true, y_pred):
    """
    Compute normalized and absolute metrics.

    Utility: Calculates error metrics for a single series.

    Arguments:
    - y_true: numpy array, true values.
    - y_pred: numpy array, predictions.

    Returns:
    - tuple: (nrmse, nmae, nmbe, r2, rmse, mae, mbe)
    """
    from sklearn.metrics import r2_score
    import numpy as np

    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    mbe = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    mean_true = np.mean(y_true)
    nmae = mae / mean_true if mean_true != 0 else np.nan
    nmbe = mbe / mean_true if mean_true != 0 else np.nan
    nrmse = rmse / mean_true if mean_true != 0 else np.nan

    return nrmse, nmae, nmbe, r2, rmse, mae, mbe