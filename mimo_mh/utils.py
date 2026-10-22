# ==============================================================================
# utils.py
# ==============================================================================
# Utility functions for data loading and preprocessing.
import unicodedata
import re
import pandas as pd
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

def slugify(col: str) -> str:
    """
    Converts a string to a slug-like format (lowercase, underscores).

    Utility: Normalizes column names by removing accents and special characters.

    Arguments:
    - col: str, column name to normalize.

    Returns:
    - str: normalized column name.
    """
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = re.sub(r"[^\w\s-]", "", col).strip().lower().replace(" ", "_")
    return col

def load_and_preprocess_data(
    file_path: Path,
    date_col: str = "Date",
    drop_policy: str = "mean", # "first", "last" or "mean"
) -> pd.DataFrame:
    """
    Load a CSV, normalize column names, create 'ds' from `date_col`,
    and deduplicate on 'ds' according to `drop_policy`.

    Utility: Prepares the dataset for modeling by handling dates, duplicates, and renaming columns to English.

    Arguments:
    - file_path: Path, path to CSV file.
    - date_col: str, name of date column (default: "Date").
    - drop_policy: str, how to handle duplicates ("first", "last", "mean").

    Returns:
    - df: pandas DataFrame with unique, sorted 'ds' column and English-renamed columns.
    """
    print(">>> Loading and pre-processing data...")
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    # 1) normalize column names
    df.columns = [slugify(c) for c in df.columns]
    date_col = slugify(date_col)
    
    # 2) rename specific columns to full English names
    column_renames = {
        'production_totale_mw': 'total_production_mw',
        'thermique_mw': 'thermal_mw',
        'hydraulique_mw': 'hydraulic_mw',
        'micro-hydraulique_mw': 'micro_hydraulic_mw',
        'solaire_photovoltaique_mw': 'solar_photovoltaic_mw',
        'eolien_mw': 'wind_mw',
        'bioenergies_mw': 'bioenergy_mw',
        'importations_mw': 'imports_mw'
    }
    df = df.rename(columns=column_renames)
    
    # 3) convert date → UTC
    df["ds"] = pd.to_datetime(df[date_col], utc=True)
    df = df.drop(columns=[date_col])
    
    # 4) handle duplicates on ds
    if drop_policy in ("first", "last"):
        before = len(df)
        df = df.drop_duplicates(subset="ds", keep=drop_policy)
        removed = before - len(df)
        if removed:
            print(f"    - {removed} duplicate(s) removed (keep='{drop_policy}').")
    elif drop_policy == "mean":
        df = df.groupby("ds", as_index=False).mean(numeric_only=True)
        print("    - Duplicates aggregated by taking the mean.")
    else:
        raise ValueError("drop_policy must be 'first', 'last' or 'mean'.")
    # 5) sort and reset index
    df = df.sort_values("ds").reset_index(drop=True)
    # 6) drop timezone if present
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None) # Make timezone-naive for compatibility
        
    print(f"    - Data loaded successfully: {len(df)} rows, {df.shape[1]} columns.")
    print(">>> Data loading complete.\n")
    return df

def prepare_data_mimo_mh(input_matrix, num_rows, max_horizon, window_size, num_outputs):
    """
    Build supervised learning matrices (X, Y) for MIMO multi-horizon training.

    Utility: Creates input and target arrays for joint multi-output multi-horizon forecasting using sliding windows.

    Arguments:
    - input_matrix: numpy array, full time series matrix.
    - num_rows: int, number of rows in input_matrix.
    - max_horizon: int, maximum forecast horizon.
    - window_size: int, input window length.
    - num_outputs: int, number of output variables.

    Returns:
    - X_windows: numpy array (samples, flattened features).
    - Y_windows: numpy array (samples, flattened targets).
    """
    samples = num_rows - window_size - max_horizon + 1
    # Input windows: [sample, flattened_window_features]
    X_windows = np.array([
        input_matrix[t : t + window_size].flatten()
        for t in range(samples)
    ])
    # Build Y as a concatenation of the future target block:
    # [ (t+1..t+H) × num_outputs ] flattened for each sample.
    Y_windows = np.array([
        input_matrix[t + window_size : t + window_size + max_horizon, :num_outputs].flatten()
        for t in range(samples)
    ])
    return X_windows, Y_windows