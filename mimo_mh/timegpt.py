# ==============================================================================
# timegpt.py
# ==============================================================================
# Placeholder for loading TimeGPT results.
import numpy as np
import pandas as pd
import os

def load_timegpt_metrics():
    """
    Load pre-computed TimeGPT metrics from a CSV file and ensure English column names.

    Utility: This function loads a single metrics file for TimeGPT results. If the file is missing, it warns and returns None.
    Also translates French variable names to English to match the rest of the pipeline.

    Returns:
    - metrics_df: pandas DataFrame containing the metrics with English variable names, or None if file not found.
    """
    file_path = "metrics_with_tgpt_8000S.csv"
    if os.path.exists(file_path):
        metrics_df = pd.read_csv(file_path)
        
        # Translation mapping for TimeGPT variable names (French to English)
        variable_translation = {
            'production_totale_mw': 'total_production_mw',
            'thermique_mw': 'thermal_mw', 
            'hydraulique_mw': 'hydraulic_mw',
            'micro-hydraulique_mw': 'micro_hydraulic_mw',
            'solaire_photovoltaique_mw': 'solar_photovoltaic_mw',
            'eolien_mw': 'wind_mw',
            'bioenergies_mw': 'bioenergy_mw',
            'importations_mw': 'imports_mw'
        }
        
        # Apply translation to the 'output' column if it exists
        if 'output' in metrics_df.columns:
            metrics_df['output'] = metrics_df['output'].replace(variable_translation)
            print("    - TimeGPT variable names translated to English")
        
        # Also check if there are other columns that need translation
        metrics_df = metrics_df.rename(columns=variable_translation)
        
    else:
        print(f"      - Warning: Metrics file not found at {file_path}")
        metrics_df = None
    return metrics_df

