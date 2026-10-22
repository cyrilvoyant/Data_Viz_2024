# ==============================================================================
# prophet.py
# ==============================================================================
# Functions to load and process Prophet model results.
import pandas as pd
from pathlib import Path

def load_prophet_forecasts(targets):
    """
    Loads pre-computed Prophet forecast files.

    Utility: Retrieves forecast CSVs for each target from a directory. Warns if files are missing.

    Arguments:
    - targets: list of str, target variable names in English.

    Returns:
    - forecast_dfs: dict of {target: DataFrame}, forecasts for each target with English column names.
    """
    print("    - Loading pre-computed Prophet forecasts...")
    # This assumes forecast files are stored in a specific directory
    base_path = Path("neuralprophet_results/forecasts/")
    forecast_dfs = {}
    
    # Mapping from English to French for file names (Prophet files might be in French)
    english_to_french = {
        'total_production_mw': 'production_totale_mw',
        'thermal_mw': 'thermique_mw',
        'hydraulic_mw': 'hydraulique_mw',
        'micro_hydraulic_mw': 'micro-hydraulique_mw',
        'solar_photovoltaic_mw': 'solaire_photovoltaique_mw',
        'wind_mw': 'eolien_mw',
        'bioenergy_mw': 'bioenergies_mw',
        'imports_mw': 'importations_mw'
    }
    
    # Reverse mapping for column renaming
    french_to_english = {v: k for k, v in english_to_french.items()}
    
    for target in targets:
        # Try with English name first, then French name
        file_path_eng = base_path / f"{target}_forecast_last2y.csv"
        file_path_fr = base_path / f"{english_to_french.get(target, target)}_forecast_last2y.csv"
        
        df = None
        if file_path_eng.exists():
            df = pd.read_csv(file_path_eng)
        elif file_path_fr.exists():
            df = pd.read_csv(file_path_fr)
        else:
            print(f"      - Warning: Prophet forecast file not found for '{target}' at {file_path_eng} or {file_path_fr}")
            continue
            
        # Ensure column names are in English
        if df is not None:
            # Rename columns from French to English if needed
            df = df.rename(columns=french_to_english)
            
            # Standardize any remaining column names
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
            
            forecast_dfs[target] = df
            
    return forecast_dfs