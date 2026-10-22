# ==============================================================================
# models.py
# ==============================================================================
# Functions to run different forecasting strategies (SISO, MIMO, MIMO-MH).

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict

# --- Import local modules
from utils import prepare_data_mimo_mh
from elm import train_elm, predict_elm
from reconciliation import create_S_from_outputs, estimate_W_from_train_residuals, reconcile_all_horizons
from metrics import get_metrics

def run_mimo_mh_all_horizons(
    df_all,
    split_date,
    output_names,
    max_horizon,
    window_size,
    elm_params, # Added to pass ELM hyperparameters
    reconcile=False
):
    """
    Train a true MIMO-MH model jointly predicting all horizons,
    and optionally apply MinT-Diagonal (WLS) hierarchical reconciliation.

    Utility: Runs the MIMO-MH model, trains ELM, predicts, and optionally reconciles.

    Arguments:
    - df_all: DataFrame, full dataset with 'ds' and outputs.
    - split_date: str, date to split train/test.
    - output_names: list of str, output variables.
    - max_horizon: int, max forecast horizon.
    - window_size: int, input window size.
    - elm_params: dict, ELM hyperparameters (n_hidden, lambda_reg, n_init).
    - reconcile: bool, whether to apply reconciliation.

    Returns:
    - df_mh: DataFrame with datetime, horizon, output, y_true, y_pred (and y_pred_rec if reconciled).
    """
    model_name = "MIMO-MH-REC" if reconcile else "MIMO-MH"
    print(f"    - Running {model_name} model...")
    # ---------- Split train/test
    train_df = df_all[df_all['ds'] < split_date].copy()
    test_df  = df_all[df_all['ds'] >= split_date].copy()

    # ---------- Prepare matrices
    feature_cols = [c for c in df_all.columns if c != 'ds']
    input_matrix_train = train_df[feature_cols].fillna(0).values
    input_matrix_test  = test_df[feature_cols].fillna(0).values
    datetime_index_test = test_df['ds'].reset_index(drop=True)

    num_rows_train = input_matrix_train.shape[0]
    num_rows_test  = input_matrix_test.shape[0]
    num_outputs    = len(output_names)
    assert num_outputs == 1 + (create_S_from_outputs(output_names).shape[1]), \
        "output_names must be [total, bottoms...] to match S."

    # ---------- Build MIMO-MH datasets
    X_train, Y_train = prepare_data_mimo_mh(
        input_matrix_train, num_rows_train, max_horizon, window_size, num_outputs
    )
    X_test, Y_test = prepare_data_mimo_mh(
        input_matrix_test, num_rows_test, max_horizon, window_size, num_outputs
    )

    # ---------- Train ELM and predict
    model = train_elm(
        X_train, Y_train, X_test, Y_test,
        num_hidden=elm_params['n_hidden'],
        lambda_reg=elm_params['lambda_reg'],
        num_initializations=elm_params['n_init']
    )

    Y_pred_train = predict_elm(X_train, model)
    Y_pred_test  = predict_elm(X_test,  model)

    # ---------- Reshape to (samples, horizons, outputs)
    Y_train_mh = Y_train.reshape(-1, max_horizon, num_outputs)
    Y_test_mh  = Y_test.reshape(-1,  max_horizon, num_outputs)
    Y_pred_mh  = Y_pred_test.reshape(-1, max_horizon, num_outputs)

    # ---------- Optional reconciliation (MinT-Diagonal / WLS)
    Y_pred_reconciled = None
    if reconcile:
        print("        - Applying reconciliation...")
        S = create_S_from_outputs(output_names)
        Y_pred_train_mh = Y_pred_train.reshape(-1, max_horizon, num_outputs)
        W_list = estimate_W_from_train_residuals(Y_train_mh, Y_pred_train_mh, diagonal=True)
        Y_pred_reconciled = reconcile_all_horizons(Y_pred_mh, S, W_list)

        # Diagnostic: coherence improvement
        total_idx = 0
        raw_incoherence = np.mean(np.abs(Y_pred_mh[:, :, total_idx] - Y_pred_mh[:, :, 1:].sum(axis=2)))
        rec_incoherence = np.mean(np.abs(Y_pred_reconciled[:, :, total_idx] - Y_pred_reconciled[:, :, 1:].sum(axis=2)))
        print(f"        - Coherence Error |Total - sum(children)|: Raw={raw_incoherence:.4f} -> Reconciled={rec_incoherence:.4f}")

    # ---------- Build tidy result dataframe
    min_samples = Y_test_mh.shape[0]
    records = []
    for h in range(max_horizon):
        dt_h = datetime_index_test[window_size + h : window_size + h + min_samples]
        for sample_idx in range(min_samples):
            for o_idx, o_name in enumerate(output_names):
                row = {
                    'datetime': dt_h.iloc[sample_idx],
                    'horizon': h + 1,
                    'output': o_name,
                    'y_true': Y_test_mh[sample_idx, h, o_idx],
                    'y_pred': Y_pred_mh[sample_idx, h, o_idx]
                }
                if Y_pred_reconciled is not None:
                    row["y_pred_rec"] = float(Y_pred_reconciled[sample_idx, h, o_idx])
                records.append(row)
                
    print(f"    - {model_name} model finished.")
    df_mh = pd.DataFrame(records)
    return df_mh

def run_mimo_all_horizons(df_all, split_date, output_names, max_horizon, window_size, elm_params, reconcile=False):
    """
    Run MIMO model (one per horizon) and apply reconciliation as a post-processing step.

    Utility: Trains separate MIMO models per horizon, predicts, and optionally reconciles.

    Arguments: (same as run_mimo_mh_all_horizons)

    Returns:
    - pd.DataFrame with datetime, horizon, output, y_true, y_pred (and y_pred_rec if reconciled).
    """
    model_name = "MIMO-REC" if reconcile else "MIMO"
    print(f"    - Running {model_name} model...")
    input_matrix = df_all[output_names].values
    num_rows, num_outputs = len(df_all), len(output_names)
    datetime_index = pd.to_datetime(df_all['ds'])
    
    # --- Storage for assembling 3D arrays ---
    Y_true_train_list, Y_pred_train_list = [], []
    Y_true_test_list,  Y_pred_test_list  = [], []
    test_datetimes_list = []

    for h in range(1, max_horizon + 1):
        samples = num_rows - window_size - h + 1
        X_windows = np.array([input_matrix[t : t + window_size].flatten() for t in range(samples)])
        Y_windows = np.array([input_matrix[t + window_size + h - 1, :num_outputs] for t in range(samples)])
        
        sample_datetimes = datetime_index[window_size + h - 1 : window_size + h - 1 + samples]
        train_mask = sample_datetimes < pd.to_datetime(split_date)
        test_mask = ~train_mask
        
        X_train, Y_train = X_windows[train_mask], Y_windows[train_mask]
        X_test, Y_test = X_windows[test_mask], Y_windows[test_mask]
        
        model = train_elm(
            X_train, Y_train, X_test, Y_test, 
            num_hidden=elm_params['n_hidden'], 
            lambda_reg=elm_params['lambda_reg'],
            num_initializations=elm_params['n_init']
        )
        
        # Store predictions for both sets
        Y_pred_train_list.append(predict_elm(X_train, model))
        Y_pred_test_list.append(predict_elm(X_test, model))
        Y_true_train_list.append(Y_train)
        Y_true_test_list.append(Y_test)
        test_datetimes_list.append(sample_datetimes[test_mask])

    # --- Assemble forecasts into (samples, horizons, outputs) arrays ---
    # Find common number of samples to align arrays
    min_train_samples = min(len(y) for y in Y_true_train_list)
    min_test_samples = min(len(y) for y in Y_true_test_list)

    Y_true_train_mh = np.stack([y[:min_train_samples] for y in Y_true_train_list], axis=1)
    Y_pred_train_mh = np.stack([y[:min_train_samples] for y in Y_pred_train_list], axis=1)
    Y_true_test_mh  = np.stack([y[:min_test_samples] for y in Y_true_test_list], axis=1)
    Y_pred_test_mh  = np.stack([y[:min_test_samples] for y in Y_pred_test_list], axis=1)

    # --- Optional Reconciliation ---
    Y_pred_reconciled = None
    if reconcile:
        print("        - Applying reconciliation...")
        S = create_S_from_outputs(output_names)
        W_list = estimate_W_from_train_residuals(Y_true_train_mh, Y_pred_train_mh, diagonal=True)
        Y_pred_reconciled = reconcile_all_horizons(Y_pred_test_mh, S, W_list)

    # --- Build tidy DataFrame ---
    records = []
    for h_idx, h in enumerate(range(1, max_horizon + 1)):
        dt_h = test_datetimes_list[h_idx][:min_test_samples]
        for sample_idx in range(min_test_samples):
            for o_idx, o_name in enumerate(output_names):
                row = {'datetime': dt_h.iloc[sample_idx], 'horizon': h, 'output': o_name,
                       'y_true': Y_true_test_mh[sample_idx, h_idx, o_idx],
                       'y_pred': Y_pred_test_mh[sample_idx, h_idx, o_idx]}
                if Y_pred_reconciled is not None:
                    row["y_pred_rec"] = Y_pred_reconciled[sample_idx, h_idx, o_idx]
                records.append(row)

    print(f"    - {model_name} model finished.")
    return pd.DataFrame(records)

def run_siso_all_horizons(df_all, split_date, output_names, max_horizon, window_size, elm_params, reconcile=False):
    """
    Run SISO model (one per var/horizon) and apply reconciliation as a post-processing step.

    Utility: Trains separate SISO models per variable and horizon, predicts, and optionally reconciles.

    Arguments: (same as run_mimo_mh_all_horizons)

    Returns:
    - pd.DataFrame with datetime, horizon, output, y_true, y_pred (and y_pred_rec if reconciled).
    """
    model_name = "SISO-REC" if reconcile else "SISO"
    print(f"    - Running {model_name} model...")
    datetime_index = pd.to_datetime(df_all["ds"])
    T, H, W = len(df_all), int(max_horizon), int(window_size)
    
    # --- Storage for all predictions before assembling ---
    preds = defaultdict(lambda: defaultdict(dict)) # preds[set][horizon][output]

    for out_idx, out_name in enumerate(output_names):
        y = df_all[out_name].to_numpy()
        S = T - W - H + 1
        if S <= 0: continue
        X_all_windows = sliding_window_view(y, window_shape=W)[:S]
        
        for h in range(1, H + 1):
            Y_h = y[W + h - 1 : S + W + h - 1]
            datetimes_h = datetime_index[W + h - 1 : S + W + h - 1]
            train_mask = datetimes_h < pd.to_datetime(split_date)
            test_mask = ~train_mask
            
            if not np.any(train_mask) or not np.any(test_mask): continue

            X_train, Y_train = X_all_windows[train_mask], Y_h[train_mask].reshape(-1, 1)
            X_test, Y_test = X_all_windows[test_mask], Y_h[test_mask].reshape(-1, 1)
            
            model = train_elm(
                X_train, Y_train, X_test, Y_test, 
                num_hidden=elm_params['n_hidden'], 
                lambda_reg=elm_params['lambda_reg'],
                num_initializations=elm_params['n_init']
            )
            
            # Store all predictions
            preds['train'][h][out_name] = {'true': Y_train, 'pred': predict_elm(X_train, model)}
            preds['test'][h][out_name]  = {'true': Y_test,  'pred': predict_elm(X_test, model), 'dates': datetimes_h[test_mask]}

    # --- Assemble forecasts into (samples, horizons, outputs) arrays ---
    min_train_samples = min(len(d['true']) for h in range(1, H + 1) for o, d in preds['train'][h].items())
    min_test_samples = min(len(d['true']) for h in range(1, H + 1) for o, d in preds['test'][h].items())

    def assemble_array(dset, key):
        tensors_by_h = []
        for h in range(1, H + 1):
            tensors_by_o = [preds[dset][h][o][key][:min_train_samples if dset=='train' else min_test_samples] for o in output_names]
            tensors_by_h.append(np.hstack(tensors_by_o))
        return np.stack(tensors_by_h, axis=1)

    Y_true_train_mh = assemble_array('train', 'true')
    Y_pred_train_mh = assemble_array('train', 'pred')
    Y_true_test_mh  = assemble_array('test', 'true')
    Y_pred_test_mh  = assemble_array('test', 'pred')

    # --- Optional Reconciliation ---
    Y_pred_reconciled = None
    if reconcile:
        print("        - Applying reconciliation...")
        S_matrix = create_S_from_outputs(output_names)
        W_list = estimate_W_from_train_residuals(Y_true_train_mh, Y_pred_train_mh, diagonal=True)
        Y_pred_reconciled = reconcile_all_horizons(Y_pred_test_mh, S_matrix, W_list)

    # --- Build tidy DataFrame ---
    records = []
    for h_idx, h in enumerate(range(1, H + 1)):
        dt_h = preds['test'][h][output_names[0]]['dates'][:min_test_samples]
        for sample_idx in range(min_test_samples):
            for o_idx, o_name in enumerate(output_names):
                row = {'datetime': dt_h.iloc[sample_idx], 'horizon': h, 'output': o_name,
                       'y_true': Y_true_test_mh[sample_idx, h_idx, o_idx],
                       'y_pred': Y_pred_test_mh[sample_idx, h_idx, o_idx]}
                if Y_pred_reconciled is not None:
                    row["y_pred_rec"] = Y_pred_reconciled[sample_idx, h_idx, o_idx]
                records.append(row)
    
    print(f"    - {model_name} model finished.")
    return pd.DataFrame(records)


def run_persistence_models(df_all, split_date, output_names, window_size, max_horizon):
    """
    Calculates metrics for two persistence models:
      - 'Persistence' : takes the value at horizon h
      - 'Persistence-24h' : takes the value 24h back, regardless of h

    Utility: Computes baseline persistence forecasts for benchmarking.

    Arguments:
    - df_all: DataFrame, full dataset.
    - split_date: str, split date.
    - output_names: list of str, outputs.
    - window_size: int, window size.
    - max_horizon: int, max horizon.

    Returns:
    - results: list of lists, each with model, horizon, output, and metrics.
    """
    results = []
    print(f"\n{'='*60}")
    print("Calculating persistence performances (horizon and 24h)…")

    test_df = df_all[df_all['ds'] >= split_date].copy()
    input_matrix = test_df[output_names].values
    num_rows = len(test_df)

    for horizon in range(1, max_horizon + 1):
        print(f" → Persistence horizon {horizon}h")
        num_obs = num_rows - window_size - horizon
        if num_obs <= 0:
            continue

        Y_test = np.zeros((num_obs, len(output_names)))
        for i in range(num_obs):
            Y_test[i, :] = input_matrix[i + window_size + horizon - 1, :]

        Y_test = np.nan_to_num(Y_test)

        Y_pers = np.zeros_like(Y_test)
        Y_pers_24h = np.zeros_like(Y_test)
        for i in range(num_obs):
            if i - horizon >= 0:
                Y_pers[i] = Y_test[i - horizon]
            if i - 24 >= 0:
                Y_pers_24h[i] = Y_test[i - 24]

        for j in range(len(output_names)):
            y_true = Y_test[:, j]
            y_hor = Y_pers[:, j]
            y_24h = Y_pers_24h[:, j]

            nrmse1, nmae1, nmbe1, r21, rmse1, mae1, mbe1 = get_metrics(y_true, y_hor)
            nrmse2, nmae2, nmbe2, r22, rmse2, mae2, mbe2 = get_metrics(y_true, y_24h)

            results.append(['Persistence_h', horizon, output_names[j],
                            float(nrmse1), float(nmae1), float(nmbe1), float(r21),
                            float(rmse1), float(mae1), float(mbe1)])
            results.append(['Persistence_24h', horizon, output_names[j],
                            float(nrmse2), float(nmae2), float(nmbe2), float(r22),
                            float(rmse2), float(mae2), float(mbe2)])

    return results