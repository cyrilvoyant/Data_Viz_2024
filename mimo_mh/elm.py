# ==============================================================================
# elm.py
# ==============================================================================
# Extreme Learning Machine (ELM) model implementation.
from scipy.linalg import pinv
import numpy as np
import pandas as pd
from pathlib import Path

def train_elm(X_train, Y_train, X_test, Y_test,
              num_hidden=1000, num_initializations=1, lambda_reg=1e-6):
    """
    Train an ELM (Extreme Learning Machine) with Ridge-regularized linear readout.

    Utility: Initializes random hidden layer, computes readout weights using closed-form ridge regression, and selects best initialization based on test RMSE.

    Arguments:
    - X_train: numpy array, training inputs.
    - Y_train: numpy array, training targets.
    - X_test: numpy array, test inputs (for validation).
    - Y_test: numpy array, test targets (for validation).
    - num_hidden: int, number of hidden neurons.
    - num_initializations: int, number of random restarts.
    - lambda_reg: float, ridge regularization parameter.

    Returns:
    - best_model: tuple (W, b, beta), best weights, biases, readout.
    """
    best_rmse = float('inf')
    best_model = None
    for _ in range(num_initializations):
        # 1. Initialize random input weights (W) and biases (b)
        W = np.random.rand(num_hidden, X_train.shape[1])
        b = np.random.rand(num_hidden, 1)
        
        # 2. Calculate the hidden layer output matrix (H)
        H = np.maximum(0, X_train @ W.T + b.T) # ReLU activation
        
        # 3. Ridge-regularized closed-form solution for beta (output weights).
        HTH = H.T @ H               # (m, m)
        HTY = H.T @ Y_train         # (m, n_outputs or flattened dim)
        try:
            # More stable solution using solve
            beta = np.linalg.solve(HTH + lambda_reg * np.eye(HTH.shape[0]), HTY)
        except np.linalg.LinAlgError:
            # Fallback to Moore-Penrose pseudo-inverse if solve fails
            beta = pinv(H) @ Y_train
            
        # Evaluate this initialization on the test set
        H_test = np.maximum(0, X_test @ W.T + b.T)
        Y_pred = np.maximum(H_test @ beta, 0) # Ensure non-negative predictions
        rmse = np.mean(np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=0)))
        
        # Keep the best model
        if rmse < best_rmse:
            best_model = (W, b, beta)
            best_rmse = rmse
            
    return best_model

def predict_elm(X, model):
    """
    Forward pass for a trained ELM.

    Utility: Computes predictions using trained weights and ReLU activation.

    Arguments:
    - X: numpy array, input features.
    - model: tuple (W, b, beta), trained parameters.

    Returns:
    - Y_pred: numpy array, predicted targets.
    """
    W, b, beta = model
    H = np.maximum(0, X @ W.T + b.T)
    Y_pred = np.maximum(H @ beta, 0)
    return Y_pred