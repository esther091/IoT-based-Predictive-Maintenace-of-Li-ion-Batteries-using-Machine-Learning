"""
utils.py
Helper functions for SOH, RUL computation and evaluation metrics.
Equations reference Chapter 3 of the project report.
"""

import numpy as np


def compute_soh(capacity_actual, capacity_rated=2.0):
    """
    SOH = (Q_actual / Q_rated) x 100
    Equation 3.8
    """
    return (capacity_actual / capacity_rated) * 100


def compute_rul(n_eol, n_current, cycle_duration_days=0.5):
    """
    RUL(n) = n_EOL - n, converted to days
    Equation 3.9
    """
    return max(0, (n_eol - n_current) * cycle_duration_days)


def clip_normalized(x):
    """
    Clips a normalized value to the [0, 1] range
    Equation 3.11
    """
    return max(0.0, min(1.0, x))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)
    Equation 3.19
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100