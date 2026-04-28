"""
Evaluation metrics for Monte Carlo experiments.

Implements:
- Trace R^2 for factor space recovery
- Relative forecast accuracy for ragged-edge experiments
- Mean absolute error for AR(1) coefficient estimation

# Custom — Context7 found no library equivalent for these specific metrics
# as defined in the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def trace_r2(F_true: np.ndarray, F_hat: np.ndarray) -> float:
    """
    Compute trace R^2 for factor space recovery.

    Measures how well the estimated factor space spans the true factor space,
    invariant to rotation and sign changes.

    trace R^2 = Trace(F' Fhat (Fhat'Fhat)^{-1} Fhat' F) / Trace(F'F)

    Parameters
    ----------
    F_true : (T, r) true factor matrix
    F_hat : (T, rhat) estimated factor matrix

    Returns
    -------
    float in [0, 1], higher is better
    """
    T = F_true.shape[0]
    if T == 0:
        return 0.0

    # Numerator: Trace(F' Fhat (Fhat'Fhat)^{-1} Fhat' F)
    # = Trace(F' P_Fhat F) where P_Fhat is the projection onto col(Fhat)
    FtF = F_true.T @ F_true  # (r, r)
    denom = np.trace(FtF)
    if denom < 1e-10:
        return 0.0

    # Projection matrix P_Fhat = Fhat (Fhat'Fhat)^{-1} Fhat'
    # Trace(F' P_Fhat F) = Trace(F' Fhat (Fhat'Fhat)^{-1} Fhat' F)
    FhFh = F_hat.T @ F_hat  # (rhat, rhat)
    try:
        # Use lstsq for numerical stability
        # Trace = sum_j ||P_Fhat F[:,j]||^2 = sum_j F[:,j]' P_Fhat F[:,j]
        # = ||Fhat^+ F||_F^2 where Fhat^+ is pseudo-inverse
        # Equivalently: solve Fhat'Fhat x = Fhat'F, then Trace = Trace(F'Fhat x)
        FhF = F_hat.T @ F_true  # (rhat, r)
        x = np.linalg.lstsq(FhFh, FhF, rcond=None)[0]  # (rhat, r)
        numer = np.trace(FhF.T @ x)  # = Trace(F'Fhat (Fhat'Fhat)^{-1} Fhat'F)
    except np.linalg.LinAlgError:
        return 0.0

    return float(np.clip(numer / denom, 0.0, 1.0))


def relative_forecast_accuracy(
    chi_true: float,
    chi_hat: float,
    var_chi: float,
) -> float:
    """
    Compute relative forecast accuracy for ragged-edge experiment.

    Metric: 1 - (chi_{1,T} - chihat_{1,T})^2 / Var(chi_{1,t})

    Parameters
    ----------
    chi_true : true common component value
    chi_hat : estimated common component value
    var_chi : variance of the common component over time

    Returns
    -------
    float, can be negative (recorded as '-' in paper)
    """
    if var_chi < 1e-10:
        return 0.0
    return 1.0 - (chi_true - chi_hat) ** 2 / var_chi


def mean_abs_error_ar(alpha_hat: np.ndarray, alpha_true: float) -> float:
    """
    Compute mean absolute error of AR(1) coefficient estimates.

    Parameters
    ----------
    alpha_hat : (n,) estimated AR(1) coefficients
    alpha_true : float, true AR(1) coefficient

    Returns
    -------
    float, mean |alpha_hat_i - alpha_true|
    """
    return float(np.mean(np.abs(alpha_hat - alpha_true)))


def rmsfe(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Root Mean Squared Forecast Error.

    Parameters
    ----------
    forecasts : (N,) array of forecasts
    actuals : (N,) array of actual values

    Returns
    -------
    float, RMSFE
    """
    errors = forecasts - actuals
    return float(np.sqrt(np.mean(errors ** 2)))


def align_factors_ols(F_true: np.ndarray, F_hat: np.ndarray) -> np.ndarray:
    """
    Align estimated factors to true factors via OLS rotation.

    Finds rotation matrix H such that F_hat @ H is closest to F_true.

    Parameters
    ----------
    F_true : (T, r) true factors
    F_hat : (T, rhat) estimated factors

    Returns
    -------
    F_aligned : (T, r) aligned factor estimates
    """
    # OLS: H = (F_hat'F_hat)^{-1} F_hat' F_true
    try:
        H = np.linalg.lstsq(F_hat, F_true, rcond=None)[0]  # (rhat, r)
        return F_hat @ H
    except np.linalg.LinAlgError:
        return F_hat[:, :F_true.shape[1]]
