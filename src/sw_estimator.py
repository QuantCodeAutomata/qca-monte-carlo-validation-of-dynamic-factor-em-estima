"""
Stock-Watson (SW) Missing-Data EM Principal Components Estimator.

Implements the iterative principal components approach for handling missing
data in large factor models, as described in:
  Stock & Watson (2002), "Forecasting Using Principal Components from a
  Large Number of Predictors"

The algorithm iterates between:
  1. Estimating factors by PCA on the current filled dataset
  2. Filling missing values with fitted values from the factor model

# Custom — Context7 found no library equivalent for SW missing-data EM PC
# as used in the paper's Monte Carlo comparison.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA


def em_sw(
    Y: np.ndarray,
    mask: np.ndarray,
    r: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """
    Stock-Watson missing-data EM principal components.

    Iterates between PCA estimation and missing-value imputation.

    Parameters
    ----------
    Y : (T, n) observed data matrix (NaN for missing)
    mask : (T, n) boolean array, True = observed
    r : int, number of factors
    max_iter : int, maximum iterations
    tol : float, convergence tolerance (relative change in factor estimates)
    verbose : bool

    Returns
    -------
    dict with:
        Lambda : (n, r) loading matrix
        F_hat : (T, r) estimated factors
        Y_filled : (T, n) data with missing values imputed
        n_iter : int
        converged : bool
    """
    T, n = Y.shape

    # Initialize: fill missing with column means (or 0 if standardized)
    Y_filled = Y.copy().astype(float)
    col_means = np.nanmean(Y_filled, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    for i in range(n):
        missing_t = ~mask[:, i]
        Y_filled[missing_t, i] = col_means[i]

    F_prev = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # PCA step
        pca = PCA(n_components=r)
        F_hat = pca.fit_transform(Y_filled)  # (T, r)
        Lambda = pca.components_.T           # (n, r)

        # Normalize factors
        scale = np.sqrt(np.diag(F_hat.T @ F_hat / T))
        scale = np.where(scale < 1e-10, 1.0, scale)
        F_hat = F_hat / scale[None, :]
        Lambda = Lambda * scale[None, :]

        # Imputation step: fill missing with fitted values
        Y_hat = F_hat @ Lambda.T  # (T, n)
        Y_new = Y_filled.copy()
        Y_new[~mask] = Y_hat[~mask]

        # Check convergence: relative change in factor estimates
        if F_prev is not None:
            # Align factors (sign ambiguity)
            F_aligned = _align_factors(F_hat, F_prev)
            change = np.linalg.norm(F_aligned - F_prev) / max(np.linalg.norm(F_prev), 1e-10)
            if verbose:
                print(f"  SW Iter {iteration}: change={change:.6f}")
            if change < tol:
                converged = True
                n_iter = iteration + 1
                Y_filled = Y_new
                break

        F_prev = F_hat.copy()
        Y_filled = Y_new
        n_iter = iteration + 1

    return {
        "Lambda": Lambda,
        "F_hat": F_hat,
        "Y_filled": Y_filled,
        "n_iter": n_iter,
        "converged": converged,
    }


def _align_factors(F_new: np.ndarray, F_old: np.ndarray) -> np.ndarray:
    """
    Align factor signs to minimize distance between F_new and F_old.

    Parameters
    ----------
    F_new : (T, r) new factor estimates
    F_old : (T, r) old factor estimates

    Returns
    -------
    F_aligned : (T, r) sign-aligned F_new
    """
    r = F_new.shape[1]
    F_aligned = F_new.copy()
    for j in range(r):
        if np.dot(F_new[:, j], F_old[:, j]) < 0:
            F_aligned[:, j] = -F_new[:, j]
    return F_aligned
