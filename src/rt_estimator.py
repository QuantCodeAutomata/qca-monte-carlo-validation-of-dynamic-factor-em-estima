"""
Rubin-Thayer (RT) EM Estimator for Static Factor Analysis.

Implements the EM algorithm for maximum-likelihood estimation of the static
factor model under missing data:
  y_t = Lambda f_t + epsilon_t,  epsilon_t ~ N(0, R)
  f_t ~ N(0, I_r)  (static, no dynamics)

This is the standard factor analysis EM algorithm extended to handle
missing data patterns.

Reference: Rubin & Thayer (1982), "EM algorithms for ML factor analysis"

# Custom — Context7 found no library equivalent for RT EM with arbitrary
# missing data patterns as used in the paper's Monte Carlo comparison.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA


def em_rt(
    Y: np.ndarray,
    mask: np.ndarray,
    r: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """
    Rubin-Thayer EM for static factor analysis with missing data.

    Model: y_t = Lambda f_t + epsilon_t
           f_t ~ N(0, I_r)
           epsilon_t ~ N(0, R), R diagonal

    Parameters
    ----------
    Y : (T, n) observed data matrix (NaN for missing)
    mask : (T, n) boolean array, True = observed
    r : int, number of factors
    max_iter : int, maximum EM iterations
    tol : float, convergence tolerance
    verbose : bool

    Returns
    -------
    dict with:
        Lambda : (n, r) loading matrix
        R : (n, n) diagonal idiosyncratic covariance
        F_hat : (T, r) estimated factors (posterior means)
        log_lik_history : list
        converged : bool
    """
    T, n = Y.shape

    # Initialize via PCA on filled data
    rng = np.random.default_rng(42)
    Y_filled = Y.copy()
    Y_filled[~mask] = rng.standard_normal(np.sum(~mask))

    pca = PCA(n_components=r)
    F_hat = pca.fit_transform(Y_filled)  # (T, r)
    Lambda = pca.components_.T           # (n, r)

    # Normalize
    scale = np.sqrt(np.diag(F_hat.T @ F_hat / T))
    scale = np.where(scale < 1e-10, 1.0, scale)
    F_hat = F_hat / scale[None, :]
    Lambda = Lambda * scale[None, :]

    # Initialize R from residuals
    Resid = Y_filled - F_hat @ Lambda.T
    R_diag = np.maximum(np.var(Resid, axis=0), 1e-6)

    log_lik_history = []
    converged = False

    for iteration in range(max_iter):
        # E-step: compute E[f_t | y_t^obs] and E[f_t f_t' | y_t^obs]
        # For static factor model:
        # Sigma_y = Lambda Lambda' + R
        # E[f_t | y_t^obs] = Lambda_obs' (Lambda_obs Lambda_obs' + R_obs)^{-1} y_t^obs
        # Var[f_t | y_t^obs] = I - Lambda_obs' (Lambda_obs Lambda_obs' + R_obs)^{-1} Lambda_obs

        Ef = np.zeros((T, r))
        Eff = np.zeros((T, r, r))
        log_lik = 0.0

        for t in range(T):
            obs_idx = np.where(mask[t])[0]
            if len(obs_idx) == 0:
                Ef[t] = 0.0
                Eff[t] = np.eye(r)
                continue

            Lambda_obs = Lambda[obs_idx, :]   # (n_obs, r)
            R_obs = np.diag(R_diag[obs_idx])  # (n_obs, n_obs)
            y_obs = Y[t, obs_idx]             # (n_obs,)

            # Sigma_obs = Lambda_obs Lambda_obs' + R_obs
            Sigma_obs = Lambda_obs @ Lambda_obs.T + R_obs
            Sigma_obs = 0.5 * (Sigma_obs + Sigma_obs.T)

            try:
                # Posterior mean: E[f_t | y_obs] = Lambda_obs' Sigma_obs^{-1} y_obs
                Sigma_inv_y = np.linalg.solve(Sigma_obs, y_obs)
                Sigma_inv_L = np.linalg.solve(Sigma_obs, Lambda_obs)
                Ef[t] = Lambda_obs.T @ Sigma_inv_y
                # Posterior variance: I - Lambda_obs' Sigma_obs^{-1} Lambda_obs
                Var_f = np.eye(r) - Lambda_obs.T @ Sigma_inv_L
                Var_f = 0.5 * (Var_f + Var_f.T)
                Eff[t] = Var_f + np.outer(Ef[t], Ef[t])

                # Log-likelihood contribution
                sign, logdet = np.linalg.slogdet(Sigma_obs)
                if sign > 0:
                    n_obs = len(obs_idx)
                    log_lik += -0.5 * (
                        n_obs * np.log(2 * np.pi)
                        + logdet
                        + y_obs @ Sigma_inv_y
                    )
            except np.linalg.LinAlgError:
                Ef[t] = 0.0
                Eff[t] = np.eye(r)

        log_lik_history.append(log_lik)

        # Check convergence
        if iteration > 0:
            ll_prev = log_lik_history[-2]
            ll_curr = log_lik_history[-1]
            denom = (abs(ll_curr) + abs(ll_prev)) / 2.0
            if denom > 1e-10:
                c_j = (ll_curr - ll_prev) / denom
            else:
                c_j = 0.0
            if verbose:
                print(f"  RT Iter {iteration}: log_lik={ll_curr:.4f}, c_j={c_j:.6f}")
            if abs(c_j) < tol:
                converged = True
                break

        # M-step: update Lambda and R
        # Lambda[i,:] = [sum_{t:obs_i} Eff_t]^{-1} [sum_{t:obs_i} y_{i,t} Ef_t]
        Lambda_new = np.zeros((n, r))
        for i in range(n):
            obs_t = mask[:, i]
            if obs_t.sum() < r + 1:
                Lambda_new[i] = 0.0
                continue
            sum_Eff = np.sum(Eff[obs_t], axis=0)
            sum_yEf = np.sum(Y[obs_t, i:i+1] * Ef[obs_t], axis=0)
            try:
                Lambda_new[i] = np.linalg.solve(sum_Eff, sum_yEf)
            except np.linalg.LinAlgError:
                Lambda_new[i] = np.linalg.lstsq(sum_Eff, sum_yEf, rcond=None)[0]

        # R update
        R_diag_new = np.zeros(n)
        for i in range(n):
            obs_t = mask[:, i]
            T_i = obs_t.sum()
            if T_i < 1:
                R_diag_new[i] = 1.0
                continue
            y_i = Y[obs_t, i]
            Ef_i = Ef[obs_t]
            Eff_i = Eff[obs_t]
            lam_i = Lambda_new[i]

            term1 = np.sum(y_i ** 2)
            term2 = -2.0 * np.sum(y_i * (Ef_i @ lam_i))
            term3 = np.sum(np.einsum('tj,j->t', np.einsum('tij,j->ti', Eff_i, lam_i), lam_i))
            R_diag_new[i] = max((term1 + term2 + term3) / T_i, 1e-6)

        Lambda = Lambda_new
        R_diag = R_diag_new

    return {
        "Lambda": Lambda,
        "R": np.diag(R_diag),
        "F_hat": Ef,
        "log_lik_history": log_lik_history,
        "n_iter": iteration + 1,
        "converged": converged,
    }
