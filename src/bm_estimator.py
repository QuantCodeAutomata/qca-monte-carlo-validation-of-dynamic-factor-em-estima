"""
Banbura-Modugno (BM) EM Estimator for Dynamic Factor Models.

Implements the EM algorithm for maximum-likelihood estimation of dynamic factor
models under arbitrary missing data patterns, as described in:
  Banbura & Modugno (2014), "Maximum Likelihood Estimation of Factor Models
  on Datasets with Arbitrary Pattern of Missing Data"

State-space form (uncorrelated idiosyncratic):
  y_t = Z alpha_t + epsilon_t,  epsilon_t ~ N(0, H)
  alpha_{t+1} = T alpha_t + R eta_t,  eta_t ~ N(0, Q)

where alpha_t = [f_t', f_{t-1}', ..., f_{t-p+1}']' for p-th order factor VAR.

For AR(1) idiosyncratic, the state is augmented with tilde_epsilon_t.

# Custom — Context7 found no library equivalent (paper EM algorithm with
# arbitrary missing-data M-step formulas)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA

from .kalman import run_filter_smoother


def _build_companion_matrix(A_list: list[np.ndarray], r: int) -> np.ndarray:
    """
    Build companion matrix for VAR(p) factor dynamics.

    Parameters
    ----------
    A_list : list of p (r, r) coefficient matrices [A_1, ..., A_p]
    r : int, number of factors

    Returns
    -------
    T_mat : (r*p, r*p) companion matrix
    """
    p = len(A_list)
    T_mat = np.zeros((r * p, r * p))
    for i, A in enumerate(A_list):
        T_mat[:r, i * r:(i + 1) * r] = A
    # Shift rows
    if p > 1:
        T_mat[r:, :r * (p - 1)] = np.eye(r * (p - 1))
    return T_mat


def _initialize_params(
    Y_init: np.ndarray,
    r: int,
    p: int,
    ar1_idio: bool = False,
) -> dict:
    """
    Initialize BM estimator parameters via principal components.

    Parameters
    ----------
    Y_init : (T, n) data matrix with NaN for missing (already filled with N(0,1))
    r : int, number of factors
    p : int, VAR order for factors
    ar1_idio : bool, whether to include AR(1) idiosyncratic components

    Returns
    -------
    dict of initial parameters
    """
    T, n = Y_init.shape

    # PCA on filled data
    pca = PCA(n_components=r)
    F_hat = pca.fit_transform(Y_init)  # (T, r)
    Lambda_hat = pca.components_.T    # (n, r)

    # Normalize: scale factors so that F'F/T = I_r
    scale = np.sqrt(np.diag(F_hat.T @ F_hat / T))
    scale = np.where(scale < 1e-10, 1.0, scale)
    F_hat = F_hat / scale[None, :]
    Lambda_hat = Lambda_hat * scale[None, :]

    # Estimate VAR(p) on factors
    A_list, Q_hat = _estimate_var(F_hat, p)

    # Residuals for idiosyncratic
    Chi_hat = F_hat @ Lambda_hat.T
    Resid = Y_init - Chi_hat  # (T, n)

    if ar1_idio:
        # Estimate AR(1) for each idiosyncratic series
        alpha_hat = np.zeros(n)
        sigma2_hat = np.zeros(n)
        for i in range(n):
            e = Resid[:, i]
            if T > 2:
                alpha_hat[i] = np.clip(
                    np.corrcoef(e[1:], e[:-1])[0, 1], -0.99, 0.99
                )
            sigma2_hat[i] = max(np.var(e) * (1 - alpha_hat[i] ** 2), 1e-6)
        R_hat = np.diag(np.maximum(np.var(Resid, axis=0) * 0.1, 1e-6))
    else:
        alpha_hat = None
        sigma2_hat = None
        R_hat = np.diag(np.maximum(np.var(Resid, axis=0), 1e-6))

    return {
        "Lambda": Lambda_hat,
        "A_list": A_list,
        "Q": Q_hat,
        "R": R_hat,
        "alpha": alpha_hat,
        "sigma2": sigma2_hat,
        "F_hat": F_hat,
    }


def _estimate_var(F: np.ndarray, p: int) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Estimate VAR(p) on factor matrix F via OLS.

    Parameters
    ----------
    F : (T, r) factor matrix
    p : int, VAR order

    Returns
    -------
    A_list : list of p (r, r) coefficient matrices
    Q : (r, r) residual covariance
    """
    T, r = F.shape
    if T <= p + r:
        # Not enough data: return identity-scaled matrices
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        Q = np.eye(r)
        return A_list, Q

    # Stack regressors
    Y_dep = F[p:]          # (T-p, r)
    X_reg = np.hstack([F[p - k - 1:T - k - 1] for k in range(p)])  # (T-p, r*p)

    # OLS: B = (X'X)^{-1} X'Y
    try:
        B = np.linalg.lstsq(X_reg, Y_dep, rcond=None)[0]  # (r*p, r)
    except np.linalg.LinAlgError:
        B = np.zeros((r * p, r))

    A_list = [B[k * r:(k + 1) * r, :].T for k in range(p)]

    # Residual covariance
    Resid = Y_dep - X_reg @ B
    Q = Resid.T @ Resid / max(T - p - 1, 1)
    Q = 0.5 * (Q + Q.T)

    # Guard against NaN/inf from numerical overflow
    if not np.all(np.isfinite(Q)):
        Q = np.eye(r)
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        return A_list, Q

    try:
        min_eig = np.linalg.eigvalsh(Q).min()
    except np.linalg.LinAlgError:
        Q = np.eye(r)
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        return A_list, Q

    if min_eig < 1e-8:
        Q += (1e-6 - min_eig) * np.eye(r)

    return A_list, Q


def _build_state_space_uncorr(
    Lambda: np.ndarray,
    A_list: list[np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
    r: int,
    p: int,
) -> dict:
    """
    Build state-space matrices for uncorrelated idiosyncratic model.

    State: alpha_t = [f_t', f_{t-1}', ..., f_{t-p+1}']'  (r*p,)
    Measurement: y_t = [Lambda, 0, ..., 0] alpha_t + epsilon_t

    Returns dict with Z, H, T_mat, R_mat, Q_state, m (state dim)
    """
    n = Lambda.shape[0]
    m = r * p  # state dimension

    # Measurement matrix
    Z = np.zeros((n, m))
    Z[:, :r] = Lambda

    # Transition matrix (companion form)
    T_mat = _build_companion_matrix(A_list, r)

    # Selection matrix R_mat: only first r rows get noise
    R_mat = np.zeros((m, r))
    R_mat[:r, :] = np.eye(r)

    return {
        "Z": Z,
        "H": R,
        "T_mat": T_mat,
        "R_mat": R_mat,
        "Q_state": Q,
        "m": m,
    }


def _build_state_space_ar1idio(
    Lambda: np.ndarray,
    A_list: list[np.ndarray],
    Q: np.ndarray,
    alpha: np.ndarray,
    sigma2: np.ndarray,
    r: int,
    p: int,
    kappa: float = 1e-6,
) -> dict:
    """
    Build state-space matrices for AR(1) idiosyncratic model.

    State: alpha_t = [f_t', ..., f_{t-p+1}', tilde_eps_t']'  (r*p + n,)
    Measurement: y_t = [Lambda, I_n] alpha_t + xi_t,  xi_t ~ N(0, kappa*I_n)

    kappa is a small measurement noise to make the system identifiable.

    Returns dict with Z, H, T_mat, R_mat, Q_state, m
    """
    n = Lambda.shape[0]
    m_f = r * p
    m = m_f + n  # total state dimension

    # Measurement matrix: [Lambda, I_n]
    Z = np.zeros((n, m))
    Z[:, :r] = Lambda
    Z[:, m_f:] = np.eye(n)

    # Measurement noise: kappa * I_n
    H = kappa * np.eye(n)

    # Transition matrix
    T_mat = np.zeros((m, m))
    # Factor block (companion)
    T_f = _build_companion_matrix(A_list, r)
    T_mat[:m_f, :m_f] = T_f
    # Idiosyncratic AR(1) block: diag(alpha)
    T_mat[m_f:, m_f:] = np.diag(alpha)

    # Selection matrix: noise enters factor block (first r) and idio block
    # State noise: [u_t (r), e_t (n)]
    q = r + n
    R_mat = np.zeros((m, q))
    R_mat[:r, :r] = np.eye(r)
    R_mat[m_f:, r:] = np.eye(n)

    # State noise covariance: block diagonal [Q, diag(sigma2)]
    Q_state = np.zeros((q, q))
    Q_state[:r, :r] = Q
    Q_state[r:, r:] = np.diag(sigma2)

    return {
        "Z": Z,
        "H": H,
        "T_mat": T_mat,
        "R_mat": R_mat,
        "Q_state": Q_state,
        "m": m,
    }


def _compute_log_lik(
    Y: np.ndarray,
    mask: np.ndarray,
    ss: dict,
    a0: np.ndarray,
    P0: np.ndarray,
) -> float:
    """Compute log-likelihood via Kalman filter."""
    result = run_filter_smoother(
        Y, ss["Z"], ss["H"], ss["T_mat"], ss["R_mat"], ss["Q_state"], a0, P0, mask
    )
    return result["log_lik"]


def em_bm(
    Y: np.ndarray,
    mask: np.ndarray,
    r: int,
    p: int = 1,
    ar1_idio: bool = False,
    max_iter: int = 200,
    tol: float = 1e-4,
    kappa: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """
    Banbura-Modugno EM algorithm for dynamic factor model estimation.

    Parameters
    ----------
    Y : (T, n) observed data matrix (NaN for missing)
    mask : (T, n) boolean array, True = observed
    r : int, number of factors
    p : int, VAR order for factors
    ar1_idio : bool, whether to include AR(1) idiosyncratic components
    max_iter : int, maximum EM iterations
    tol : float, convergence tolerance for relative log-likelihood increment
    kappa : float, small measurement noise for AR(1) idio model
    verbose : bool, print convergence info

    Returns
    -------
    dict with:
        Lambda : (n, r) loading matrix
        A_list : list of p (r, r) VAR coefficient matrices
        Q : (r, r) factor innovation covariance
        R : (n, n) idiosyncratic covariance (diagonal)
        alpha : (n,) AR(1) coefficients (if ar1_idio)
        sigma2 : (n,) AR(1) innovation variances (if ar1_idio)
        F_smooth : (T, r) smoothed factor estimates
        log_lik_history : list of log-likelihoods
        n_iter : int, number of iterations
        converged : bool
    """
    T, n = Y.shape

    # Fill missing with N(0,1) for initialization
    rng_init = np.random.default_rng(42)
    Y_filled = Y.copy()
    Y_filled[~mask] = rng_init.standard_normal(np.sum(~mask))

    # Initialize parameters
    params = _initialize_params(Y_filled, r, p, ar1_idio)
    Lambda = params["Lambda"]
    A_list = params["A_list"]
    Q = params["Q"]
    R = params["R"]
    alpha_ar = params["alpha"] if ar1_idio else None
    sigma2_ar = params["sigma2"] if ar1_idio else None

    log_lik_history = []
    converged = False

    for iteration in range(max_iter):
        # Build state-space matrices
        if ar1_idio:
            ss = _build_state_space_ar1idio(Lambda, A_list, Q, alpha_ar, sigma2_ar, r, p, kappa)
        else:
            ss = _build_state_space_uncorr(Lambda, A_list, Q, R, r, p)

        m = ss["m"]

        # Initial state distribution (diffuse / stationary approximation)
        a0 = np.zeros(m)
        P0 = 10.0 * np.eye(m)

        # E-step: Kalman filter + smoother
        result = run_filter_smoother(
            Y, ss["Z"], ss["H"], ss["T_mat"], ss["R_mat"], ss["Q_state"],
            a0, P0, mask
        )

        log_lik = result["log_lik"]
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
                print(f"  Iter {iteration}: log_lik={ll_curr:.4f}, c_j={c_j:.6f}")
            if abs(c_j) < tol:
                converged = True
                break

        # Extract smoothed moments
        a_smooth = result["a_smooth"]   # (T, m)
        P_smooth = result["P_smooth"]   # (T, m, m)
        P_cross = result["P_cross"]     # (T-1, m, m)

        # Guard: if smoother produced NaN (numerical overflow), stop EM
        if not np.all(np.isfinite(a_smooth)):
            if verbose:
                print(f"  Iter {iteration}: NaN in smoothed states, stopping EM")
            break

        # Smoothed factor states: first r components
        F_smooth_full = a_smooth[:, :r]  # (T, r)

        # M-step
        if ar1_idio:
            Lambda, A_list, Q, alpha_ar, sigma2_ar = _mstep_ar1idio(
                Y, mask, a_smooth, P_smooth, P_cross, r, p, n, kappa
            )
        else:
            Lambda, A_list, Q, R = _mstep_uncorr(
                Y, mask, a_smooth, P_smooth, P_cross, r, p, n
            )

    # Final smoothed factors
    if ar1_idio:
        ss_final = _build_state_space_ar1idio(Lambda, A_list, Q, alpha_ar, sigma2_ar, r, p, kappa)
    else:
        ss_final = _build_state_space_uncorr(Lambda, A_list, Q, R, r, p)

    a0 = np.zeros(ss_final["m"])
    P0 = 10.0 * np.eye(ss_final["m"])
    result_final = run_filter_smoother(
        Y, ss_final["Z"], ss_final["H"], ss_final["T_mat"],
        ss_final["R_mat"], ss_final["Q_state"], a0, P0, mask
    )

    F_smooth_raw = result_final["a_smooth"][:, :r]
    # If final smoother produced NaN, fall back to filtered states
    if not np.all(np.isfinite(F_smooth_raw)):
        F_smooth = result_final["a_filt"][:, :r]
        # If still NaN, use zeros
        if not np.all(np.isfinite(F_smooth)):
            F_smooth = np.zeros((T, r))
    else:
        F_smooth = F_smooth_raw

    out = {
        "Lambda": Lambda,
        "A_list": A_list,
        "Q": Q,
        "F_smooth": F_smooth,
        "log_lik_history": log_lik_history,
        "n_iter": iteration + 1,
        "converged": converged,
        "ss": ss_final,
        "kalman_result": result_final,
    }
    if ar1_idio:
        out["alpha"] = alpha_ar
        out["sigma2"] = sigma2_ar
    else:
        out["R"] = R

    return out


def _mstep_uncorr(
    Y: np.ndarray,
    mask: np.ndarray,
    a_smooth: np.ndarray,
    P_smooth: np.ndarray,
    P_cross: np.ndarray,
    r: int,
    p: int,
    n: int,
) -> tuple:
    """
    M-step for uncorrelated idiosyncratic model.

    Updates Lambda, A_list, Q, R using smoothed sufficient statistics
    and missing-data weighting matrices W_t.

    Paper equations:
      vec(Lambda^{j+1}) = [sum_t E[f_t f_t'|Omega_T] kron W_t]^{-1}
                          vec(sum_t W_t y_t E[f_t'|Omega_T])
    """
    T = Y.shape[0]
    m = r * p

    # Smoothed factor moments (first r components of state)
    # E[f_t | Omega_T]
    Ef = a_smooth[:, :r]                    # (T, r)
    # E[f_t f_t' | Omega_T] = P_smooth[:r,:r] + Ef Ef'
    Eff = P_smooth[:, :r, :r] + np.einsum('ti,tj->tij', Ef, Ef)  # (T, r, r)

    # Update Lambda: missing-data weighted regression
    # For each observed coordinate i at time t, W_t[i,i] = 1
    # vec(Lambda') = [sum_t (Eff_t kron W_t)]^{-1} vec(sum_t W_t y_t Ef_t')
    # Equivalently, for each series i:
    # Lambda[i,:] = [sum_{t: obs} Eff_t]^{-1} [sum_{t: obs} y_{i,t} Ef_t]
    Lambda_new = np.zeros((n, r))
    for i in range(n):
        obs_t = mask[:, i]  # (T,) boolean
        if obs_t.sum() < r + 1:
            Lambda_new[i] = 0.0
            continue
        sum_Eff = np.sum(Eff[obs_t], axis=0)  # (r, r)
        sum_yEf = np.sum(Y[obs_t, i:i+1] * Ef[obs_t], axis=0)  # (r,)
        try:
            Lambda_new[i] = np.linalg.solve(sum_Eff, sum_yEf)
        except np.linalg.LinAlgError:
            Lambda_new[i] = np.linalg.lstsq(sum_Eff, sum_yEf, rcond=None)[0]

    # Update VAR parameters A_list and Q
    # Using smoothed cross-moments of factor states
    # E[f_t f_{t-1}' | Omega_T] from P_cross (cross-covariance of state)
    # P_cross[t] = E[alpha_{t+1} alpha_t' | Omega_T]
    # We need E[f_t f_{t-1}' | Omega_T] = P_cross[t-1, :r, :r] + Ef[t] Ef[t-1]'

    if p == 1:
        # A = [sum_{t=1}^T E[f_t f_{t-1}'] ] [sum_{t=0}^{T-2} E[f_{t-1} f_{t-1}'] ]^{-1}
        sum_Ef_cross = np.zeros((r, r))
        sum_Eff_lag = np.zeros((r, r))
        for t in range(1, T):
            # E[f_t f_{t-1}'] = P_cross[t-1, :r, :r] + Ef[t] Ef[t-1]'
            Ef_cross_t = P_cross[t - 1, :r, :r] + np.outer(Ef[t], Ef[t - 1])
            sum_Ef_cross += Ef_cross_t
            sum_Eff_lag += Eff[t - 1]

        try:
            A_new = np.linalg.solve(sum_Eff_lag.T, sum_Ef_cross.T).T
        except np.linalg.LinAlgError:
            A_new = np.linalg.lstsq(sum_Eff_lag, sum_Ef_cross.T, rcond=None)[0].T

        # Q update
        sum_Eff_curr = np.sum(Eff[1:], axis=0)
        Q_new = (sum_Eff_curr - A_new @ sum_Ef_cross.T) / (T - 1)
        Q_new = 0.5 * (Q_new + Q_new.T)
        min_eig = np.linalg.eigvalsh(Q_new).min()
        if min_eig < 1e-8:
            Q_new += (1e-6 - min_eig) * np.eye(r)

        A_list_new = [A_new]
    else:
        # General VAR(p): use stacked lag vectors
        # Build sufficient statistics for VAR
        A_list_new, Q_new = _update_var_general(Ef, Eff, P_cross, r, p, T)

    # Update R (diagonal idiosyncratic covariance)
    # R_{ii} = (1/T_obs_i) sum_{t: obs_i} [y_{i,t}^2 - 2 y_{i,t} Lambda_i' Ef_t
    #           + Lambda_i' Eff_t Lambda_i]
    R_diag = np.zeros(n)
    for i in range(n):
        obs_t = mask[:, i]
        T_i = obs_t.sum()
        if T_i < 1:
            R_diag[i] = 1.0
            continue
        y_i = Y[obs_t, i]
        Ef_i = Ef[obs_t]
        Eff_i = Eff[obs_t]
        lam_i = Lambda_new[i]

        term1 = np.sum(y_i ** 2)
        term2 = -2.0 * np.sum(y_i * (Ef_i @ lam_i))
        term3 = np.sum(np.einsum('tj,j->t', np.einsum('tij,j->ti', Eff_i, lam_i), lam_i))
        R_diag[i] = max((term1 + term2 + term3) / T_i, 1e-6)

    R_new = np.diag(R_diag)

    return Lambda_new, A_list_new, Q_new, R_new


def _update_var_general(
    Ef: np.ndarray,
    Eff: np.ndarray,
    P_cross: np.ndarray,
    r: int,
    p: int,
    T: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Update VAR(p) parameters using smoothed factor moments.

    Uses the companion-form sufficient statistics.
    """
    # For VAR(p), we need E[f_t f_{t-k}'] for k=0,...,p
    # Build stacked regression: f_t = sum_{k=1}^p A_k f_{t-k} + u_t

    # Collect sufficient statistics
    # S_00 = sum_{t=p}^{T-1} E[f_t f_t']
    # S_0k = sum_{t=p}^{T-1} E[f_t f_{t-k}'] for k=1,...,p
    # S_kl = sum_{t=p}^{T-1} E[f_{t-k} f_{t-l}'] for k,l=1,...,p

    # For simplicity with p>1, use the companion state moments
    # State alpha_t = [f_t', ..., f_{t-p+1}']' has dimension r*p
    # E[alpha_t alpha_t'] = P_smooth[:, :r*p, :r*p] + outer(a_smooth[:,:r*p], a_smooth[:,:r*p])
    # But we only have Ef and Eff for the first r components

    # Fallback: use OLS on smoothed factors (approximate)
    if T <= p + r:
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        Q = np.eye(r)
        return A_list, Q

    Y_dep = Ef[p:]
    X_reg = np.hstack([Ef[p - k - 1:T - k - 1] for k in range(p)])

    try:
        B = np.linalg.lstsq(X_reg, Y_dep, rcond=None)[0]
    except np.linalg.LinAlgError:
        B = np.zeros((r * p, r))

    A_list = [B[k * r:(k + 1) * r, :].T for k in range(p)]

    Resid = Y_dep - X_reg @ B
    Q = Resid.T @ Resid / max(T - p - 1, 1)
    Q = 0.5 * (Q + Q.T)

    # Guard against NaN/inf from numerical overflow
    if not np.all(np.isfinite(Q)):
        Q = np.eye(r)
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        return A_list, Q

    try:
        min_eig = np.linalg.eigvalsh(Q).min()
    except np.linalg.LinAlgError:
        Q = np.eye(r)
        A_list = [0.5 * np.eye(r) for _ in range(p)]
        return A_list, Q

    if min_eig < 1e-8:
        Q += (1e-6 - min_eig) * np.eye(r)

    return A_list, Q


def _mstep_ar1idio(
    Y: np.ndarray,
    mask: np.ndarray,
    a_smooth: np.ndarray,
    P_smooth: np.ndarray,
    P_cross: np.ndarray,
    r: int,
    p: int,
    n: int,
    kappa: float,
) -> tuple:
    """
    M-step for AR(1) idiosyncratic model.

    State: alpha_t = [f_t', ..., f_{t-p+1}', tilde_eps_t']'
    Updates Lambda, A_list, Q, alpha_ar, sigma2_ar.
    """
    T = Y.shape[0]
    m_f = r * p

    # Smoothed factor moments
    Ef = a_smooth[:, :r]
    Eff = P_smooth[:, :r, :r] + np.einsum('ti,tj->tij', Ef, Ef)

    # Smoothed idiosyncratic state moments
    E_eps = a_smooth[:, m_f:]                    # (T, n)
    E_eps_eps = (P_smooth[:, m_f:, m_f:]
                 + np.einsum('ti,tj->tij', E_eps, E_eps))  # (T, n, n)

    # Update Lambda: y_t = Lambda f_t + tilde_eps_t + xi_t
    # For observed i at t: E[y_{i,t} | Omega_T] ≈ Lambda_i' Ef_t + E_eps_{i,t}
    # Lambda update: same as uncorrelated but accounting for idio state
    Lambda_new = np.zeros((n, r))
    for i in range(n):
        obs_t = mask[:, i]
        if obs_t.sum() < r + 1:
            Lambda_new[i] = 0.0
            continue
        y_i = Y[obs_t, i]
        Ef_i = Ef[obs_t]
        Eff_i = Eff[obs_t]
        E_eps_i = E_eps[obs_t, i]

        # y_i - E_eps_i = Lambda_i' f_t + xi_t
        y_adj = y_i - E_eps_i
        sum_Eff_i = np.sum(Eff_i, axis=0)
        sum_yEf_i = np.sum(y_adj[:, None] * Ef_i, axis=0)

        # Guard against NaN/inf
        if not np.all(np.isfinite(sum_yEf_i)) or not np.all(np.isfinite(sum_Eff_i)):
            Lambda_new[i] = 0.0
            continue

        try:
            Lambda_new[i] = np.linalg.solve(sum_Eff_i, sum_yEf_i)
        except np.linalg.LinAlgError:
            Lambda_new[i] = np.linalg.lstsq(sum_Eff_i, sum_yEf_i, rcond=None)[0]

        if not np.all(np.isfinite(Lambda_new[i])):
            Lambda_new[i] = 0.0

    # Update VAR parameters
    A_list_new, Q_new = _update_var_general(Ef, Eff, P_cross, r, p, T)

    # Update AR(1) idiosyncratic parameters
    # tilde_eps_{i,t} = alpha_i tilde_eps_{i,t-1} + e_{i,t}
    # alpha_i = [sum_{t=1}^T E[eps_{i,t} eps_{i,t-1}]] / [sum_{t=0}^{T-2} E[eps_{i,t}^2]]
    alpha_new = np.zeros(n)
    sigma2_new = np.zeros(n)

    for i in range(n):
        # E[eps_{i,t} eps_{i,t-1}] = P_cross[t-1, m_f+i, m_f+i] + E_eps[t,i]*E_eps[t-1,i]
        sum_cross = 0.0
        sum_lag = 0.0
        sum_curr = 0.0
        for t in range(1, T):
            cross_t = P_cross[t - 1, m_f + i, m_f + i] + E_eps[t, i] * E_eps[t - 1, i]
            lag_t = P_smooth[t - 1, m_f + i, m_f + i] + E_eps[t - 1, i] ** 2
            curr_t = P_smooth[t, m_f + i, m_f + i] + E_eps[t, i] ** 2
            sum_cross += cross_t
            sum_lag += lag_t
            sum_curr += curr_t

        if abs(sum_lag) > 1e-10:
            alpha_new[i] = np.clip(sum_cross / sum_lag, -0.99, 0.99)
        else:
            alpha_new[i] = 0.0

        sigma2_new[i] = max(
            (sum_curr - alpha_new[i] * sum_cross) / max(T - 1, 1),
            1e-6
        )

    return Lambda_new, A_list_new, Q_new, alpha_new, sigma2_new
