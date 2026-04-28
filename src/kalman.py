"""
Kalman filter and smoother with arbitrary missing data handling.

Implements the standard Kalman filter and RTS smoother for state-space models
of the form:
    y_t = Z_t alpha_t + epsilon_t,  epsilon_t ~ N(0, H_t)
    alpha_{t+1} = T alpha_t + R eta_t,  eta_t ~ N(0, Q)

Missing observations are handled by row-deletion: at each time t, only the
rows of Z_t and H_t corresponding to observed series are used.

# Custom — Context7 found no library equivalent for arbitrary missing-data
# Kalman filter with row-deletion as described in the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def kalman_filter(
    Y: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    T_mat: np.ndarray,
    R_mat: np.ndarray,
    Q: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Kalman filter with arbitrary missing data (row-deletion method).

    Parameters
    ----------
    Y : (T_obs, n) observation matrix (NaN or mask for missing)
    Z : (n, m) measurement matrix (time-invariant)
    H : (n, n) measurement noise covariance (diagonal, time-invariant)
    T_mat : (m, m) state transition matrix
    R_mat : (m, q) selection matrix for state noise
    Q : (q, q) state noise covariance
    a0 : (m,) initial state mean
    P0 : (m, m) initial state covariance
    mask : (T_obs, n) boolean array, True = observed; if None, use ~np.isnan(Y)

    Returns
    -------
    dict with:
        a_filt : (T_obs, m) filtered state means
        P_filt : (T_obs, m, m) filtered state covariances
        a_pred : (T_obs+1, m) predicted state means (a_pred[0] = a0)
        P_pred : (T_obs+1, m, m) predicted state covariances
        log_lik : float, total log-likelihood
        v : list of innovation vectors
        F_innov : list of innovation covariance matrices
        K : list of Kalman gain matrices
    """
    T_obs, n = Y.shape
    m = len(a0)

    if mask is None:
        mask = ~np.isnan(Y)

    # Pre-compute RQR'
    RQR = R_mat @ Q @ R_mat.T  # (m, m)

    a_pred = np.zeros((T_obs + 1, m))
    P_pred = np.zeros((T_obs + 1, m, m))
    a_filt = np.zeros((T_obs, m))
    P_filt = np.zeros((T_obs, m, m))

    a_pred[0] = a0
    P_pred[0] = P0

    log_lik = 0.0
    v_list = []
    F_list = []
    K_list = []

    for t in range(T_obs):
        obs_idx = np.where(mask[t])[0]

        if len(obs_idx) == 0:
            # All missing: prediction only
            a_filt[t] = a_pred[t]
            P_filt[t] = P_pred[t]
            v_list.append(np.zeros(0))
            F_list.append(np.zeros((0, 0)))
            K_list.append(np.zeros((m, 0)))
        else:
            Z_t = Z[obs_idx, :]          # (n_obs, m)
            H_t = H[np.ix_(obs_idx, obs_idx)]  # (n_obs, n_obs)
            y_t = Y[t, obs_idx]          # (n_obs,)

            # Innovation
            v_t = y_t - Z_t @ a_pred[t]  # (n_obs,)

            # Innovation covariance
            F_t = Z_t @ P_pred[t] @ Z_t.T + H_t  # (n_obs, n_obs)

            # Symmetrize for numerical stability
            F_t = 0.5 * (F_t + F_t.T)

            # Kalman gain: K_t = P_pred[t] Z_t' F_t^{-1}
            try:
                F_inv = np.linalg.solve(F_t, Z_t @ P_pred[t])  # (n_obs, m)
                K_t = F_inv.T  # (m, n_obs)
            except np.linalg.LinAlgError:
                F_t += 1e-8 * np.eye(len(obs_idx))
                F_inv = np.linalg.solve(F_t, Z_t @ P_pred[t])
                K_t = F_inv.T

            # Filtered state
            a_filt[t] = a_pred[t] + K_t @ v_t
            P_filt[t] = P_pred[t] - K_t @ Z_t @ P_pred[t]
            P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T)

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(F_t)
            if sign > 0:
                n_obs = len(obs_idx)
                log_lik += -0.5 * (
                    n_obs * np.log(2 * np.pi)
                    + logdet
                    + v_t @ np.linalg.solve(F_t, v_t)
                )

            v_list.append(v_t)
            F_list.append(F_t)
            K_list.append(K_t)

        # Prediction step
        a_next = T_mat @ a_filt[t]
        P_next = T_mat @ P_filt[t] @ T_mat.T + RQR
        P_next = 0.5 * (P_next + P_next.T)

        # Guard against overflow: clip state values to prevent divergence
        a_next = np.clip(a_next, -1e6, 1e6)
        if not np.all(np.isfinite(a_next)):
            a_next = np.zeros(m)
        if not np.all(np.isfinite(P_next)):
            P_next = P_pred[t] + RQR  # fallback: keep growing uncertainty
        # Clip covariance diagonal to prevent runaway variance
        np.fill_diagonal(P_next, np.clip(np.diag(P_next), 0.0, 1e8))

        a_pred[t + 1] = a_next
        P_pred[t + 1] = P_next

    return {
        "a_filt": a_filt,
        "P_filt": P_filt,
        "a_pred": a_pred,
        "P_pred": P_pred,
        "log_lik": log_lik,
        "v": v_list,
        "F_innov": F_list,
        "K": K_list,
    }


def kalman_smoother(
    a_filt: np.ndarray,
    P_filt: np.ndarray,
    a_pred: np.ndarray,
    P_pred: np.ndarray,
    T_mat: np.ndarray,
) -> dict:
    """
    RTS (Rauch-Tung-Striebel) smoother.

    Parameters
    ----------
    a_filt : (T_obs, m) filtered state means
    P_filt : (T_obs, m, m) filtered state covariances
    a_pred : (T_obs+1, m) predicted state means
    P_pred : (T_obs+1, m, m) predicted state covariances
    T_mat : (m, m) state transition matrix

    Returns
    -------
    dict with:
        a_smooth : (T_obs, m) smoothed state means
        P_smooth : (T_obs, m, m) smoothed state covariances
        P_cross : (T_obs-1, m, m) cross-covariances E[alpha_t alpha_{t-1}' | Y]
        L : (T_obs, m, m) smoother gain matrices
    """
    T_obs, m = a_filt.shape

    a_smooth = np.zeros((T_obs, m))
    P_smooth = np.zeros((T_obs, m, m))
    P_cross = np.zeros((T_obs - 1, m, m))  # P_{t,t-1|T}
    L_gain = np.zeros((T_obs, m, m))

    # Initialize at T
    a_smooth[-1] = a_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T_obs - 2, -1, -1):
        # Smoother gain: L_t = P_filt[t] T' P_pred[t+1]^{-1}
        try:
            L_t = np.linalg.solve(P_pred[t + 1].T, (T_mat @ P_filt[t]).T).T
        except np.linalg.LinAlgError:
            P_pred_reg = P_pred[t + 1] + 1e-8 * np.eye(m)
            L_t = np.linalg.solve(P_pred_reg.T, (T_mat @ P_filt[t]).T).T

        L_gain[t] = L_t

        a_smooth_t = a_filt[t] + L_t @ (a_smooth[t + 1] - a_pred[t + 1])
        P_smooth_t = P_filt[t] + L_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ L_t.T
        P_smooth_t = 0.5 * (P_smooth_t + P_smooth_t.T)

        # Guard against numerical overflow: fall back to filtered values
        if not np.all(np.isfinite(a_smooth_t)):
            a_smooth_t = a_filt[t]
        if not np.all(np.isfinite(P_smooth_t)):
            P_smooth_t = P_filt[t]

        a_smooth[t] = a_smooth_t
        P_smooth[t] = P_smooth_t

    # Compute cross-covariances P_{t+1,t|T} = P_smooth[t+1] L_t'
    # (needed for M-step: E[alpha_{t+1} alpha_t' | Y])
    for t in range(T_obs - 1):
        P_cross[t] = P_smooth[t + 1] @ L_gain[t].T

    return {
        "a_smooth": a_smooth,
        "P_smooth": P_smooth,
        "P_cross": P_cross,
        "L": L_gain,
    }


def run_filter_smoother(
    Y: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    T_mat: np.ndarray,
    R_mat: np.ndarray,
    Q: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Run Kalman filter followed by RTS smoother.

    Returns combined dict with filter and smoother results.
    """
    filt = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)
    smooth = kalman_smoother(
        filt["a_filt"], filt["P_filt"],
        filt["a_pred"], filt["P_pred"],
        T_mat,
    )
    return {**filt, **smooth}
