"""
Data Generating Process (DGP) for Monte Carlo simulations.

Implements the dynamic factor model DGP from Banbura & Modugno (2014):
  y_t = chi_t + epsilon_t = sum_{k=0}^s Lambda_k f_{t-k} + epsilon_t
  f_t = A f_{t-1} + u_t,  u_t ~ N(0, I_r),  A = rho * I_r
  epsilon_t = D epsilon_{t-1} + v_t,  D = alpha * I_n,  v_t ~ N(0, Phi)

Phi_{ij} = tau^|i-j| * (1 - alpha^2) * sqrt(gamma_i * gamma_j)
gamma_i = [beta_i / (1 - beta_i)] * [1 / (1 - rho^2)] * sum_{k=0}^s sum_{j=1}^r Lambda_{ij,k}^2

# Custom — Context7 found no library equivalent (paper DGP specification)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DGPConfig:
    """Configuration for the dynamic factor model DGP."""
    n: int = 25          # cross-sectional dimension
    T: int = 100         # time series length
    r: int = 3           # number of factors
    s: int = 0           # number of dynamic lags (0 = static loadings)
    rho: float = 0.7     # factor AR(1) coefficient
    alpha: float = 0.0   # idiosyncratic AR(1) coefficient
    tau: float = 0.0     # idiosyncratic cross-correlation parameter
    beta_fixed: Optional[float] = None  # if set, use fixed beta for all i
    seed: Optional[int] = None


def simulate_dgp(config: DGPConfig, rng: Optional[np.random.Generator] = None) -> dict:
    """
    Simulate data from the dynamic factor model DGP.

    Parameters
    ----------
    config : DGPConfig
        Model configuration parameters.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict with keys:
        Y : (T, n) observed data matrix
        F : (T, r) true factor matrix
        Lambda : list of (n, r) loading matrices [Lambda_0, ..., Lambda_s]
        Epsilon : (T, n) idiosyncratic component matrix
        Chi : (T, n) common component matrix
        Phi : (n, n) idiosyncratic covariance matrix
        beta : (n,) signal-to-noise ratios
        config : DGPConfig
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n, T, r, s = config.n, config.T, config.r, config.s
    rho, alpha, tau = config.rho, config.alpha, config.tau

    # Draw dynamic loadings Lambda_{ij,k} ~ iid N(0,1)
    # Lambda is a list of s+1 matrices, each (n, r)
    Lambda_list = [rng.standard_normal((n, r)) for _ in range(s + 1)]

    # Draw beta_i ~ Uniform(0.1, 0.9) or use fixed beta
    if config.beta_fixed is not None:
        beta = np.full(n, config.beta_fixed)
    else:
        beta = rng.uniform(0.1, 0.9, size=n)

    # Compute gamma_i = [beta_i/(1-beta_i)] * [1/(1-rho^2)] * sum_{k=0}^s sum_{j=1}^r Lambda_{ij,k}^2
    # This controls the signal-to-noise ratio
    sum_lambda_sq = np.zeros(n)
    for k in range(s + 1):
        sum_lambda_sq += np.sum(Lambda_list[k] ** 2, axis=1)  # (n,)

    gamma = (beta / (1.0 - beta)) * (1.0 / (1.0 - rho ** 2)) * sum_lambda_sq  # (n,)

    # Construct idiosyncratic covariance Phi
    # Phi_{ij} = tau^|i-j| * (1 - alpha^2) * sqrt(gamma_i * gamma_j)
    # Custom — Context7 found no library equivalent (paper Eq. for Phi)
    idx = np.arange(n)
    abs_diff = np.abs(idx[:, None] - idx[None, :])  # (n, n)
    Phi = (tau ** abs_diff) * (1.0 - alpha ** 2) * np.sqrt(np.outer(gamma, gamma))

    # Ensure Phi is positive definite (add small regularization if needed)
    min_eig = np.linalg.eigvalsh(Phi).min()
    if min_eig < 1e-10:
        Phi += (1e-8 - min_eig) * np.eye(n)

    # Simulate factors: f_t = rho * f_{t-1} + u_t, u_t ~ N(0, I_r)
    # A = rho * I_r
    F = np.zeros((T, r))
    # Initialize from stationary distribution: f_0 ~ N(0, I_r / (1 - rho^2))
    var_f0 = 1.0 / (1.0 - rho ** 2) if abs(rho) < 1.0 else 1.0
    F[0] = rng.multivariate_normal(np.zeros(r), var_f0 * np.eye(r))
    for t in range(1, T):
        F[t] = rho * F[t - 1] + rng.standard_normal(r)

    # Simulate idiosyncratic components: epsilon_t = alpha * epsilon_{t-1} + v_t
    # v_t ~ N(0, Phi)
    Phi_chol = np.linalg.cholesky(Phi)
    Epsilon = np.zeros((T, n))
    # Initialize from stationary distribution
    if abs(alpha) < 1.0:
        # Stationary variance: Sigma_eps = Phi / (1 - alpha^2)
        # But Phi already incorporates (1 - alpha^2), so Sigma_eps = Phi / (1 - alpha^2)
        # Actually Phi = (1-alpha^2) * ..., so stationary var = Phi / (1-alpha^2) * 1/(1-alpha^2)?
        # Let's compute properly: Var(eps) = alpha^2 Var(eps) + Phi => Var(eps) = Phi/(1-alpha^2)
        # But Phi = (1-alpha^2) * sqrt(gamma_i*gamma_j) * tau^|i-j|
        # So Var(eps_i) = gamma_i (diagonal)
        Sigma_eps0 = Phi / (1.0 - alpha ** 2) if abs(alpha) < 1.0 else Phi
        try:
            chol0 = np.linalg.cholesky(Sigma_eps0)
            Epsilon[0] = chol0 @ rng.standard_normal(n)
        except np.linalg.LinAlgError:
            Epsilon[0] = rng.standard_normal(n)
    else:
        Epsilon[0] = rng.standard_normal(n)

    for t in range(1, T):
        v_t = Phi_chol @ rng.standard_normal(n)
        Epsilon[t] = alpha * Epsilon[t - 1] + v_t

    # Compute common component: chi_t = sum_{k=0}^s Lambda_k f_{t-k}
    Chi = np.zeros((T, n))
    for k in range(s + 1):
        for t in range(k, T):
            Chi[t] += F[t - k] @ Lambda_list[k].T  # (r,) @ (r, n) -> (n,)

    # Observed data
    Y = Chi + Epsilon

    return {
        "Y": Y,
        "F": F,
        "Lambda": Lambda_list,
        "Epsilon": Epsilon,
        "Chi": Chi,
        "Phi": Phi,
        "gamma": gamma,
        "beta": beta,
        "config": config,
    }


def impose_random_missing(
    Y: np.ndarray,
    missing_share: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Impose random missing data on the panel Y.

    Parameters
    ----------
    Y : (T, n) observed data matrix
    missing_share : float in [0, 1], fraction of entries to set missing
    rng : random number generator
    seed : int, optional seed if rng is None

    Returns
    -------
    mask : (T, n) boolean array, True = observed, False = missing
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T, n = Y.shape
    mask = rng.random((T, n)) >= missing_share
    return mask


def impose_ragged_edge_missing(
    Y: np.ndarray,
    horizon: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Impose ragged-edge missing data pattern near the end of the sample.

    Patterns (hor1-hor4) from the paper:
      hor1: 20% missing at T
      hor2: 20% at T-1, 40% at T
      hor3: 20% at T-2, 40% at T-1, 60% at T
      hor4: 20% at T-3, 40% at T-2, 60% at T-1, 80% at T

    The forecast target y_{1,T} is always missing.

    Parameters
    ----------
    Y : (T, n) observed data matrix
    horizon : int in {1, 2, 3, 4}
    rng : random number generator
    seed : optional seed

    Returns
    -------
    mask : (T, n) boolean array, True = observed, False = missing
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T, n = Y.shape
    mask = np.ones((T, n), dtype=bool)

    # Missing shares at each lag from end
    patterns = {
        1: [(0, 0.20)],
        2: [(1, 0.20), (0, 0.40)],
        3: [(2, 0.20), (1, 0.40), (0, 0.60)],
        4: [(3, 0.20), (2, 0.40), (1, 0.60), (0, 0.80)],
    }
    assert horizon in patterns, f"horizon must be in {{1,2,3,4}}, got {horizon}"

    for lag, share in patterns[horizon]:
        t_idx = T - 1 - lag
        if t_idx >= 0:
            n_missing = max(1, int(np.round(share * n)))
            missing_cols = rng.choice(n, size=n_missing, replace=False)
            mask[t_idx, missing_cols] = False

    # Ensure forecast target y_{1,T} is missing (series index 0, time T-1)
    mask[T - 1, 0] = False

    return mask


def standardize_panel(Y: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize each series to mean 0 and variance 1 using observed values.

    Parameters
    ----------
    Y : (T, n) data matrix (may contain NaN for missing)
    mask : (T, n) boolean mask (True = observed), optional

    Returns
    -------
    Y_std : (T, n) standardized data
    means : (n,) series means
    stds : (n,) series standard deviations
    """
    T, n = Y.shape
    Y_work = Y.copy().astype(float)
    if mask is not None:
        Y_work[~mask] = np.nan

    means = np.nanmean(Y_work, axis=0)
    stds = np.nanstd(Y_work, axis=0, ddof=1)
    stds = np.where(stds < 1e-12, 1.0, stds)

    Y_std = (Y_work - means[None, :]) / stds[None, :]
    return Y_std, means, stds
