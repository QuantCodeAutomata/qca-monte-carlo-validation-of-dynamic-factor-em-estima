"""
Monte Carlo Framework for Dynamic Factor Model Validation.

Implements three substudies:
  A: Factor-space recovery (Tables 1 & 2)
  B: Ragged-edge forecast accuracy (Table 4)
  C: AR(1) idiosyncratic coefficient estimation (Table 3)

# Custom — Context7 found no library equivalent for this specific Monte Carlo
# framework as described in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
import warnings

from .dgp import DGPConfig, simulate_dgp, impose_random_missing, impose_ragged_edge_missing
from .bm_estimator import em_bm
from .rt_estimator import em_rt
from .sw_estimator import em_sw
from .metrics import trace_r2, relative_forecast_accuracy, mean_abs_error_ar


warnings.filterwarnings("ignore")

N_REPLICATIONS = 500  # Paper uses 500 Monte Carlo replications


@dataclass
class MCResult:
    """Container for Monte Carlo results."""
    bm_r2: float = 0.0
    rt_r2: float = 0.0
    sw_r2: float = 0.0
    bm_rt_ratio: float = 0.0
    bm_sw_ratio: float = 0.0
    n_reps: int = 0


def run_single_replication_substudy_a(
    config: DGPConfig,
    rhat: int,
    missing_share: float,
    seed: int,
) -> dict:
    """
    Run a single Monte Carlo replication for Substudy A (factor-space recovery).

    Parameters
    ----------
    config : DGPConfig
    rhat : int, estimated number of factors
    missing_share : float, fraction of missing data
    seed : int, random seed

    Returns
    -------
    dict with bm_r2, rt_r2, sw_r2
    """
    rng = np.random.default_rng(seed)

    # Simulate DGP
    data = simulate_dgp(config, rng)
    Y = data["Y"]
    F_true = data["F"]

    # Impose missing data (same mask for all estimators)
    mask = impose_random_missing(Y, missing_share, rng)

    # Replace missing with NaN for estimators
    Y_nan = Y.copy().astype(float)
    Y_nan[~mask] = np.nan

    results = {}

    # BM estimator
    try:
        bm_res = em_bm(Y_nan, mask, r=rhat, p=1, ar1_idio=False, max_iter=50, tol=1e-4)
        F_bm = bm_res["F_smooth"]
        results["bm_r2"] = trace_r2(F_true, F_bm)
    except Exception:
        results["bm_r2"] = 0.0

    # RT estimator
    try:
        rt_res = em_rt(Y_nan, mask, r=rhat, max_iter=50, tol=1e-4)
        F_rt = rt_res["F_hat"]
        results["rt_r2"] = trace_r2(F_true, F_rt)
    except Exception:
        results["rt_r2"] = 0.0

    # SW estimator
    try:
        sw_res = em_sw(Y_nan, mask, r=rhat, max_iter=50, tol=1e-4)
        F_sw = sw_res["F_hat"]
        results["sw_r2"] = trace_r2(F_true, F_sw)
    except Exception:
        results["sw_r2"] = 0.0

    return results


def run_substudy_a(
    n_reps: int = N_REPLICATIONS,
    pilot: bool = False,
) -> pd.DataFrame:
    """
    Substudy A: Factor-space recovery (Tables 1 & 2).

    Runs Monte Carlo over parameter grid and computes average trace R^2.

    Parameters
    ----------
    n_reps : int, number of Monte Carlo replications
    pilot : bool, if True use reduced grid for testing

    Returns
    -------
    pd.DataFrame with results
    """
    if pilot:
        n_reps = 10

    # Parameter grid (subset matching paper's Tables 1 & 2)
    param_grid = []

    r_vals = [1, 3] if not pilot else [1]
    n_vals = [10, 25, 50, 100] if not pilot else [25]
    T_vals = [50, 100] if not pilot else [100]
    rho_vals = [0.5, 0.7, 0.9] if not pilot else [0.7]
    alpha_tau_vals = [(0, 0), (0.5, 0.5)] if not pilot else [(0, 0)]
    s_vals = [0, 1] if not pilot else [0]
    rhat_offsets = [0, 1] if not pilot else [0]  # rhat = r + offset
    missing_shares = [0.0, 0.10, 0.25, 0.40] if not pilot else [0.0, 0.25]

    for r in r_vals:
        for n in n_vals:
            for T in T_vals:
                for rho in rho_vals:
                    for alpha, tau in alpha_tau_vals:
                        for s in s_vals:
                            for rhat_offset in rhat_offsets:
                                rhat = r + rhat_offset
                                for miss in missing_shares:
                                    param_grid.append({
                                        "r": r, "n": n, "T": T, "rho": rho,
                                        "alpha": alpha, "tau": tau, "s": s,
                                        "rhat": rhat, "missing_share": miss,
                                    })

    rows = []
    total = len(param_grid)
    print(f"Substudy A: {total} parameter cells × {n_reps} replications")

    for cell_idx, params in enumerate(param_grid):
        if cell_idx % 50 == 0:
            print(f"  Cell {cell_idx}/{total}...")

        config = DGPConfig(
            n=params["n"],
            T=params["T"],
            r=params["r"],
            s=params["s"],
            rho=params["rho"],
            alpha=params["alpha"],
            tau=params["tau"],
        )

        bm_r2_list, rt_r2_list, sw_r2_list = [], [], []

        for rep in range(n_reps):
            seed = cell_idx * 10000 + rep
            res = run_single_replication_substudy_a(
                config, params["rhat"], params["missing_share"], seed
            )
            bm_r2_list.append(res["bm_r2"])
            rt_r2_list.append(res["rt_r2"])
            sw_r2_list.append(res["sw_r2"])

        bm_mean = np.mean(bm_r2_list)
        rt_mean = np.mean(rt_r2_list)
        sw_mean = np.mean(sw_r2_list)

        row = {**params}
        row["bm_r2"] = bm_mean
        row["rt_r2"] = rt_mean
        row["sw_r2"] = sw_mean
        row["bm_rt_ratio"] = bm_mean / max(rt_mean, 1e-6)
        row["bm_sw_ratio"] = bm_mean / max(sw_mean, 1e-6)
        rows.append(row)

    return pd.DataFrame(rows)


def run_single_replication_substudy_b(
    config: DGPConfig,
    rhat: int,
    horizon: int,
    seed: int,
) -> dict:
    """
    Run a single Monte Carlo replication for Substudy B (ragged-edge forecast).

    Parameters
    ----------
    config : DGPConfig (with beta_fixed=0.5)
    rhat : int, estimated number of factors
    horizon : int in {1,2,3,4}
    seed : int

    Returns
    -------
    dict with bm_acc, rt_acc, sw_acc
    """
    rng = np.random.default_rng(seed)

    # Simulate DGP
    data = simulate_dgp(config, rng)
    Y = data["Y"]
    Chi = data["Chi"]

    # True common component at T for series 1
    chi_1T = Chi[-1, 0]
    var_chi = np.var(Chi[:, 0])
    if var_chi < 1e-10:
        var_chi = 1.0

    # Impose ragged-edge missing data
    mask = impose_ragged_edge_missing(Y, horizon, rng)

    Y_nan = Y.copy().astype(float)
    Y_nan[~mask] = np.nan

    results = {}

    # BM estimator
    try:
        bm_res = em_bm(Y_nan, mask, r=rhat, p=1, ar1_idio=False, max_iter=50, tol=1e-4)
        # Forecast chi_{1,T}: use smoothed factor and loading
        F_bm = bm_res["F_smooth"]
        Lambda_bm = bm_res["Lambda"]
        chi_hat_bm = float(F_bm[-1] @ Lambda_bm[0])
        results["bm_acc"] = relative_forecast_accuracy(chi_1T, chi_hat_bm, var_chi)
    except Exception:
        results["bm_acc"] = 0.0

    # RT estimator
    try:
        rt_res = em_rt(Y_nan, mask, r=rhat, max_iter=50, tol=1e-4)
        F_rt = rt_res["F_hat"]
        Lambda_rt = rt_res["Lambda"]
        chi_hat_rt = float(F_rt[-1] @ Lambda_rt[0])
        results["rt_acc"] = relative_forecast_accuracy(chi_1T, chi_hat_rt, var_chi)
    except Exception:
        results["rt_acc"] = 0.0

    # SW estimator
    try:
        sw_res = em_sw(Y_nan, mask, r=rhat, max_iter=50, tol=1e-4)
        F_sw = sw_res["F_hat"]
        Lambda_sw = sw_res["Lambda"]
        chi_hat_sw = float(F_sw[-1] @ Lambda_sw[0])
        results["sw_acc"] = relative_forecast_accuracy(chi_1T, chi_hat_sw, var_chi)
    except Exception:
        results["sw_acc"] = 0.0

    return results


def run_substudy_b(
    n_reps: int = N_REPLICATIONS,
    pilot: bool = False,
) -> pd.DataFrame:
    """
    Substudy B: Ragged-edge forecast accuracy (Table 4).

    Parameters
    ----------
    n_reps : int
    pilot : bool

    Returns
    -------
    pd.DataFrame
    """
    if pilot:
        n_reps = 10

    param_grid = []

    r_vals = [3]
    n_vals = [10, 25, 50, 100] if not pilot else [25]
    T_vals = [50, 100] if not pilot else [100]
    rho_vals = [0.5, 0.7, 0.9] if not pilot else [0.7]
    alpha_tau_vals = [(0, 0), (0.5, 0.5)] if not pilot else [(0, 0)]
    s_vals = [0, 1] if not pilot else [0]
    rhat_offsets = [0, 1] if not pilot else [0]
    horizons = [1, 2, 3, 4] if not pilot else [1, 4]

    for r in r_vals:
        for n in n_vals:
            for T in T_vals:
                for rho in rho_vals:
                    for alpha, tau in alpha_tau_vals:
                        for s in s_vals:
                            for rhat_offset in rhat_offsets:
                                rhat = r + rhat_offset
                                for hor in horizons:
                                    param_grid.append({
                                        "r": r, "n": n, "T": T, "rho": rho,
                                        "alpha": alpha, "tau": tau, "s": s,
                                        "rhat": rhat, "horizon": hor,
                                    })

    rows = []
    total = len(param_grid)
    print(f"Substudy B: {total} parameter cells × {n_reps} replications")

    for cell_idx, params in enumerate(param_grid):
        if cell_idx % 50 == 0:
            print(f"  Cell {cell_idx}/{total}...")

        config = DGPConfig(
            n=params["n"],
            T=params["T"],
            r=params["r"],
            s=params["s"],
            rho=params["rho"],
            alpha=params["alpha"],
            tau=params["tau"],
            beta_fixed=0.5,  # Table 4 uses beta=0.5 for all i
        )

        bm_acc_list, rt_acc_list, sw_acc_list = [], [], []

        for rep in range(n_reps):
            seed = cell_idx * 10000 + rep + 1000000
            res = run_single_replication_substudy_b(
                config, params["rhat"], params["horizon"], seed
            )
            bm_acc_list.append(res["bm_acc"])
            rt_acc_list.append(res["rt_acc"])
            sw_acc_list.append(res["sw_acc"])

        bm_mean = np.mean(bm_acc_list)
        rt_mean = np.mean(rt_acc_list)
        sw_mean = np.mean(sw_acc_list)

        row = {**params}
        row["bm_acc"] = bm_mean
        row["rt_acc"] = rt_mean
        row["sw_acc"] = sw_mean
        row["bm_rt_ratio"] = bm_mean / max(rt_mean, 1e-6) if rt_mean > 0 else None
        row["bm_sw_ratio"] = bm_mean / max(sw_mean, 1e-6) if sw_mean > 0 else None
        rows.append(row)

    return pd.DataFrame(rows)


def run_single_replication_substudy_c(
    T: int,
    missing_share: float,
    seed: int,
    n: int = 25,
    r: int = 3,
    rho: float = 0.7,
    alpha: float = 0.7,
    tau: float = 0.0,
    s: int = 0,
) -> float:
    """
    Run a single Monte Carlo replication for Substudy C (AR(1) idio estimation).

    Parameters
    ----------
    T : int, sample size
    missing_share : float
    seed : int
    n, r, rho, alpha, tau, s : DGP parameters (fixed per paper)

    Returns
    -------
    float, mean absolute error of alpha estimates
    """
    rng = np.random.default_rng(seed)

    config = DGPConfig(n=n, T=T, r=r, s=s, rho=rho, alpha=alpha, tau=tau)
    data = simulate_dgp(config, rng)
    Y = data["Y"]
    alpha_true = alpha  # scalar, same for all i

    # Impose missing data
    mask = impose_random_missing(Y, missing_share, rng)
    Y_nan = Y.copy().astype(float)
    Y_nan[~mask] = np.nan

    try:
        bm_res = em_bm(
            Y_nan, mask, r=r, p=1, ar1_idio=True,
            max_iter=50, tol=1e-4, kappa=1e-6
        )
        alpha_hat = bm_res["alpha"]
        mae = mean_abs_error_ar(alpha_hat, alpha_true)
    except Exception:
        mae = np.nan

    return mae


def run_substudy_c(
    n_reps: int = N_REPLICATIONS,
    pilot: bool = False,
) -> pd.DataFrame:
    """
    Substudy C: AR(1) idiosyncratic coefficient estimation (Table 3).

    Fixed parameters: n=25, rho=0.7, alpha=0.7, tau=0, beta~U(0.1,0.9), s=0, rhat=r=3.
    Vary T in {50,100,200,500,1000} and missing_share in {0, 0.20}.

    Paper target values:
      no missing:  {0.127, 0.075, 0.047, 0.028, 0.020}
      20% missing: {0.135, 0.079, 0.050, 0.030, 0.021}

    Parameters
    ----------
    n_reps : int
    pilot : bool

    Returns
    -------
    pd.DataFrame
    """
    if pilot:
        n_reps = 10

    T_vals = [50, 100, 200, 500, 1000] if not pilot else [50, 100, 200]
    missing_shares = [0.0, 0.20] if not pilot else [0.0]

    # Paper target values for comparison
    paper_targets = {
        0.0: [0.127, 0.075, 0.047, 0.028, 0.020],
        0.20: [0.135, 0.079, 0.050, 0.030, 0.021],
    }

    rows = []
    total = len(T_vals) * len(missing_shares)
    print(f"Substudy C: {total} parameter cells × {n_reps} replications")

    for miss_idx, miss in enumerate(missing_shares):
        for T_idx, T in enumerate(T_vals):
            print(f"  T={T}, missing={miss:.0%}...")

            mae_list = []
            for rep in range(n_reps):
                seed = T_idx * 10000 + rep + 2000000 + miss_idx * 100000
                mae = run_single_replication_substudy_c(T, miss, seed)
                if not np.isnan(mae):
                    mae_list.append(mae)

            mean_mae = np.mean(mae_list) if mae_list else np.nan
            target = paper_targets[miss][T_idx] if T_idx < len(paper_targets[miss]) else np.nan

            rows.append({
                "T": T,
                "missing_share": miss,
                "mean_mae": mean_mae,
                "paper_target": target,
                "n_reps_valid": len(mae_list),
            })

    return pd.DataFrame(rows)


def run_all_substudies(
    n_reps: int = N_REPLICATIONS,
    pilot: bool = False,
) -> dict:
    """
    Run all three Monte Carlo substudies.

    Parameters
    ----------
    n_reps : int
    pilot : bool, if True use reduced grids for testing

    Returns
    -------
    dict with keys 'substudy_a', 'substudy_b', 'substudy_c'
    """
    print("=" * 60)
    print("Running Monte Carlo Experiment (exp_1)")
    print("=" * 60)

    print("\n--- Substudy A: Factor-Space Recovery ---")
    df_a = run_substudy_a(n_reps=n_reps, pilot=pilot)

    print("\n--- Substudy B: Ragged-Edge Forecast Accuracy ---")
    df_b = run_substudy_b(n_reps=n_reps, pilot=pilot)

    print("\n--- Substudy C: AR(1) Idiosyncratic Estimation ---")
    df_c = run_substudy_c(n_reps=n_reps, pilot=pilot)

    return {
        "substudy_a": df_a,
        "substudy_b": df_b,
        "substudy_c": df_c,
    }
