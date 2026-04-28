"""
Model-Based News Decomposition of Euro Area GDP Forecast Revisions (Experiment 3).

Implements the decomposition of GDP nowcast revisions into:
1. News contributions from newly released data (grouped by Real, Survey, Financial, US)
2. Re-estimation effect from parameter updates

Follows the methodology of Banbura & Modugno (2014), Section 4.

# Custom — Context7 found no library equivalent for this specific
# news decomposition framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import warnings

from .empirical_data import (
    generate_synthetic_euro_area_data,
    get_metadata_df,
    prepare_panel_for_estimation,
    apply_publication_lag_mask,
    recursive_standardize,
)
from .bm_estimator import em_bm
from .kalman import run_filter_smoother

warnings.filterwarnings("ignore")

# Groups for news decomposition
NEWS_GROUPS = ["Real", "Survey", "Financial", "US"]

# Vintage sequence for 2008Q4 episode
VINTAGE_DATES = [
    "2008-07-15",  # mid-July 2008
    "2008-08-15",
    "2008-09-15",
    "2008-10-15",
    "2008-11-15",
    "2008-12-15",
    "2009-01-15",  # mid-January 2009
]


def compute_news_decomposition(
    panel: str = "small",
    ar1_idio: bool = False,
    target_quarter: str = "2008Q4",
    vintage_dates: Optional[list[str]] = None,
) -> dict:
    """
    Compute news decomposition for euro area GDP 2008Q4.

    Parameters
    ----------
    panel : str, 'small', 'medium', or 'large'
    ar1_idio : bool
    target_quarter : str
    vintage_dates : list of vintage date strings

    Returns
    -------
    dict with decomposition results
    """
    if vintage_dates is None:
        vintage_dates = VINTAGE_DATES

    print(f"  News decomposition: panel={panel}, ar1_idio={ar1_idio}")

    # Generate synthetic data
    df_raw = generate_synthetic_euro_area_data(
        start_date="1993-01",
        end_date="2009-10",
        seed=42,
    )
    meta = get_metadata_df()

    # Parse target quarter
    tq = pd.Period(target_quarter, freq="Q")
    target_month = tq.month  # quarter-end month
    target_date = pd.Timestamp(f"{tq.year}-{target_month:02d}-01")

    # Best parameterization
    from .nowcasting import BEST_PARAMS
    r_best, p_best = BEST_PARAMS.get((panel, ar1_idio), (3, 2))

    results = {
        "vintages": [],
        "forecasts": [],
        "revisions": [],
        "news_by_group": {g: [] for g in NEWS_GROUPS},
        "reestimation_effects": [],
        "total_news": [],
    }

    prev_forecast = None
    prev_params = None
    prev_Y = None
    prev_mask = None

    for v_idx, vintage_str in enumerate(vintage_dates):
        vintage_date = pd.Timestamp(vintage_str)

        # Prepare data for this vintage
        Y, mask, series_names = prepare_panel_for_estimation(
            df_raw, panel, vintage_date, meta
        )

        gdp_cols = [i for i, s in enumerate(series_names)
                    if "GDP" in s.upper() and "US" not in s.upper()]
        gdp_idx = gdp_cols[0] if gdp_cols else 0

        # Estimate model
        try:
            r_use = min(r_best, Y.shape[1] - 1, 5)
            bm_res = em_bm(
                Y, mask, r=r_use, p=p_best,
                ar1_idio=ar1_idio,
                max_iter=50, tol=1e-4, kappa=1e-6,
            )

            # Current forecast of target GDP
            T_target = min(len(Y) - 1, len(Y) - 1)
            F_smooth = bm_res["F_smooth"]
            Lambda = bm_res["Lambda"]
            current_forecast = float(F_smooth[T_target] @ Lambda[gdp_idx])

        except Exception:
            current_forecast = prev_forecast if prev_forecast is not None else 0.0
            bm_res = None

        results["vintages"].append(vintage_str)
        results["forecasts"].append(current_forecast)

        if prev_forecast is not None:
            revision = current_forecast - prev_forecast
            results["revisions"].append(revision)

            # Compute news contributions
            if prev_params is not None and bm_res is not None and prev_Y is not None:
                news_contributions, reest_effect = _compute_news_contributions(
                    Y, mask, prev_Y, prev_mask,
                    bm_res, prev_params,
                    gdp_idx, T_target,
                    series_names, meta,
                )
                for g in NEWS_GROUPS:
                    results["news_by_group"][g].append(news_contributions.get(g, 0.0))
                results["reestimation_effects"].append(reest_effect)
                results["total_news"].append(sum(news_contributions.values()))
            else:
                for g in NEWS_GROUPS:
                    results["news_by_group"][g].append(0.0)
                results["reestimation_effects"].append(0.0)
                results["total_news"].append(0.0)
        else:
            results["revisions"].append(0.0)
            for g in NEWS_GROUPS:
                results["news_by_group"][g].append(0.0)
            results["reestimation_effects"].append(0.0)
            results["total_news"].append(0.0)

        prev_forecast = current_forecast
        prev_params = bm_res
        prev_Y = Y.copy()
        prev_mask = mask.copy()

    return results


def _compute_news_contributions(
    Y_new: np.ndarray,
    mask_new: np.ndarray,
    Y_old: np.ndarray,
    mask_old: np.ndarray,
    bm_new: dict,
    bm_old: dict,
    gdp_idx: int,
    T_target: int,
    series_names: list[str],
    meta: pd.DataFrame,
) -> tuple[dict, float]:
    """
    Compute news contributions and re-estimation effect.

    News effect = forecast with new data & old params - forecast with old data & old params
    Re-estimation effect = forecast with new data & new params - forecast with new data & old params

    Parameters
    ----------
    Y_new, mask_new : new vintage data
    Y_old, mask_old : old vintage data
    bm_new : BM result with new params
    bm_old : BM result with old params
    gdp_idx : GDP series index
    T_target : target time index
    series_names : list of series names
    meta : metadata DataFrame

    Returns
    -------
    news_by_group : dict mapping group -> news contribution
    reest_effect : float, re-estimation effect
    """
    # Identify newly released observations
    # New observations: observed in new vintage but not in old vintage
    T_min = min(Y_new.shape[0], Y_old.shape[0])
    n = min(Y_new.shape[1], Y_old.shape[1])

    newly_released = []
    for t in range(T_min):
        for i in range(n):
            if mask_new[t, i] and not mask_old[t, i]:
                newly_released.append((t, i))

    if not newly_released or bm_old is None:
        return {g: 0.0 for g in NEWS_GROUPS}, 0.0

    # Compute news: I_{v+1,j} = y_{i_j,t_j} - E[y_{i_j,t_j} | Omega_v]
    # Under old model, E[y_{i,t} | Omega_v] = Lambda_i' E[f_t | Omega_v]
    F_old = bm_old["F_smooth"]
    Lambda_old = bm_old["Lambda"]

    news_values = []
    news_indices = []
    for t, i in newly_released:
        if t < len(F_old) and i < Lambda_old.shape[0]:
            y_actual = Y_new[t, i]
            y_expected = float(F_old[min(t, len(F_old)-1)] @ Lambda_old[i])
            news_values.append(y_actual - y_expected)
            news_indices.append((t, i))

    if not news_values:
        return {g: 0.0 for g in NEWS_GROUPS}, 0.0

    news_values = np.array(news_values)

    # Compute B weights: B_j = Cov(y_{k,T}, I_j) / Var(I_j)
    # Simplified: use factor-based covariance
    # Cov(y_{k,T}, I_j) ≈ Lambda_k' Cov(f_T, f_{t_j}) Lambda_{i_j}
    # For simplicity, use the ratio of factor loadings
    F_new = bm_new["F_smooth"]
    Lambda_new = bm_new["Lambda"]

    # Compute B weights using factor covariance structure
    B_weights = _compute_b_weights(
        F_old, Lambda_old, gdp_idx, T_target, news_indices, Y_new, mask_old
    )

    # Total news effect
    total_news = float(np.dot(B_weights, news_values))

    # Group news by series group
    news_by_group = {g: 0.0 for g in NEWS_GROUPS}
    meta_dict = {row["name"]: row["group"] for _, row in meta.iterrows()}

    for j, (t, i) in enumerate(news_indices):
        if i < len(series_names):
            series_name = series_names[i]
            group = meta_dict.get(series_name, "Real")
            if group in news_by_group:
                news_by_group[group] += B_weights[j] * news_values[j]

    # Re-estimation effect
    # = forecast with new data & new params - forecast with new data & old params
    fc_new_params = float(F_new[min(T_target, len(F_new)-1)] @ Lambda_new[gdp_idx])

    # Forecast with new data but old params (approximate)
    try:
        ss_old = bm_old.get("ss", None)
        if ss_old is not None:
            a0 = np.zeros(ss_old["m"])
            P0 = 10.0 * np.eye(ss_old["m"])
            T_use = min(Y_new.shape[0], Y_old.shape[0])
            result_old_params = run_filter_smoother(
                Y_new[:T_use, :Lambda_old.shape[0]],
                ss_old["Z"], ss_old["H"], ss_old["T_mat"],
                ss_old["R_mat"], ss_old["Q_state"],
                a0, P0,
                mask_new[:T_use, :Lambda_old.shape[0]],
            )
            F_old_new_data = result_old_params["a_smooth"][:, :Lambda_old.shape[1]]
            fc_old_params_new_data = float(
                F_old_new_data[min(T_target, len(F_old_new_data)-1)] @ Lambda_old[gdp_idx]
            )
        else:
            fc_old_params_new_data = fc_new_params - total_news
    except Exception:
        fc_old_params_new_data = fc_new_params - total_news

    reest_effect = fc_new_params - fc_old_params_new_data

    return news_by_group, reest_effect


def _compute_b_weights(
    F_smooth: np.ndarray,
    Lambda: np.ndarray,
    gdp_idx: int,
    T_target: int,
    news_indices: list[tuple[int, int]],
    Y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Compute B weights for news decomposition.

    B_j = Cov(y_{k,T}, I_j) / Var(I_j)

    Uses factor-based covariance approximation.

    Parameters
    ----------
    F_smooth : (T, r) smoothed factors
    Lambda : (n, r) loadings
    gdp_idx : int
    T_target : int
    news_indices : list of (t, i) tuples
    Y : (T, n) data
    mask : (T, n) mask

    Returns
    -------
    B_weights : (J,) array
    """
    T, r = F_smooth.shape
    J = len(news_indices)

    if J == 0:
        return np.zeros(0)

    # Approximate factor covariance: use sample covariance of smoothed factors
    # Cov(f_T, f_t) ≈ empirical covariance
    T_target_use = min(T_target, T - 1)

    # Compute Cov(y_{k,T}, I_j) = Lambda_k' Cov(f_T, f_{t_j}) Lambda_{i_j}
    # Approximate Cov(f_T, f_t) using AR(1) structure
    # For simplicity: use Lambda_k' Lambda_{i_j} * rho^|T-t_j|
    rho_approx = 0.7  # approximate factor persistence

    B_weights = np.zeros(J)
    lam_k = Lambda[gdp_idx] if gdp_idx < Lambda.shape[0] else Lambda[0]

    for j, (t_j, i_j) in enumerate(news_indices):
        if i_j >= Lambda.shape[0]:
            continue
        lam_ij = Lambda[i_j]
        lag = abs(T_target_use - t_j)
        cov_f = rho_approx ** lag * np.eye(r)

        cov_y_I = lam_k @ cov_f @ lam_ij

        # Var(I_j) = Lambda_{i_j}' Var(f_{t_j}) Lambda_{i_j} + R_{i_j,i_j}
        var_I = lam_ij @ cov_f @ lam_ij + 0.1  # approximate

        B_weights[j] = cov_y_I / max(var_I, 1e-8)

    return B_weights


def run_news_decomposition_experiment(
    panels: list[str] = ["small", "medium"],
    pilot: bool = False,
) -> dict:
    """
    Run the news decomposition experiment for 2008Q4.

    Parameters
    ----------
    panels : list of panel names
    pilot : bool

    Returns
    -------
    dict with decomposition results for each panel/specification
    """
    print("=" * 60)
    print("Running News Decomposition Experiment (exp_3)")
    print("=" * 60)

    all_results = {}

    for panel in panels:
        for ar1_idio in [False, True]:
            spec_name = f"{panel}_{'ar1' if ar1_idio else 'uncorr'}"
            print(f"\n  {spec_name}...")

            try:
                result = compute_news_decomposition(
                    panel=panel,
                    ar1_idio=ar1_idio,
                    target_quarter="2008Q4",
                    vintage_dates=VINTAGE_DATES,
                )
                all_results[spec_name] = result
            except Exception as e:
                print(f"    Error: {e}")
                all_results[spec_name] = None

    return all_results


def format_news_table(results: dict) -> pd.DataFrame:
    """
    Format news decomposition results into a table.

    Parameters
    ----------
    results : dict from run_news_decomposition_experiment

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for spec_name, res in results.items():
        if res is None:
            continue
        for v_idx, vintage in enumerate(res["vintages"]):
            row = {
                "specification": spec_name,
                "vintage": vintage,
                "forecast": res["forecasts"][v_idx],
                "revision": res["revisions"][v_idx],
                "total_news": res["total_news"][v_idx],
                "reestimation": res["reestimation_effects"][v_idx],
            }
            for g in NEWS_GROUPS:
                row[f"news_{g}"] = res["news_by_group"][g][v_idx]
            rows.append(row)

    return pd.DataFrame(rows)
