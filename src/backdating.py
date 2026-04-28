"""
Backdating Euro Area GDP Using Smoother-Based Estimates (Experiment 4).

Reproduces the paper's backdating exercise by:
1. Truncating historical GDP observations before March 2001
2. Using the Kalman smoother to infer missing GDP history from other indicators

# Custom — Context7 found no library equivalent for this specific
# backdating exercise with mixed-frequency dynamic factor models.
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
    recursive_standardize,
)
from .bm_estimator import em_bm

warnings.filterwarnings("ignore")

# GDP truncation date: delete all GDP before March 2001
GDP_TRUNCATION_DATE = pd.Timestamp("2001-03-01")

# Best ex-post parameterizations for backdating (AR(1) idio spec)
BACKDATING_PARAMS = {
    "small": (4, 2),   # r=4, p=2
    "medium": (5, 2),  # r=5, p=2
    "large": (5, 2),   # r=5, p=2
}


def run_backdating_experiment(
    panels: list[str] = ["small", "medium", "large"],
    pilot: bool = False,
) -> dict:
    """
    Run the backdating experiment.

    For each panel, estimates the AR(1)-idiosyncratic dynamic factor model
    with GDP observations before March 2001 set to missing, then uses the
    Kalman smoother to reconstruct the missing GDP history.

    Parameters
    ----------
    panels : list of panel names
    pilot : bool

    Returns
    -------
    dict with backdated GDP estimates for each panel
    """
    print("=" * 60)
    print("Running Backdating Experiment (exp_4)")
    print("=" * 60)

    # Generate synthetic euro area data
    df_raw = generate_synthetic_euro_area_data(
        start_date="1993-01",
        end_date="2009-10",
        seed=42,
    )
    meta = get_metadata_df()

    # Full sample end date
    end_date = pd.Timestamp("2009-10-01")

    all_results = {}

    for panel in panels:
        print(f"\n  Panel: {panel}...")

        r_best, p_best = BACKDATING_PARAMS[panel]

        try:
            # Prepare full-sample data
            Y, mask, series_names = prepare_panel_for_estimation(
                df_raw, panel, end_date, meta
            )

            # Find GDP series index
            gdp_indices = [i for i, s in enumerate(series_names)
                          if "GDP" in s.upper() and "US" not in s.upper()]
            gdp_idx = gdp_indices[0] if gdp_indices else 0

            # Get date index for the estimation sample
            dates = pd.date_range(start="1993-01", end="2009-10", freq="MS")
            dates = dates[:len(Y)]

            # Truncate GDP before March 2001
            truncation_idx = np.searchsorted(dates, GDP_TRUNCATION_DATE)
            mask_backdating = mask.copy()
            mask_backdating[:truncation_idx, gdp_idx] = False

            # Save original GDP before truncation (for accuracy evaluation)
            gdp_original = Y[:, gdp_idx].copy()

            # Also set Y to NaN for truncated GDP
            Y_backdating = Y.copy()
            Y_backdating[:truncation_idx, gdp_idx] = np.nan

            # Estimate model with truncated GDP
            r_use = min(r_best, Y.shape[1] - 1, 5)

            def _run_bm(r_val: int, p_val: int, ar1: bool) -> dict:
                return em_bm(
                    Y_backdating, mask_backdating,
                    r=r_val, p=p_val,
                    ar1_idio=ar1,
                    max_iter=100, tol=1e-4, kappa=1e-6,
                )

            def _is_valid(res: dict, gdp_i: int) -> bool:
                """Check that F_smooth and Lambda are finite and Lambda is non-zero."""
                F = res["F_smooth"]
                L = res["Lambda"]
                return (
                    np.all(np.isfinite(F))
                    and np.all(np.isfinite(L))
                    and np.linalg.norm(L[gdp_i]) > 1e-8
                )

            bm_res = None
            # Try AR(1) idio first; fall back progressively
            for ar1_flag, r_val, p_val in [
                (True,  r_use,          p_best),
                (False, r_use,          p_best),
                (False, min(r_use, 3),  1),
                (False, min(r_use, 2),  1),
            ]:
                try:
                    candidate = _run_bm(r_val, p_val, ar1_flag)
                    if _is_valid(candidate, gdp_idx):
                        bm_res = candidate
                        r_use = r_val
                        p_best = p_val
                        break
                except Exception:
                    continue

            if bm_res is None:
                raise RuntimeError("All BM fallbacks failed for backdating")

            # Extract smoothed GDP estimates
            F_smooth = bm_res["F_smooth"]
            Lambda = bm_res["Lambda"]

            # Reconstruct GDP: chi_{gdp,t} = Lambda_{gdp}' f_t
            gdp_reconstructed = F_smooth @ Lambda[gdp_idx]  # (T,)

            # Actual GDP (original, before truncation, for comparison)
            gdp_actual = gdp_original

            # Dates for plotting
            result = {
                "dates": dates,
                "gdp_actual": gdp_actual,
                "gdp_reconstructed": gdp_reconstructed,
                "truncation_date": GDP_TRUNCATION_DATE,
                "truncation_idx": truncation_idx,
                "series_names": series_names,
                "gdp_idx": gdp_idx,
                "r": r_use,
                "p": p_best,
                "converged": bm_res["converged"],
                "n_iter": bm_res["n_iter"],
            }

            all_results[panel] = result
            print(f"    Converged: {bm_res['converged']}, Iterations: {bm_res['n_iter']}")

        except Exception as e:
            print(f"    Error: {e}")
            all_results[panel] = None

    return all_results


def compute_backdating_accuracy(result: dict) -> dict:
    """
    Compute accuracy metrics for backdated GDP estimates.

    Parameters
    ----------
    result : dict from run_backdating_experiment for one panel

    Returns
    -------
    dict with accuracy metrics
    """
    if result is None:
        return {}

    trunc_idx = result["truncation_idx"]
    gdp_actual = result["gdp_actual"]
    gdp_recon = result["gdp_reconstructed"]

    # Pre-truncation period (where GDP was missing)
    actual_pre = gdp_actual[:trunc_idx]
    recon_pre = gdp_recon[:trunc_idx]

    # Remove NaN
    valid = ~np.isnan(actual_pre) & ~np.isnan(recon_pre)

    if valid.sum() == 0:
        return {
            "rmse_pre_truncation": np.nan,
            "corr_pre_truncation": np.nan,
            "rmse_post_truncation": np.nan,
            "corr_post_truncation": np.nan,
            "n_missing_gdp": 0,
        }

    rmse_pre = float(np.sqrt(np.mean((actual_pre[valid] - recon_pre[valid]) ** 2)))
    corr_pre = float(np.corrcoef(actual_pre[valid], recon_pre[valid])[0, 1])

    # Post-truncation period (where GDP was observed)
    actual_post = gdp_actual[trunc_idx:]
    recon_post = gdp_recon[trunc_idx:]
    valid_post = ~np.isnan(actual_post) & ~np.isnan(recon_post)

    if valid_post.sum() > 0:
        rmse_post = float(np.sqrt(np.mean((actual_post[valid_post] - recon_post[valid_post]) ** 2)))
        corr_post = float(np.corrcoef(actual_post[valid_post], recon_post[valid_post])[0, 1])
    else:
        rmse_post = np.nan
        corr_post = np.nan

    return {
        "rmse_pre_truncation": rmse_pre,
        "corr_pre_truncation": corr_pre,
        "rmse_post_truncation": rmse_post,
        "corr_post_truncation": corr_post,
        "n_missing_gdp": int(valid.sum()),
    }
