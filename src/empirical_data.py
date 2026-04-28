"""
Euro Area Empirical Dataset Handling.

Implements the data preparation pipeline for the empirical nowcasting
experiment, including:
- Series metadata from Appendix A
- Transformation rules (log, diff)
- Mariano-Murasawa quarterly aggregation
- Pseudo-real-time publication-delay masks
- Recursive standardization

Since the actual euro area dataset is not publicly available in this
implementation, we generate a synthetic dataset that mimics the structure
described in Appendix A for demonstration purposes.

# Custom — Context7 found no library equivalent for this specific
# euro area dataset construction and mixed-frequency treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


# ============================================================
# Appendix A Metadata
# ============================================================

# Series metadata: (name, group, is_quarterly, log_transform, diff, pub_lag_m1, pub_lag_m2, pub_lag_m3, small, medium, large, br)
# pub_lag_mx = publication lag in months within quarter month x (1=first, 2=second, 3=third)
# Positive lag means data is available that many months after the reference period

APPENDIX_A_METADATA = [
    # Real activity
    {"name": "IP_total_excl_construction", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": True, "medium": True, "large": True, "br": True},
    {"name": "New_passenger_cars", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": True, "medium": True, "large": True, "br": False},
    {"name": "New_orders", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": True, "medium": True, "large": True, "br": True},
    {"name": "Retail_turnover_deflated", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": True, "medium": True, "large": True, "br": False},
    {"name": "Employment", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": True, "medium": True, "large": True, "br": False},
    {"name": "GDP", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": True, "medium": True, "large": True, "br": True},
    # Surveys
    {"name": "EC_ESI", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": True, "medium": True, "large": True, "br": True},
    {"name": "PMI_Manufacturing", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": True, "medium": True, "large": True, "br": True},
    {"name": "EC_Capacity_Utilisation", "group": "Survey", "quarterly": True,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": True, "medium": True, "large": True, "br": False},
    # Labour
    {"name": "Unemployment_rate", "group": "Real", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": True, "medium": True, "large": True, "br": False},
    # Trade
    {"name": "Extra_EA_trade_export_value", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": True, "medium": True, "large": True, "br": False},
    # Financial
    {"name": "DJ_Euro_Stock_Price_Index", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": True, "medium": True, "large": True, "br": False},
    {"name": "World_raw_materials_prices_euro", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": True, "medium": True, "large": True, "br": False},
    # US
    {"name": "US_GDP", "group": "US", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": True, "medium": True, "large": True, "br": False},
    # Additional medium/large series (simplified representation)
    {"name": "IP_consumer_goods", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "IP_capital_goods", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "IP_intermediate_goods", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Construction_output", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Retail_food", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Retail_non_food", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "EC_consumer_confidence", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "EC_industry_confidence", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "EC_services_confidence", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "EC_retail_confidence", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "PMI_Services", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "PMI_Composite", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Ifo_business_climate", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "ZEW_economic_sentiment", "group": "Survey", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Employment_services", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Compensation_employees", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Private_consumption", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Gross_fixed_capital_formation", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Government_consumption", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Exports_goods_services", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Imports_goods_services", "group": "Real", "quarterly": True,
     "log": True, "diff": True, "pub_lag_m1": 3, "pub_lag_m2": 3, "pub_lag_m3": 3,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "HICP_total", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "HICP_core", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "PPI", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "M1", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "M3", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Credit_private_sector", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 1, "pub_lag_m2": 1, "pub_lag_m3": 1,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Short_term_interest_rate", "group": "Financial", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Long_term_interest_rate", "group": "Financial", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Term_spread", "group": "Financial", "quarterly": False,
     "log": False, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "EUR_USD_exchange_rate", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "Oil_price_USD", "group": "Financial", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 0, "pub_lag_m2": 0, "pub_lag_m3": 0,
     "small": False, "medium": True, "large": True, "br": False},
    {"name": "US_IP", "group": "US", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": True, "large": True, "br": False},
]

# Extend to ~101 series for Large panel by adding more series
_EXTRA_LARGE_SERIES = [
    {"name": f"Large_series_{i}", "group": "Real", "quarterly": False,
     "log": True, "diff": True, "pub_lag_m1": 2, "pub_lag_m2": 2, "pub_lag_m3": 2,
     "small": False, "medium": False, "large": True, "br": False}
    for i in range(55)
]
APPENDIX_A_METADATA = APPENDIX_A_METADATA + _EXTRA_LARGE_SERIES


def get_metadata_df() -> pd.DataFrame:
    """Return metadata as a DataFrame."""
    return pd.DataFrame(APPENDIX_A_METADATA)


def get_panel_series(panel: str) -> list[str]:
    """
    Get series names for a given panel.

    Parameters
    ----------
    panel : str, one of 'small', 'medium', 'large'

    Returns
    -------
    list of series names
    """
    meta = get_metadata_df()
    return meta[meta[panel.lower()]]["name"].tolist()


def mariano_murasawa_weights() -> np.ndarray:
    """
    Return Mariano-Murasawa aggregation weights [1, 2, 3, 2, 1].

    These weights approximate quarterly growth from monthly growth rates:
    bar{y}_t^Q = (1+L+L^2)^2 y_t^Q = y_t^Q + 2*y_{t-1}^Q + 3*y_{t-2}^Q + 2*y_{t-3}^Q + y_{t-4}^Q
    """
    return np.array([1.0, 2.0, 3.0, 2.0, 1.0])


def generate_synthetic_euro_area_data(
    start_date: str = "1993-01",
    end_date: str = "2009-10",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic euro area data mimicking the structure of Appendix A.

    Since the actual euro area dataset is proprietary, this function generates
    synthetic data with realistic statistical properties for demonstration.

    Parameters
    ----------
    start_date : str, start of sample (YYYY-MM)
    end_date : str, end of sample (YYYY-MM)
    seed : int

    Returns
    -------
    pd.DataFrame with monthly index and series as columns
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    T = len(dates)
    meta = get_metadata_df()
    n = len(meta)

    # Generate correlated factor-based data
    r_true = 3
    rho_f = 0.7

    # Simulate factors
    F = np.zeros((T, r_true))
    F[0] = rng.standard_normal(r_true)
    for t in range(1, T):
        F[t] = rho_f * F[t - 1] + rng.standard_normal(r_true)

    # Generate loadings
    Lambda = rng.standard_normal((n, r_true)) * 0.5

    # Generate idiosyncratic components
    alpha_idio = 0.3
    Epsilon = np.zeros((T, n))
    Epsilon[0] = rng.standard_normal(n) * 0.5
    for t in range(1, T):
        Epsilon[t] = alpha_idio * Epsilon[t - 1] + rng.standard_normal(n) * 0.5

    # Observed data (in "transformed" space)
    Y = F @ Lambda.T + Epsilon

    # Create DataFrame
    df = pd.DataFrame(Y, index=dates, columns=meta["name"].tolist())

    # For quarterly series, set non-quarter-end months to NaN
    for _, row in meta.iterrows():
        if row["quarterly"]:
            # Only keep values at quarter-end months (March, June, Sep, Dec)
            non_quarter_end = ~df.index.month.isin([3, 6, 9, 12])
            df.loc[non_quarter_end, row["name"]] = np.nan

    return df


def apply_publication_lag_mask(
    df: pd.DataFrame,
    vintage_date: pd.Timestamp,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply pseudo-real-time publication delay mask.

    For each series, set observations to NaN if they would not have been
    available at the given vintage date according to the publication lag calendar.

    Parameters
    ----------
    df : pd.DataFrame with monthly index
    vintage_date : pd.Timestamp, the current information date
    meta : pd.DataFrame, series metadata

    Returns
    -------
    pd.DataFrame with publication-lag mask applied
    """
    df_masked = df.copy()

    for _, row in meta.iterrows():
        series_name = row["name"]
        if series_name not in df_masked.columns:
            continue

        # Determine publication lag based on vintage month within quarter
        vintage_month = vintage_date.month
        quarter_month = ((vintage_month - 1) % 3) + 1  # 1, 2, or 3

        if quarter_month == 1:
            pub_lag = row["pub_lag_m1"]
        elif quarter_month == 2:
            pub_lag = row["pub_lag_m2"]
        else:
            pub_lag = row["pub_lag_m3"]

        # Latest available observation date
        latest_available = vintage_date - pd.DateOffset(months=pub_lag)

        # Mask future observations
        future_mask = df_masked.index > latest_available
        df_masked.loc[future_mask, series_name] = np.nan

    return df_masked


def recursive_standardize(
    df: pd.DataFrame,
    current_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Recursively standardize series using only data up to current_end.

    Parameters
    ----------
    df : pd.DataFrame
    current_end : pd.Timestamp

    Returns
    -------
    df_std : standardized DataFrame
    means : pd.Series of means
    stds : pd.Series of standard deviations
    """
    df_current = df[df.index <= current_end]
    means = df_current.mean()
    stds = df_current.std()
    stds = stds.replace(0, 1.0)
    stds = stds.fillna(1.0)

    df_std = (df - means) / stds
    return df_std, means, stds


def prepare_panel_for_estimation(
    df: pd.DataFrame,
    panel: str,
    vintage_date: pd.Timestamp,
    meta: pd.DataFrame,
    start_date: str = "1993-01",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare a panel dataset for model estimation at a given vintage.

    Parameters
    ----------
    df : pd.DataFrame, full dataset
    panel : str, 'small', 'medium', or 'large'
    vintage_date : pd.Timestamp
    meta : pd.DataFrame
    start_date : str

    Returns
    -------
    Y : (T, n) data array
    mask : (T, n) boolean mask
    series_names : list of series names
    """
    # Get panel series
    panel_meta = meta[meta[panel.lower()]]
    series_names = panel_meta["name"].tolist()

    # Filter to available series
    available = [s for s in series_names if s in df.columns]

    # Apply publication lag mask
    df_masked = apply_publication_lag_mask(df[available], vintage_date, panel_meta)

    # Restrict to estimation sample
    start = pd.Timestamp(start_date)
    df_sample = df_masked[(df_masked.index >= start) & (df_masked.index <= vintage_date)]

    # Recursive standardization
    df_std, _, _ = recursive_standardize(df_sample, vintage_date)

    Y = df_std.values.astype(float)
    mask = ~np.isnan(Y)

    return Y, mask, available
