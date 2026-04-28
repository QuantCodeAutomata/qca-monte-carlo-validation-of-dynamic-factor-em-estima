"""
Visualization module for all experiments.

Creates plots for:
- Monte Carlo results (factor recovery, forecast accuracy, AR estimation)
- Empirical nowcasting results
- News decomposition (Figure 1 style)
- Backdating results (Figure 2 style)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional

# Use seaborn style for all plots
sns.set_theme(style="whitegrid", palette="colorblind")
RESULTS_DIR = Path("results")


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to results directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_substudy_a_results(df: pd.DataFrame) -> None:
    """
    Plot Substudy A: Factor-space recovery results.

    Shows BM trace R^2 and BM/RT, BM/SW ratios across parameter cells.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: BM R^2 by missing share and n
    ax = axes[0]
    for n_val in df["n"].unique():
        subset = df[df["n"] == n_val].groupby("missing_share")["bm_r2"].mean()
        ax.plot(subset.index * 100, subset.values, marker="o", label=f"n={n_val}")
    ax.set_xlabel("Missing Share (%)")
    ax.set_ylabel("Avg Trace R²")
    ax.set_title("BM Factor Recovery by Missing Share")
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 2: BM/RT ratio by missing share
    ax = axes[1]
    for n_val in df["n"].unique():
        subset = df[df["n"] == n_val].groupby("missing_share")["bm_rt_ratio"].mean()
        ax.plot(subset.index * 100, subset.values, marker="s", label=f"n={n_val}")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Missing Share (%)")
    ax.set_ylabel("BM/RT Ratio")
    ax.set_title("BM vs RT: Factor Recovery Ratio")
    ax.legend()

    # Plot 3: BM/SW ratio by missing share
    ax = axes[2]
    for n_val in df["n"].unique():
        subset = df[df["n"] == n_val].groupby("missing_share")["bm_sw_ratio"].mean()
        ax.plot(subset.index * 100, subset.values, marker="^", label=f"n={n_val}")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Missing Share (%)")
    ax.set_ylabel("BM/SW Ratio")
    ax.set_title("BM vs SW: Factor Recovery Ratio")
    ax.legend()

    fig.suptitle("Substudy A: Factor-Space Recovery (Monte Carlo)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "substudy_a_factor_recovery.png")


def plot_substudy_a_heatmap(df: pd.DataFrame) -> None:
    """Plot heatmap of BM trace R^2 across parameter combinations."""
    # Aggregate over replications
    pivot_data = df.groupby(["n", "missing_share"])["bm_r2"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_data,
        annot=True, fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Missing Share")
    ax.set_ylabel("n (cross-sectional dimension)")
    ax.set_title("BM Trace R² Heatmap: n vs Missing Share")
    plt.tight_layout()
    save_figure(fig, "substudy_a_heatmap.png")


def plot_substudy_b_results(df: pd.DataFrame) -> None:
    """
    Plot Substudy B: Ragged-edge forecast accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: BM accuracy by horizon and n
    ax = axes[0]
    for n_val in df["n"].unique():
        subset = df[df["n"] == n_val].groupby("horizon")["bm_acc"].mean()
        ax.plot(subset.index, subset.values, marker="o", label=f"n={n_val}")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Relative Forecast Accuracy")
    ax.set_title("BM Ragged-Edge Forecast Accuracy")
    ax.legend()
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["hor1", "hor2", "hor3", "hor4"])

    # Plot 2: BM/RT and BM/SW ratios by horizon
    ax = axes[1]
    bm_rt = df.groupby("horizon")["bm_rt_ratio"].mean()
    bm_sw = df.groupby("horizon")["bm_sw_ratio"].mean()
    ax.plot(bm_rt.index, bm_rt.values, marker="s", label="BM/RT")
    ax.plot(bm_sw.index, bm_sw.values, marker="^", label="BM/SW")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Ratio")
    ax.set_title("BM vs Benchmarks: Forecast Accuracy Ratios")
    ax.legend()
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["hor1", "hor2", "hor3", "hor4"])

    fig.suptitle("Substudy B: Ragged-Edge Forecast Accuracy (Monte Carlo)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "substudy_b_forecast_accuracy.png")


def plot_substudy_c_results(df: pd.DataFrame) -> None:
    """
    Plot Substudy C: AR(1) idiosyncratic coefficient estimation.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for miss in df["missing_share"].unique():
        subset = df[df["missing_share"] == miss].sort_values("T")
        label = f"Missing={miss:.0%}"
        ax.plot(subset["T"], subset["mean_mae"], marker="o", label=label)

        # Plot paper targets if available
        if "paper_target" in subset.columns:
            ax.plot(
                subset["T"], subset["paper_target"],
                marker="x", linestyle="--",
                label=f"Paper target ({miss:.0%})",
            )

    ax.set_xlabel("Sample Size T")
    ax.set_ylabel("Mean Absolute Error of α̂")
    ax.set_title("Substudy C: AR(1) Idiosyncratic Coefficient Estimation")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    save_figure(fig, "substudy_c_ar1_estimation.png")


def plot_nowcasting_results(results: dict) -> None:
    """
    Plot empirical nowcasting results (Table 5 style).
    """
    from .nowcasting import HORIZONS, format_results_table

    df_table = format_results_table(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Average RMSFE by specification
    ax = axes[0]
    specs = df_table["specification"].tolist()
    avg_rmsfe = df_table["avg_rmsfe"].tolist()
    colors = ["steelblue" if "benchmark" not in s else "coral" for s in specs]
    bars = ax.bar(range(len(specs)), avg_rmsfe, color=colors)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(specs, rotation=45, ha="right")
    ax.set_ylabel("Average RMSFE")
    ax.set_title("Nowcasting: Average RMSFE by Specification")
    ax.axhline(y=0.22, color="green", linestyle="--", alpha=0.5, label="Paper target ~0.22")
    ax.legend()

    # Plot 2: RMSFE by horizon for selected specifications
    ax = axes[1]
    for _, row in df_table.iterrows():
        spec = row["specification"]
        if "benchmark" in spec:
            continue
        horizon_vals = [row.get(h, np.nan) for h in HORIZONS]
        if any(not np.isnan(v) for v in horizon_vals):
            ax.plot(range(len(HORIZONS)), horizon_vals, marker="o", label=spec)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, rotation=45, ha="right")
    ax.set_ylabel("RMSFE")
    ax.set_title("Nowcasting: RMSFE by Forecast Horizon")
    ax.legend(fontsize=7)

    fig.suptitle("Empirical Nowcasting Results (exp_2)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "nowcasting_results.png")


def plot_news_decomposition(results: dict) -> None:
    """
    Plot news decomposition results (Figure 1 style).
    """
    from .news_decomposition import NEWS_GROUPS

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    colors = {
        "Real": "steelblue",
        "Survey": "orange",
        "Financial": "green",
        "US": "red",
        "reestimation": "purple",
    }

    for ax, (spec_name, res) in zip(axes, results.items()):
        if res is None:
            ax.set_title(f"{spec_name}: No results")
            continue

        vintages = res["vintages"]
        forecasts = res["forecasts"]
        revisions = res["revisions"]

        x = range(len(vintages))

        # Plot forecast path
        ax.plot(x, forecasts, "k-o", linewidth=2, label="GDP Nowcast", zorder=5)

        # Plot stacked bar for news contributions
        bottom_pos = np.zeros(len(vintages))
        bottom_neg = np.zeros(len(vintages))

        for group in NEWS_GROUPS:
            news = np.array(res["news_by_group"][group])
            pos_news = np.where(news > 0, news, 0)
            neg_news = np.where(news < 0, news, 0)
            ax.bar(x, pos_news, bottom=bottom_pos, label=f"{group} News",
                   color=colors.get(group, "gray"), alpha=0.7)
            ax.bar(x, neg_news, bottom=bottom_neg,
                   color=colors.get(group, "gray"), alpha=0.7)
            bottom_pos += pos_news
            bottom_neg += neg_news

        # Re-estimation effect
        reest = np.array(res["reestimation_effects"])
        ax.bar(x, np.where(reest > 0, reest, 0), bottom=bottom_pos,
               label="Re-estimation", color=colors["reestimation"], alpha=0.5)
        ax.bar(x, np.where(reest < 0, reest, 0), bottom=bottom_neg,
               color=colors["reestimation"], alpha=0.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(vintages, rotation=45, ha="right")
        ax.set_ylabel("GDP Nowcast / News Contribution")
        ax.set_title(f"News Decomposition: {spec_name} (2008Q4 GDP)")
        ax.legend(fontsize=7, loc="upper left")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    fig.suptitle("Model-Based News Decomposition of GDP Forecast Revisions (exp_3)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "news_decomposition.png")


def plot_backdating_results(results: dict) -> None:
    """
    Plot backdating results (Figure 2 style).
    """
    n_panels = sum(1 for v in results.values() if v is not None)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for panel_name, result in results.items():
        if result is None:
            continue

        ax = axes[ax_idx]
        dates = result["dates"]
        gdp_actual = result["gdp_actual"]
        gdp_recon = result["gdp_reconstructed"]
        trunc_date = result["truncation_date"]
        trunc_idx = result["truncation_idx"]

        # Plot actual GDP
        valid_actual = ~np.isnan(gdp_actual)
        ax.plot(dates[valid_actual], gdp_actual[valid_actual],
                "b-", linewidth=2, label="Actual GDP", alpha=0.8)

        # Plot reconstructed GDP
        ax.plot(dates, gdp_recon, "r--", linewidth=2,
                label="Backdated GDP (smoother)", alpha=0.8)

        # Mark truncation date
        ax.axvline(x=trunc_date, color="gray", linestyle=":", alpha=0.7,
                   label=f"Truncation: {trunc_date.strftime('%Y-%m')}")

        # Shade pre-truncation region
        ax.axvspan(dates[0], trunc_date, alpha=0.1, color="red",
                   label="Missing GDP region")

        ax.set_xlabel("Date")
        ax.set_ylabel("GDP (standardized)")
        ax.set_title(f"Backdating: {panel_name.capitalize()} Panel\n"
                     f"(r={result['r']}, p={result['p']})")
        ax.legend(fontsize=8)

        ax_idx += 1

    fig.suptitle("Backdating Euro Area GDP (exp_4)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "backdating_results.png")


def plot_em_convergence(log_lik_history: list[float], title: str = "EM Convergence") -> None:
    """Plot EM algorithm convergence."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(log_lik_history)), log_lik_history, "b-o", markersize=3)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, "em_convergence.png")
