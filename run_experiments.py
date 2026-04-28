"""
Main experiment runner for all four experiments.

Runs:
  exp_1: Monte Carlo validation of dynamic-factor EM estimation
  exp_2: Empirical euro area GDP nowcasting
  exp_3: Model-based news decomposition
  exp_4: Backdating euro area GDP

Results are saved to the results/ directory.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.monte_carlo import run_all_substudies
from src.nowcasting import run_nowcasting_experiment, format_results_table
from src.news_decomposition import run_news_decomposition_experiment, format_news_table
from src.backdating import run_backdating_experiment, compute_backdating_accuracy
from src.visualization import (
    plot_substudy_a_results,
    plot_substudy_a_heatmap,
    plot_substudy_b_results,
    plot_substudy_c_results,
    plot_nowcasting_results,
    plot_news_decomposition,
    plot_backdating_results,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Use pilot mode for faster execution (set to False for full 500-rep runs)
# Full runs take several hours; pilot runs take minutes
PILOT_MODE = True
N_REPS = 10 if PILOT_MODE else 500


def run_exp1() -> dict:
    """Run Monte Carlo experiment (exp_1)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Monte Carlo Validation of Dynamic-Factor EM Estimation")
    print("=" * 70)
    print(f"Mode: {'PILOT' if PILOT_MODE else 'FULL'} ({N_REPS} replications)")

    t0 = time.time()
    results = run_all_substudies(n_reps=N_REPS, pilot=PILOT_MODE)
    elapsed = time.time() - t0
    print(f"\nExp 1 completed in {elapsed:.1f}s")

    # Save results
    df_a = results["substudy_a"]
    df_b = results["substudy_b"]
    df_c = results["substudy_c"]

    df_a.to_csv(RESULTS_DIR / "substudy_a_factor_recovery.csv", index=False)
    df_b.to_csv(RESULTS_DIR / "substudy_b_forecast_accuracy.csv", index=False)
    df_c.to_csv(RESULTS_DIR / "substudy_c_ar1_estimation.csv", index=False)

    # Generate plots
    if len(df_a) > 0:
        plot_substudy_a_results(df_a)
        plot_substudy_a_heatmap(df_a)
    if len(df_b) > 0:
        plot_substudy_b_results(df_b)
    if len(df_c) > 0:
        plot_substudy_c_results(df_c)

    return results


def run_exp2() -> dict:
    """Run empirical nowcasting experiment (exp_2)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Empirical Euro Area GDP Nowcasting")
    print("=" * 70)

    t0 = time.time()
    results = run_nowcasting_experiment(
        panels=["small", "medium"] if PILOT_MODE else ["small", "medium", "large"],
        ar1_idio_options=[False, True],
        pilot=PILOT_MODE,
    )
    elapsed = time.time() - t0
    print(f"\nExp 2 completed in {elapsed:.1f}s")

    # Format and save results
    df_table = format_results_table(results)
    df_table.to_csv(RESULTS_DIR / "nowcasting_results.csv", index=False)

    # Generate plots
    plot_nowcasting_results(results)

    return results


def run_exp3() -> dict:
    """Run news decomposition experiment (exp_3)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Model-Based News Decomposition")
    print("=" * 70)

    t0 = time.time()
    results = run_news_decomposition_experiment(
        panels=["small"] if PILOT_MODE else ["small", "medium"],
        pilot=PILOT_MODE,
    )
    elapsed = time.time() - t0
    print(f"\nExp 3 completed in {elapsed:.1f}s")

    # Format and save results
    df_news = format_news_table(results)
    df_news.to_csv(RESULTS_DIR / "news_decomposition.csv", index=False)

    # Generate plots
    plot_news_decomposition(results)

    return results


def run_exp4() -> dict:
    """Run backdating experiment (exp_4)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Backdating Euro Area GDP")
    print("=" * 70)

    t0 = time.time()
    results = run_backdating_experiment(
        panels=["small", "medium"] if PILOT_MODE else ["small", "medium", "large"],
        pilot=PILOT_MODE,
    )
    elapsed = time.time() - t0
    print(f"\nExp 4 completed in {elapsed:.1f}s")

    # Compute accuracy metrics
    accuracy_rows = []
    for panel, result in results.items():
        if result is not None:
            acc = compute_backdating_accuracy(result)
            acc["panel"] = panel
            accuracy_rows.append(acc)

    if accuracy_rows:
        df_acc = pd.DataFrame(accuracy_rows)
        df_acc.to_csv(RESULTS_DIR / "backdating_accuracy.csv", index=False)

    # Generate plots
    plot_backdating_results(results)

    return results


def save_results_markdown(
    exp1_results: dict,
    exp2_results: dict,
    exp3_results: dict,
    exp4_results: dict,
) -> None:
    """Save all results to RESULTS.md."""
    lines = [
        "# Experiment Results",
        "",
        f"**Mode**: {'Pilot' if PILOT_MODE else 'Full'} ({N_REPS} Monte Carlo replications)",
        "",
        "---",
        "",
        "## Experiment 1: Monte Carlo Validation of Dynamic-Factor EM Estimation",
        "",
        "### Substudy A: Factor-Space Recovery",
        "",
    ]

    # Substudy A summary
    df_a = exp1_results.get("substudy_a", pd.DataFrame())
    if len(df_a) > 0:
        # Summary table: BM R^2 by missing share
        summary_a = df_a.groupby("missing_share")[["bm_r2", "rt_r2", "sw_r2",
                                                     "bm_rt_ratio", "bm_sw_ratio"]].mean()
        lines.append("Average Trace R² by Missing Share:")
        lines.append("")
        lines.append("| Missing Share | BM R² | RT R² | SW R² | BM/RT | BM/SW |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for miss, row in summary_a.iterrows():
            lines.append(
                f"| {miss:.0%} | {row['bm_r2']:.3f} | {row['rt_r2']:.3f} | "
                f"{row['sw_r2']:.3f} | {row['bm_rt_ratio']:.3f} | {row['bm_sw_ratio']:.3f} |"
            )
        lines.append("")

        # By n
        summary_n = df_a.groupby("n")[["bm_r2", "rt_r2", "sw_r2"]].mean()
        lines.append("Average Trace R² by Cross-Sectional Dimension n:")
        lines.append("")
        lines.append("| n | BM R² | RT R² | SW R² |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for n_val, row in summary_n.iterrows():
            lines.append(
                f"| {n_val} | {row['bm_r2']:.3f} | {row['rt_r2']:.3f} | {row['sw_r2']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "### Substudy B: Ragged-Edge Forecast Accuracy",
        "",
    ])

    df_b = exp1_results.get("substudy_b", pd.DataFrame())
    if len(df_b) > 0:
        summary_b = df_b.groupby("horizon")[["bm_acc", "rt_acc", "sw_acc"]].mean()
        lines.append("Average Relative Forecast Accuracy by Horizon:")
        lines.append("")
        lines.append("| Horizon | BM Acc | RT Acc | SW Acc |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for hor, row in summary_b.iterrows():
            lines.append(
                f"| hor{hor} | {row['bm_acc']:.3f} | {row['rt_acc']:.3f} | {row['sw_acc']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "### Substudy C: AR(1) Idiosyncratic Coefficient Estimation",
        "",
    ])

    df_c = exp1_results.get("substudy_c", pd.DataFrame())
    if len(df_c) > 0:
        lines.append("Mean Absolute Error of α̂ by T and Missing Share:")
        lines.append("")
        lines.append("| T | Missing | Mean MAE | Paper Target |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for _, row in df_c.iterrows():
            target_str = f"{row['paper_target']:.3f}" if not pd.isna(row.get('paper_target', np.nan)) else "N/A"
            mae_str = f"{row['mean_mae']:.3f}" if not pd.isna(row['mean_mae']) else "N/A"
            lines.append(
                f"| {int(row['T'])} | {row['missing_share']:.0%} | {mae_str} | {target_str} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Experiment 2: Empirical Euro Area GDP Nowcasting",
        "",
    ])

    # Nowcasting results
    df_now = format_results_table(exp2_results)
    if len(df_now) > 0:
        lines.append("RMSFE Results (Table 5 style):")
        lines.append("")
        lines.append("| Specification | Avg RMSFE | Best r | Best p |")
        lines.append("|:---|:---:|:---:|:---:|")
        for _, row in df_now.iterrows():
            r_str = str(int(row["best_r"])) if not pd.isna(row.get("best_r", np.nan)) else "N/A"
            p_str = str(int(row["best_p"])) if not pd.isna(row.get("best_p", np.nan)) else "N/A"
            avg_str = f"{row['avg_rmsfe']:.4f}" if not pd.isna(row['avg_rmsfe']) else "N/A"
            lines.append(f"| {row['specification']} | {avg_str} | {r_str} | {p_str} |")
        lines.append("")
        lines.append("**Paper targets**: Small ~0.22-0.23, Medium ~0.22, Large ~0.24, AR ~0.30, Mean ~0.31")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Experiment 3: Model-Based News Decomposition",
        "",
    ])

    df_news = format_news_table(exp3_results)
    if len(df_news) > 0:
        lines.append("News Decomposition for 2008Q4 GDP:")
        lines.append("")
        lines.append("| Specification | Vintage | Forecast | Revision | Real News | Survey News | Financial News | US News | Re-estimation |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for _, row in df_news.iterrows():
            lines.append(
                f"| {row['specification']} | {row['vintage']} | "
                f"{row['forecast']:.4f} | {row['revision']:.4f} | "
                f"{row.get('news_Real', 0):.4f} | {row.get('news_Survey', 0):.4f} | "
                f"{row.get('news_Financial', 0):.4f} | {row.get('news_US', 0):.4f} | "
                f"{row['reestimation']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Experiment 4: Backdating Euro Area GDP",
        "",
    ])

    for panel, result in exp4_results.items():
        if result is not None:
            acc = compute_backdating_accuracy(result)
            lines.append(f"### {panel.capitalize()} Panel (r={result['r']}, p={result['p']})")
            lines.append("")
            lines.append(f"- Converged: {result['converged']}")
            lines.append(f"- Iterations: {result['n_iter']}")
            lines.append(f"- RMSE (pre-truncation): {acc.get('rmse_pre_truncation', 'N/A'):.4f}" if isinstance(acc.get('rmse_pre_truncation'), float) else f"- RMSE (pre-truncation): N/A")
            lines.append(f"- Correlation (pre-truncation): {acc.get('corr_pre_truncation', 'N/A'):.4f}" if isinstance(acc.get('corr_pre_truncation'), float) else f"- Correlation (pre-truncation): N/A")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- All experiments use synthetic data generated from the paper's DGP",
        "- The actual euro area dataset is not publicly available; synthetic data mimics its structure",
        f"- Monte Carlo replications: {N_REPS} ({'pilot' if PILOT_MODE else 'full'} mode)",
        "- kappa = 1e-6 for AR(1) idiosyncratic measurement noise",
        "- Phi_{ij} = tau^|i-j| * (1-alpha^2) * sqrt(gamma_i * gamma_j)",
        "- EM convergence criterion: |c_j| < 1e-4",
        "",
    ])

    with open(RESULTS_DIR / "RESULTS.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {RESULTS_DIR / 'RESULTS.md'}")


def main():
    """Run all experiments and save results."""
    print("Dynamic Factor Model EM Estimation - Experiment Runner")
    print("=" * 70)

    total_start = time.time()

    # Run all experiments
    exp1_results = run_exp1()
    exp2_results = run_exp2()
    exp3_results = run_exp3()
    exp4_results = run_exp4()

    # Save combined results
    save_results_markdown(exp1_results, exp2_results, exp3_results, exp4_results)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
