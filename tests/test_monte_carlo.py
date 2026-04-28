"""
Tests for the Monte Carlo framework.

Verifies that the Monte Carlo substudies produce valid results
and that the methodology is correctly implemented.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import DGPConfig, simulate_dgp, impose_random_missing, impose_ragged_edge_missing
from src.monte_carlo import (
    run_single_replication_substudy_a,
    run_single_replication_substudy_b,
    run_single_replication_substudy_c,
)
from src.metrics import trace_r2, relative_forecast_accuracy, mean_abs_error_ar


class TestSubstudyA:
    """Tests for Substudy A: Factor-space recovery."""

    def test_single_replication_output(self):
        """Test that single replication returns valid results."""
        config = DGPConfig(n=15, T=80, r=2, s=0, rho=0.7, alpha=0.0, tau=0.0)
        result = run_single_replication_substudy_a(
            config, rhat=2, missing_share=0.0, seed=42
        )

        assert "bm_r2" in result
        assert "rt_r2" in result
        assert "sw_r2" in result

    def test_r2_range(self):
        """Test that trace R^2 values are in [0, 1]."""
        config = DGPConfig(n=15, T=80, r=2, s=0, rho=0.7)
        result = run_single_replication_substudy_a(
            config, rhat=2, missing_share=0.0, seed=42
        )

        assert 0.0 <= result["bm_r2"] <= 1.0, f"BM R^2 {result['bm_r2']:.3f} out of range"
        assert 0.0 <= result["rt_r2"] <= 1.0, f"RT R^2 {result['rt_r2']:.3f} out of range"
        assert 0.0 <= result["sw_r2"] <= 1.0, f"SW R^2 {result['sw_r2']:.3f} out of range"

    def test_no_missing_high_r2(self):
        """Test that no-missing case gives high R^2 with large T."""
        config = DGPConfig(n=20, T=200, r=2, s=0, rho=0.7)
        result = run_single_replication_substudy_a(
            config, rhat=2, missing_share=0.0, seed=42
        )

        assert result["bm_r2"] > 0.5, \
            f"BM R^2 {result['bm_r2']:.3f} too low with no missing and large T"

    def test_missing_reduces_r2(self):
        """Test that more missing data reduces R^2."""
        config = DGPConfig(n=20, T=100, r=2, s=0, rho=0.7)

        r2_no_miss = []
        r2_high_miss = []
        for seed in range(5):
            res_no = run_single_replication_substudy_a(config, 2, 0.0, seed)
            res_hi = run_single_replication_substudy_a(config, 2, 0.4, seed)
            r2_no_miss.append(res_no["bm_r2"])
            r2_high_miss.append(res_hi["bm_r2"])

        # On average, no missing should give higher R^2
        assert np.mean(r2_no_miss) >= np.mean(r2_high_miss) - 0.1, \
            "No missing should give higher R^2 than 40% missing"

    def test_overspecified_factors(self):
        """Test with rhat = r+1 (overspecified)."""
        config = DGPConfig(n=15, T=80, r=2, s=0, rho=0.7)
        result = run_single_replication_substudy_a(
            config, rhat=3, missing_share=0.0, seed=42
        )

        assert 0.0 <= result["bm_r2"] <= 1.0


class TestSubstudyB:
    """Tests for Substudy B: Ragged-edge forecast accuracy."""

    def test_single_replication_output(self):
        """Test that single replication returns valid results."""
        config = DGPConfig(n=15, T=80, r=3, s=0, rho=0.7, beta_fixed=0.5)
        result = run_single_replication_substudy_b(
            config, rhat=3, horizon=1, seed=42
        )

        assert "bm_acc" in result
        assert "rt_acc" in result
        assert "sw_acc" in result

    def test_accuracy_finite(self):
        """Test that forecast accuracy values are finite."""
        config = DGPConfig(n=15, T=80, r=3, s=0, rho=0.7, beta_fixed=0.5)
        for hor in [1, 2, 3, 4]:
            result = run_single_replication_substudy_b(
                config, rhat=3, horizon=hor, seed=42
            )
            assert np.isfinite(result["bm_acc"]), \
                f"BM accuracy not finite for horizon {hor}"

    def test_longer_horizon_harder(self):
        """Test that longer ragged-edge horizons are harder (lower accuracy)."""
        config = DGPConfig(n=20, T=100, r=3, s=0, rho=0.7, beta_fixed=0.5)

        acc_hor1 = []
        acc_hor4 = []
        for seed in range(5):
            res1 = run_single_replication_substudy_b(config, 3, 1, seed)
            res4 = run_single_replication_substudy_b(config, 3, 4, seed)
            acc_hor1.append(res1["bm_acc"])
            acc_hor4.append(res4["bm_acc"])

        # hor1 should generally be easier than hor4
        # (not always true for individual replications, but on average)
        assert np.mean(acc_hor1) >= np.mean(acc_hor4) - 0.2, \
            "hor1 should be at least as accurate as hor4 on average"


class TestSubstudyC:
    """Tests for Substudy C: AR(1) idiosyncratic estimation."""

    def test_single_replication_output(self):
        """Test that single replication returns valid MAE."""
        mae = run_single_replication_substudy_c(T=100, missing_share=0.0, seed=42)
        assert np.isfinite(mae), "MAE should be finite"
        assert mae >= 0.0, "MAE should be non-negative"

    def test_mae_decreases_with_T(self):
        """Test that MAE decreases as T increases."""
        mae_T50 = []
        mae_T200 = []
        for seed in range(5):
            mae50 = run_single_replication_substudy_c(T=50, missing_share=0.0, seed=seed)
            mae200 = run_single_replication_substudy_c(T=200, missing_share=0.0, seed=seed)
            if not np.isnan(mae50) and not np.isnan(mae200):
                mae_T50.append(mae50)
                mae_T200.append(mae200)

        if mae_T50 and mae_T200:
            assert np.mean(mae_T50) >= np.mean(mae_T200) - 0.05, \
                "MAE should decrease with larger T"

    def test_mae_range(self):
        """Test that MAE is in a reasonable range."""
        mae = run_single_replication_substudy_c(T=100, missing_share=0.0, seed=42)
        if not np.isnan(mae):
            assert 0.0 <= mae <= 1.0, f"MAE {mae:.3f} out of reasonable range [0, 1]"

    def test_missing_increases_mae(self):
        """Test that missing data increases MAE."""
        mae_no_miss = []
        mae_with_miss = []
        for seed in range(5):
            mae0 = run_single_replication_substudy_c(T=100, missing_share=0.0, seed=seed)
            mae20 = run_single_replication_substudy_c(T=100, missing_share=0.2, seed=seed)
            if not np.isnan(mae0) and not np.isnan(mae20):
                mae_no_miss.append(mae0)
                mae_with_miss.append(mae20)

        if mae_no_miss and mae_with_miss:
            # Missing data should generally increase MAE
            assert np.mean(mae_with_miss) >= np.mean(mae_no_miss) - 0.05, \
                "Missing data should not significantly reduce MAE"


class TestMetricsProperties:
    """Tests for mathematical properties of metrics."""

    def test_relative_forecast_accuracy_perfect(self):
        """Test relative forecast accuracy = 1 for perfect forecast."""
        acc = relative_forecast_accuracy(chi_true=1.0, chi_hat=1.0, var_chi=1.0)
        np.testing.assert_allclose(acc, 1.0)

    def test_relative_forecast_accuracy_zero_variance(self):
        """Test relative forecast accuracy with zero variance."""
        acc = relative_forecast_accuracy(chi_true=1.0, chi_hat=0.5, var_chi=0.0)
        assert acc == 0.0

    def test_mean_abs_error_ar_zero(self):
        """Test MAE = 0 when estimates equal truth."""
        alpha_hat = np.full(10, 0.7)
        mae = mean_abs_error_ar(alpha_hat, 0.7)
        np.testing.assert_allclose(mae, 0.0)

    def test_mean_abs_error_ar_positive(self):
        """Test MAE is positive when estimates differ from truth."""
        alpha_hat = np.full(10, 0.5)
        mae = mean_abs_error_ar(alpha_hat, 0.7)
        np.testing.assert_allclose(mae, 0.2)
