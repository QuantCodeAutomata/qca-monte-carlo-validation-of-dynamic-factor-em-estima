"""
Tests for BM, RT, and SW estimators.

Verifies that the estimators correctly implement the EM algorithms
and produce valid factor estimates.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import DGPConfig, simulate_dgp, impose_random_missing
from src.bm_estimator import em_bm
from src.rt_estimator import em_rt
from src.sw_estimator import em_sw
from src.metrics import trace_r2


def generate_test_data(n=15, T=80, r=2, s=0, rho=0.7, alpha=0.0, tau=0.0,
                       missing_share=0.0, seed=42):
    """Generate test data for estimator tests."""
    config = DGPConfig(n=n, T=T, r=r, s=s, rho=rho, alpha=alpha, tau=tau)
    rng = np.random.default_rng(seed)
    data = simulate_dgp(config, rng)
    Y = data["Y"]
    F_true = data["F"]

    mask = impose_random_missing(Y, missing_share, rng)
    Y_nan = Y.copy().astype(float)
    Y_nan[~mask] = np.nan

    return Y_nan, mask, F_true, data


class TestBMEstimator:
    """Tests for Banbura-Modugno EM estimator."""

    def test_output_keys(self):
        """Test that BM estimator returns expected keys."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=20)

        assert "Lambda" in result
        assert "A_list" in result
        assert "Q" in result
        assert "R" in result
        assert "F_smooth" in result
        assert "log_lik_history" in result
        assert "converged" in result

    def test_output_shapes(self):
        """Test that BM estimator outputs have correct shapes."""
        n, T, r = 15, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r)
        result = em_bm(Y_nan, mask, r=r, p=1, ar1_idio=False, max_iter=20)

        assert result["Lambda"].shape == (n, r)
        assert result["F_smooth"].shape == (T, r)
        assert result["Q"].shape == (r, r)
        assert result["R"].shape == (n, n)
        assert len(result["A_list"]) == 1
        assert result["A_list"][0].shape == (r, r)

    def test_log_likelihood_monotone(self):
        """Test that EM log-likelihood is non-decreasing."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=30)

        ll = result["log_lik_history"]
        if len(ll) > 2:
            # Allow small numerical decreases
            for i in range(1, len(ll)):
                assert ll[i] >= ll[i - 1] - 1.0, \
                    f"Log-likelihood decreased at iteration {i}: {ll[i-1]:.4f} -> {ll[i]:.4f}"

    def test_factor_recovery_no_missing(self):
        """Test that BM recovers factors well with no missing data."""
        Y_nan, mask, F_true, _ = generate_test_data(n=20, T=200, r=2, missing_share=0.0)
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=50)

        r2 = trace_r2(F_true, result["F_smooth"])
        assert r2 > 0.5, f"Trace R^2 {r2:.3f} too low with no missing data"

    def test_factor_recovery_with_missing(self):
        """Test that BM recovers factors with 25% missing data."""
        Y_nan, mask, F_true, _ = generate_test_data(n=20, T=200, r=2, missing_share=0.25)
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=50)

        r2 = trace_r2(F_true, result["F_smooth"])
        assert r2 > 0.3, f"Trace R^2 {r2:.3f} too low with 25% missing"

    def test_r_diagonal(self):
        """Test that R is diagonal."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=20)

        R = result["R"]
        off_diag = R - np.diag(np.diag(R))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_r_positive(self):
        """Test that R diagonal entries are positive."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=20)

        assert np.all(np.diag(result["R"]) > 0), "R diagonal must be positive"

    def test_q_positive_definite(self):
        """Test that Q is positive definite."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_bm(Y_nan, mask, r=2, p=1, ar1_idio=False, max_iter=20)

        eigenvalues = np.linalg.eigvalsh(result["Q"])
        assert np.all(eigenvalues > 0), "Q must be positive definite"

    def test_ar1_idio_output(self):
        """Test BM with AR(1) idiosyncratic components."""
        n, T, r = 10, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r, alpha=0.5)
        result = em_bm(Y_nan, mask, r=r, p=1, ar1_idio=True, max_iter=20, kappa=1e-6)

        assert "alpha" in result
        assert "sigma2" in result
        assert result["alpha"].shape == (n,)
        assert result["sigma2"].shape == (n,)

    def test_ar1_idio_alpha_range(self):
        """Test that estimated AR(1) coefficients are in (-1, 1)."""
        n, T, r = 10, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r, alpha=0.5)
        result = em_bm(Y_nan, mask, r=r, p=1, ar1_idio=True, max_iter=20, kappa=1e-6)

        assert np.all(result["alpha"] > -1.0), "alpha must be > -1"
        assert np.all(result["alpha"] < 1.0), "alpha must be < 1"

    def test_ar1_idio_sigma2_positive(self):
        """Test that AR(1) innovation variances are positive."""
        n, T, r = 10, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r, alpha=0.5)
        result = em_bm(Y_nan, mask, r=r, p=1, ar1_idio=True, max_iter=20, kappa=1e-6)

        assert np.all(result["sigma2"] > 0), "sigma2 must be positive"

    def test_var_order_p2(self):
        """Test BM with VAR(2) factor dynamics."""
        Y_nan, mask, F_true, _ = generate_test_data(n=15, T=100, r=2)
        result = em_bm(Y_nan, mask, r=2, p=2, ar1_idio=False, max_iter=20)

        assert len(result["A_list"]) == 2
        assert result["F_smooth"].shape == (100, 2)

    def test_overspecified_factors(self):
        """Test BM with rhat > r (overspecified)."""
        Y_nan, mask, F_true, _ = generate_test_data(n=15, T=100, r=2)
        # Estimate with rhat=3 when true r=2
        result = em_bm(Y_nan, mask, r=3, p=1, ar1_idio=False, max_iter=20)

        r2 = trace_r2(F_true, result["F_smooth"])
        assert r2 > 0.3, f"Trace R^2 {r2:.3f} too low with overspecified factors"

    def test_small_sample(self):
        """Test BM with small sample (T=50, n=10)."""
        Y_nan, mask, F_true, _ = generate_test_data(n=10, T=50, r=1)
        result = em_bm(Y_nan, mask, r=1, p=1, ar1_idio=False, max_iter=20)

        assert result["F_smooth"].shape == (50, 1)
        assert np.isfinite(result["log_lik_history"][-1])


class TestRTEstimator:
    """Tests for Rubin-Thayer EM estimator."""

    def test_output_keys(self):
        """Test that RT estimator returns expected keys."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_rt(Y_nan, mask, r=2, max_iter=20)

        assert "Lambda" in result
        assert "R" in result
        assert "F_hat" in result
        assert "log_lik_history" in result

    def test_output_shapes(self):
        """Test RT estimator output shapes."""
        n, T, r = 15, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r)
        result = em_rt(Y_nan, mask, r=r, max_iter=20)

        assert result["Lambda"].shape == (n, r)
        assert result["F_hat"].shape == (T, r)
        assert result["R"].shape == (n, n)

    def test_log_likelihood_monotone(self):
        """Test that RT EM log-likelihood is non-decreasing."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_rt(Y_nan, mask, r=2, max_iter=30)

        ll = result["log_lik_history"]
        if len(ll) > 2:
            for i in range(1, len(ll)):
                assert ll[i] >= ll[i - 1] - 1.0, \
                    f"RT log-likelihood decreased at iteration {i}"

    def test_factor_recovery(self):
        """Test that RT recovers factors reasonably."""
        Y_nan, mask, F_true, _ = generate_test_data(n=20, T=200, r=2, missing_share=0.0)
        result = em_rt(Y_nan, mask, r=2, max_iter=50)

        r2 = trace_r2(F_true, result["F_hat"])
        assert r2 > 0.3, f"RT Trace R^2 {r2:.3f} too low"


class TestSWEstimator:
    """Tests for Stock-Watson EM PC estimator."""

    def test_output_keys(self):
        """Test that SW estimator returns expected keys."""
        Y_nan, mask, F_true, _ = generate_test_data()
        result = em_sw(Y_nan, mask, r=2, max_iter=20)

        assert "Lambda" in result
        assert "F_hat" in result
        assert "Y_filled" in result

    def test_output_shapes(self):
        """Test SW estimator output shapes."""
        n, T, r = 15, 80, 2
        Y_nan, mask, F_true, _ = generate_test_data(n=n, T=T, r=r)
        result = em_sw(Y_nan, mask, r=r, max_iter=20)

        assert result["Lambda"].shape == (n, r)
        assert result["F_hat"].shape == (T, r)
        assert result["Y_filled"].shape == (T, n)

    def test_no_nan_in_filled(self):
        """Test that SW fills all missing values."""
        Y_nan, mask, F_true, _ = generate_test_data(missing_share=0.25)
        result = em_sw(Y_nan, mask, r=2, max_iter=20)

        assert not np.any(np.isnan(result["Y_filled"])), \
            "SW should fill all missing values"

    def test_factor_recovery(self):
        """Test that SW recovers factors reasonably."""
        Y_nan, mask, F_true, _ = generate_test_data(n=20, T=200, r=2, missing_share=0.0)
        result = em_sw(Y_nan, mask, r=2, max_iter=50)

        r2 = trace_r2(F_true, result["F_hat"])
        assert r2 > 0.3, f"SW Trace R^2 {r2:.3f} too low"


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_trace_r2_perfect(self):
        """Test trace R^2 = 1 for perfect recovery."""
        T, r = 100, 2
        F_true = np.random.default_rng(42).standard_normal((T, r))
        # Perfect recovery (up to rotation)
        r2 = trace_r2(F_true, F_true)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)

    def test_trace_r2_range(self):
        """Test that trace R^2 is in [0, 1]."""
        T, r = 100, 2
        rng = np.random.default_rng(42)
        F_true = rng.standard_normal((T, r))
        F_hat = rng.standard_normal((T, r))

        r2 = trace_r2(F_true, F_hat)
        assert 0.0 <= r2 <= 1.0, f"Trace R^2 {r2:.3f} out of [0, 1]"

    def test_trace_r2_orthogonal(self):
        """Test trace R^2 ≈ 0 for orthogonal factors."""
        T = 100
        rng = np.random.default_rng(42)
        F_true = rng.standard_normal((T, 1))
        # Orthogonal factor: project out F_true component
        F_hat = rng.standard_normal((T, 1))
        proj = (F_hat.T @ F_true) / (F_true.T @ F_true)  # scalar (1,1)
        F_hat = F_hat - F_true * proj[0, 0]

        r2 = trace_r2(F_true, F_hat)
        assert r2 < 0.1, f"Trace R^2 {r2:.3f} should be near 0 for orthogonal factors"

    def test_trace_r2_overspecified(self):
        """Test trace R^2 with rhat > r."""
        T, r = 100, 2
        rng = np.random.default_rng(42)
        F_true = rng.standard_normal((T, r))
        F_hat = rng.standard_normal((T, r + 1))  # overspecified

        r2 = trace_r2(F_true, F_hat)
        assert 0.0 <= r2 <= 1.0, f"Trace R^2 {r2:.3f} out of [0, 1] for overspecified"

    def test_trace_r2_rotation_invariant(self):
        """Test that trace R^2 is invariant to rotation of F_hat."""
        T, r = 100, 2
        rng = np.random.default_rng(42)
        F_true = rng.standard_normal((T, r))
        F_hat = rng.standard_normal((T, r))

        # Apply random rotation to F_hat
        Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
        F_hat_rotated = F_hat @ Q

        r2_orig = trace_r2(F_true, F_hat)
        r2_rotated = trace_r2(F_true, F_hat_rotated)

        np.testing.assert_allclose(r2_orig, r2_rotated, atol=1e-10)
