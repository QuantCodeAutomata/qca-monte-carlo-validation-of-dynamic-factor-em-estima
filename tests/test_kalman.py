"""
Tests for the Kalman filter and smoother module.

Verifies correctness of the Kalman filter and RTS smoother
with and without missing data.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kalman import kalman_filter, kalman_smoother, run_filter_smoother


def make_simple_model(T: int = 50, n: int = 3, r: int = 1):
    """Create a simple state-space model for testing."""
    rng = np.random.default_rng(42)

    # State-space parameters
    Z = rng.standard_normal((n, r))
    H = np.diag(np.abs(rng.standard_normal(n)) + 0.1)
    T_mat = 0.7 * np.eye(r)
    R_mat = np.eye(r)
    Q = np.eye(r)
    a0 = np.zeros(r)
    P0 = 10.0 * np.eye(r)

    # Simulate data
    alpha = np.zeros((T, r))
    alpha[0] = rng.multivariate_normal(a0, P0)
    for t in range(1, T):
        alpha[t] = T_mat @ alpha[t - 1] + rng.standard_normal(r)

    Y = alpha @ Z.T + rng.multivariate_normal(np.zeros(n), H, size=T)
    mask = np.ones((T, n), dtype=bool)

    return Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, alpha


class TestKalmanFilter:
    """Tests for Kalman filter."""

    def test_output_shapes(self):
        """Test that filter outputs have correct shapes."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        assert result["a_filt"].shape == (T, r)
        assert result["P_filt"].shape == (T, r, r)
        assert result["a_pred"].shape == (T + 1, r)
        assert result["P_pred"].shape == (T + 1, r, r)
        assert isinstance(result["log_lik"], float)

    def test_log_likelihood_finite(self):
        """Test that log-likelihood is finite."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        assert np.isfinite(result["log_lik"]), "Log-likelihood must be finite"

    def test_log_likelihood_negative(self):
        """Test that log-likelihood is negative (for proper distributions)."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        assert result["log_lik"] < 0, "Log-likelihood should be negative"

    def test_filtered_covariance_positive_definite(self):
        """Test that filtered covariances are positive definite."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        for t in range(T):
            eigenvalues = np.linalg.eigvalsh(result["P_filt"][t])
            assert np.all(eigenvalues >= -1e-8), f"P_filt[{t}] not positive semi-definite"

    def test_all_missing_prediction_only(self):
        """Test that all-missing time step uses prediction only."""
        T, n, r = 10, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        # Set all observations at t=5 to missing
        mask_with_missing = mask.copy()
        mask_with_missing[5, :] = False

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask_with_missing)

        # At t=5, filtered = predicted (no update)
        np.testing.assert_allclose(
            result["a_filt"][5], result["a_pred"][5], rtol=1e-10
        )

    def test_partial_missing(self):
        """Test filter with partial missing observations."""
        T, n, r = 50, 5, 2
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        # Set some observations to missing
        mask_partial = mask.copy()
        mask_partial[10:20, 2:] = False

        result = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask_partial)

        assert result["a_filt"].shape == (T, r)
        assert np.isfinite(result["log_lik"])

    def test_no_missing_vs_all_observed(self):
        """Test that None mask gives same result as all-True mask."""
        T, n, r = 30, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result1 = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, None)
        result2 = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        np.testing.assert_allclose(result1["a_filt"], result2["a_filt"], rtol=1e-10)
        np.testing.assert_allclose(result1["log_lik"], result2["log_lik"], rtol=1e-10)


class TestKalmanSmoother:
    """Tests for RTS smoother."""

    def test_output_shapes(self):
        """Test that smoother outputs have correct shapes."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        filt = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)
        smooth = kalman_smoother(
            filt["a_filt"], filt["P_filt"],
            filt["a_pred"], filt["P_pred"],
            T_mat,
        )

        assert smooth["a_smooth"].shape == (T, r)
        assert smooth["P_smooth"].shape == (T, r, r)
        assert smooth["P_cross"].shape == (T - 1, r, r)

    def test_smoother_last_equals_filter(self):
        """Test that smoother at T equals filter at T."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        filt = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)
        smooth = kalman_smoother(
            filt["a_filt"], filt["P_filt"],
            filt["a_pred"], filt["P_pred"],
            T_mat,
        )

        np.testing.assert_allclose(
            smooth["a_smooth"][-1], filt["a_filt"][-1], rtol=1e-10
        )

    def test_smoother_covariance_leq_filter(self):
        """Test that smoother covariance <= filter covariance (more info)."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        filt = kalman_filter(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)
        smooth = kalman_smoother(
            filt["a_filt"], filt["P_filt"],
            filt["a_pred"], filt["P_pred"],
            T_mat,
        )

        # Smoother variance should be <= filter variance
        for t in range(T - 1):
            diff = filt["P_filt"][t] - smooth["P_smooth"][t]
            eigenvalues = np.linalg.eigvalsh(diff)
            assert np.all(eigenvalues >= -1e-6), \
                f"Smoother variance exceeds filter variance at t={t}"

    def test_smoother_positive_definite(self):
        """Test that smoothed covariances are positive semi-definite."""
        T, n, r = 50, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = run_filter_smoother(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        for t in range(T):
            eigenvalues = np.linalg.eigvalsh(result["P_smooth"][t])
            assert np.all(eigenvalues >= -1e-8), \
                f"P_smooth[{t}] not positive semi-definite"


class TestRunFilterSmoother:
    """Tests for combined filter-smoother."""

    def test_combined_output(self):
        """Test that combined function returns both filter and smoother results."""
        T, n, r = 30, 3, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, _ = make_simple_model(T, n, r)

        result = run_filter_smoother(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        assert "a_filt" in result
        assert "a_smooth" in result
        assert "P_smooth" in result
        assert "P_cross" in result
        assert "log_lik" in result

    def test_state_recovery(self):
        """Test that smoother recovers true states reasonably well."""
        T, n, r = 200, 5, 1
        Y, mask, Z, H, T_mat, R_mat, Q, a0, P0, alpha_true = make_simple_model(T, n, r)

        result = run_filter_smoother(Y, Z, H, T_mat, R_mat, Q, a0, P0, mask)

        # Correlation between smoothed and true states should be high
        corr = np.corrcoef(result["a_smooth"][:, 0], alpha_true[:, 0])[0, 1]
        assert corr > 0.7, f"State recovery correlation {corr:.3f} too low"
