"""
Tests for the Data Generating Process (DGP) module.

Verifies that the DGP correctly implements the dynamic factor model
as described in the paper.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import (
    DGPConfig,
    simulate_dgp,
    impose_random_missing,
    impose_ragged_edge_missing,
    standardize_panel,
)


class TestDGPConfig:
    """Tests for DGPConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DGPConfig()
        assert config.n == 25
        assert config.T == 100
        assert config.r == 3
        assert config.s == 0
        assert config.rho == 0.7
        assert config.alpha == 0.0
        assert config.tau == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DGPConfig(n=10, T=50, r=1, s=1, rho=0.5, alpha=0.5, tau=0.5)
        assert config.n == 10
        assert config.T == 50
        assert config.r == 1
        assert config.s == 1


class TestSimulateDGP:
    """Tests for simulate_dgp function."""

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        config = DGPConfig(n=10, T=50, r=2, s=0)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        assert data["Y"].shape == (50, 10)
        assert data["F"].shape == (50, 2)
        assert data["Epsilon"].shape == (50, 10)
        assert data["Chi"].shape == (50, 10)
        assert len(data["Lambda"]) == 1  # s=0, so only Lambda_0

    def test_output_shapes_dynamic_loadings(self):
        """Test shapes with dynamic loadings (s=1)."""
        config = DGPConfig(n=10, T=50, r=2, s=1)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        assert len(data["Lambda"]) == 2  # Lambda_0 and Lambda_1
        assert data["Lambda"][0].shape == (10, 2)
        assert data["Lambda"][1].shape == (10, 2)

    def test_decomposition_identity(self):
        """Test that Y = Chi + Epsilon."""
        config = DGPConfig(n=10, T=50, r=2, s=0)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        np.testing.assert_allclose(
            data["Y"], data["Chi"] + data["Epsilon"], rtol=1e-10
        )

    def test_phi_positive_definite(self):
        """Test that Phi is positive definite."""
        config = DGPConfig(n=10, T=50, r=2, alpha=0.5, tau=0.5)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        eigenvalues = np.linalg.eigvalsh(data["Phi"])
        assert np.all(eigenvalues > 0), "Phi must be positive definite"

    def test_phi_structure_tau_zero(self):
        """Test Phi structure when tau=0 (diagonal)."""
        config = DGPConfig(n=5, T=50, r=1, alpha=0.5, tau=0.0)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        Phi = data["Phi"]
        # Off-diagonal should be zero when tau=0
        off_diag = Phi - np.diag(np.diag(Phi))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_phi_structure_tau_nonzero(self):
        """Test Phi structure when tau>0 (non-diagonal)."""
        config = DGPConfig(n=5, T=50, r=1, alpha=0.5, tau=0.5)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        Phi = data["Phi"]
        # Off-diagonal should be non-zero when tau>0
        off_diag = Phi - np.diag(np.diag(Phi))
        assert np.any(np.abs(off_diag) > 1e-10), "Phi should have non-zero off-diagonal when tau>0"

    def test_gamma_positive(self):
        """Test that gamma_i > 0 for all i."""
        config = DGPConfig(n=10, T=50, r=2)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        assert np.all(data["gamma"] > 0), "gamma_i must be positive"

    def test_beta_range(self):
        """Test that beta_i is in [0.1, 0.9]."""
        config = DGPConfig(n=100, T=50, r=2)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        assert np.all(data["beta"] >= 0.1), "beta_i must be >= 0.1"
        assert np.all(data["beta"] <= 0.9), "beta_i must be <= 0.9"

    def test_fixed_beta(self):
        """Test fixed beta parameter."""
        config = DGPConfig(n=10, T=50, r=2, beta_fixed=0.5)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        np.testing.assert_allclose(data["beta"], 0.5)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        config = DGPConfig(n=10, T=50, r=2)
        data1 = simulate_dgp(config, rng=np.random.default_rng(42))
        data2 = simulate_dgp(config, rng=np.random.default_rng(42))

        np.testing.assert_array_equal(data1["Y"], data2["Y"])
        np.testing.assert_array_equal(data1["F"], data2["F"])

    def test_factor_ar1_structure(self):
        """Test that factors follow AR(1) structure approximately."""
        config = DGPConfig(n=10, T=1000, r=1, rho=0.7)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        F = data["F"][:, 0]
        # Estimate AR(1) coefficient
        rho_hat = np.corrcoef(F[1:], F[:-1])[0, 1]
        # Should be close to 0.7 with large T
        assert abs(rho_hat - 0.7) < 0.1, f"AR(1) coefficient {rho_hat:.3f} far from 0.7"

    def test_no_missing_in_dgp(self):
        """Test that DGP output has no NaN values."""
        config = DGPConfig(n=10, T=50, r=2)
        data = simulate_dgp(config, rng=np.random.default_rng(42))

        assert not np.any(np.isnan(data["Y"]))
        assert not np.any(np.isnan(data["F"]))
        assert not np.any(np.isnan(data["Chi"]))
        assert not np.any(np.isnan(data["Epsilon"]))


class TestImposeMissing:
    """Tests for missing data generation functions."""

    def test_random_missing_share(self):
        """Test that random missing share is approximately correct."""
        rng = np.random.default_rng(42)
        Y = np.ones((100, 20))
        mask = impose_random_missing(Y, 0.25, rng)

        actual_share = 1.0 - mask.mean()
        assert abs(actual_share - 0.25) < 0.05, f"Missing share {actual_share:.3f} far from 0.25"

    def test_zero_missing(self):
        """Test that 0% missing gives all observed."""
        rng = np.random.default_rng(42)
        Y = np.ones((50, 10))
        mask = impose_random_missing(Y, 0.0, rng)

        assert np.all(mask), "All entries should be observed with 0% missing"

    def test_ragged_edge_hor1(self):
        """Test ragged-edge pattern hor1."""
        rng = np.random.default_rng(42)
        Y = np.ones((50, 20))
        mask = impose_ragged_edge_missing(Y, horizon=1, rng=rng)

        # Series 0 at T-1 must be missing
        assert not mask[-1, 0], "Series 0 at T must be missing for hor1"

        # Earlier periods should be mostly observed
        assert mask[:-1].mean() > 0.9, "Earlier periods should be mostly observed"

    def test_ragged_edge_hor4(self):
        """Test ragged-edge pattern hor4 (most missing)."""
        rng = np.random.default_rng(42)
        Y = np.ones((50, 20))
        mask = impose_ragged_edge_missing(Y, horizon=4, rng=rng)

        # Series 0 at T-1 must be missing
        assert not mask[-1, 0], "Series 0 at T must be missing for hor4"

        # Last 4 rows should have more missing than earlier rows
        last4_missing = 1.0 - mask[-4:].mean()
        earlier_missing = 1.0 - mask[:-4].mean()
        assert last4_missing > earlier_missing, "Last 4 rows should have more missing"

    def test_ragged_edge_invalid_horizon(self):
        """Test that invalid horizon raises assertion."""
        rng = np.random.default_rng(42)
        Y = np.ones((50, 20))
        with pytest.raises(AssertionError):
            impose_ragged_edge_missing(Y, horizon=5, rng=rng)

    def test_mask_dtype(self):
        """Test that mask has boolean dtype."""
        rng = np.random.default_rng(42)
        Y = np.ones((50, 10))
        mask = impose_random_missing(Y, 0.1, rng)
        assert mask.dtype == bool


class TestStandardizePanel:
    """Tests for panel standardization."""

    def test_standardized_mean_zero(self):
        """Test that standardized series have mean ~0."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((100, 10)) * 5 + 3
        Y_std, means, stds = standardize_panel(Y)

        np.testing.assert_allclose(np.nanmean(Y_std, axis=0), 0.0, atol=1e-10)

    def test_standardized_std_one(self):
        """Test that standardized series have std ~1."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((100, 10)) * 5 + 3
        Y_std, means, stds = standardize_panel(Y)

        np.testing.assert_allclose(np.nanstd(Y_std, axis=0, ddof=1), 1.0, atol=1e-10)

    def test_standardize_with_missing(self):
        """Test standardization with missing values."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((100, 5))
        mask = rng.random((100, 5)) > 0.2
        Y_std, means, stds = standardize_panel(Y, mask)

        # Check that NaN positions are preserved
        assert np.all(np.isnan(Y_std[~mask]))
