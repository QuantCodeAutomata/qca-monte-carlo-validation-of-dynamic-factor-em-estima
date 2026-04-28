"""
Tests for empirical experiments (exp_2, exp_3, exp_4).

Verifies that the empirical data handling, nowcasting, news decomposition,
and backdating modules work correctly.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.empirical_data import (
    get_metadata_df,
    get_panel_series,
    generate_synthetic_euro_area_data,
    apply_publication_lag_mask,
    recursive_standardize,
    prepare_panel_for_estimation,
    mariano_murasawa_weights,
)


class TestMetadata:
    """Tests for Appendix A metadata."""

    def test_metadata_not_empty(self):
        """Test that metadata is not empty."""
        meta = get_metadata_df()
        assert len(meta) > 0

    def test_metadata_columns(self):
        """Test that metadata has required columns."""
        meta = get_metadata_df()
        required_cols = ["name", "group", "quarterly", "log", "diff",
                        "pub_lag_m1", "pub_lag_m2", "pub_lag_m3",
                        "small", "medium", "large"]
        for col in required_cols:
            assert col in meta.columns, f"Missing column: {col}"

    def test_small_panel_size(self):
        """Test that small panel has ~14 series."""
        series = get_panel_series("small")
        assert 10 <= len(series) <= 20, f"Small panel has {len(series)} series, expected ~14"

    def test_medium_panel_larger_than_small(self):
        """Test that medium panel is larger than small."""
        small = get_panel_series("small")
        medium = get_panel_series("medium")
        assert len(medium) > len(small), "Medium panel should be larger than small"

    def test_large_panel_largest(self):
        """Test that large panel is the largest."""
        medium = get_panel_series("medium")
        large = get_panel_series("large")
        assert len(large) >= len(medium), "Large panel should be at least as large as medium"

    def test_small_subset_of_medium(self):
        """Test that small panel series are in medium panel."""
        small = set(get_panel_series("small"))
        medium = set(get_panel_series("medium"))
        assert small.issubset(medium), "Small panel should be subset of medium"

    def test_gdp_in_all_panels(self):
        """Test that GDP is in all panels."""
        for panel in ["small", "medium", "large"]:
            series = get_panel_series(panel)
            gdp_series = [s for s in series if "GDP" in s.upper() and "US" not in s.upper()]
            assert len(gdp_series) > 0, f"GDP not found in {panel} panel"

    def test_groups_valid(self):
        """Test that all series have valid group labels."""
        meta = get_metadata_df()
        valid_groups = {"Real", "Survey", "Financial", "US"}
        for _, row in meta.iterrows():
            assert row["group"] in valid_groups, \
                f"Invalid group '{row['group']}' for series '{row['name']}'"

    def test_mariano_murasawa_weights(self):
        """Test Mariano-Murasawa weights sum to 9."""
        weights = mariano_murasawa_weights()
        assert len(weights) == 5
        np.testing.assert_allclose(weights.sum(), 9.0)
        np.testing.assert_allclose(weights, [1, 2, 3, 2, 1])


class TestSyntheticData:
    """Tests for synthetic euro area data generation."""

    def test_data_shape(self):
        """Test that synthetic data has correct shape."""
        df = generate_synthetic_euro_area_data(
            start_date="1993-01", end_date="2009-10", seed=42
        )
        meta = get_metadata_df()
        assert df.shape[1] == len(meta), \
            f"Data has {df.shape[1]} columns, expected {len(meta)}"

    def test_date_range(self):
        """Test that data covers the expected date range."""
        df = generate_synthetic_euro_area_data(
            start_date="1993-01", end_date="2009-10", seed=42
        )
        assert df.index[0] == pd.Timestamp("1993-01-01")
        assert df.index[-1] <= pd.Timestamp("2009-10-01")

    def test_quarterly_series_missing(self):
        """Test that quarterly series have NaN in non-quarter-end months."""
        df = generate_synthetic_euro_area_data(
            start_date="1993-01", end_date="2009-10", seed=42
        )
        meta = get_metadata_df()
        quarterly_series = meta[meta["quarterly"]]["name"].tolist()

        for series in quarterly_series[:3]:  # Check first 3 quarterly series
            if series in df.columns:
                # Non-quarter-end months should be NaN
                non_qe = df[~df.index.month.isin([3, 6, 9, 12])][series]
                assert non_qe.isna().all(), \
                    f"Quarterly series {series} should have NaN in non-quarter-end months"

    def test_reproducibility(self):
        """Test that same seed gives same data."""
        df1 = generate_synthetic_euro_area_data(seed=42)
        df2 = generate_synthetic_euro_area_data(seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestPublicationLagMask:
    """Tests for publication lag masking."""

    def test_mask_applied(self):
        """Test that publication lag mask is applied."""
        df = generate_synthetic_euro_area_data(seed=42)
        meta = get_metadata_df()
        vintage_date = pd.Timestamp("2000-03-15")

        df_masked = apply_publication_lag_mask(df, vintage_date, meta)

        # Some series should have NaN after masking
        assert df_masked.isna().any().any(), "Mask should create some NaN values"

    def test_future_data_masked(self):
        """Test that future data is masked."""
        df = generate_synthetic_euro_area_data(seed=42)
        meta = get_metadata_df()
        vintage_date = pd.Timestamp("2000-03-15")

        df_masked = apply_publication_lag_mask(df, vintage_date, meta)

        # Data after vintage date should be NaN
        future_data = df_masked[df_masked.index > vintage_date]
        assert future_data.isna().all().all(), "All future data should be masked"


class TestRecursiveStandardize:
    """Tests for recursive standardization."""

    def test_standardized_mean_zero(self):
        """Test that standardized series have mean ~0."""
        df = generate_synthetic_euro_area_data(seed=42)
        end_date = pd.Timestamp("2005-01-01")
        df_std, means, stds = recursive_standardize(df, end_date)

        # Check means are close to 0 for the estimation sample
        df_sample = df_std[df_std.index <= end_date]
        col_means = df_sample.mean()
        assert col_means.abs().max() < 0.1, "Standardized means should be near 0"

    def test_no_lookahead(self):
        """Test that standardization uses only past data."""
        df = generate_synthetic_euro_area_data(seed=42)
        end_date1 = pd.Timestamp("2000-01-01")
        end_date2 = pd.Timestamp("2005-01-01")

        _, means1, stds1 = recursive_standardize(df, end_date1)
        _, means2, stds2 = recursive_standardize(df, end_date2)

        # Means should differ because they use different sample periods
        # (not necessarily, but the function should use different data)
        assert not np.allclose(means1.values, means2.values, atol=1e-10), \
            "Recursive standardization should use different samples"


class TestPreparePanel:
    """Tests for panel preparation."""

    def test_prepare_small_panel(self):
        """Test preparation of small panel."""
        df = generate_synthetic_euro_area_data(seed=42)
        meta = get_metadata_df()
        vintage_date = pd.Timestamp("2005-01-15")

        Y, mask, series_names = prepare_panel_for_estimation(
            df, "small", vintage_date, meta
        )

        assert Y.ndim == 2
        assert mask.ndim == 2
        assert Y.shape == mask.shape
        assert len(series_names) > 0

    def test_mask_boolean(self):
        """Test that mask is boolean."""
        df = generate_synthetic_euro_area_data(seed=42)
        meta = get_metadata_df()
        vintage_date = pd.Timestamp("2005-01-15")

        Y, mask, _ = prepare_panel_for_estimation(df, "small", vintage_date, meta)

        assert mask.dtype == bool

    def test_nan_consistency(self):
        """Test that NaN in Y corresponds to False in mask."""
        df = generate_synthetic_euro_area_data(seed=42)
        meta = get_metadata_df()
        vintage_date = pd.Timestamp("2005-01-15")

        Y, mask, _ = prepare_panel_for_estimation(df, "small", vintage_date, meta)

        # Where mask is False, Y should be NaN
        assert np.all(np.isnan(Y[~mask])), "Y should be NaN where mask is False"
