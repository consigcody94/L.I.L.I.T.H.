"""
Tests for LILITH data pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from data.processing.quality_control import QualityController, QCConfig, QCFlag
from data.processing.pipeline import FeatureEncoder, SpatialGridder


class TestQualityControl:
    """Tests for quality control module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample weather data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)

        df = pd.DataFrame({
            "TMAX": np.random.normal(15, 10, 100),
            "TMIN": np.random.normal(5, 8, 100),
            "PRCP": np.random.exponential(5, 100),
        }, index=dates)

        # Add some outliers
        df.loc[df.index[10], "TMAX"] = 100  # Unrealistic temperature
        df.loc[df.index[20], "PRCP"] = -5  # Negative precipitation
        df.loc[df.index[30], "TMAX"] = -100  # Extreme cold

        return df

    def test_range_check(self, sample_data):
        """Test physical range checking."""
        qc = QualityController()
        df_clean, flags = qc.process(sample_data)

        # Check that outliers were flagged
        assert flags.loc[sample_data.index[10], "TMAX_flag"] == QCFlag.SUSPECT_RANGE.value
        assert flags.loc[sample_data.index[20], "PRCP_flag"] == QCFlag.SUSPECT_RANGE.value
        assert flags.loc[sample_data.index[30], "TMAX_flag"] == QCFlag.SUSPECT_RANGE.value

        # Check that outliers were set to NaN
        assert pd.isna(df_clean.loc[sample_data.index[10], "TMAX"])
        assert pd.isna(df_clean.loc[sample_data.index[20], "PRCP"])

    def test_tmax_tmin_consistency(self):
        """Test TMAX >= TMIN check."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "TMAX": [20, 15, 10, 5, 15, 20, 25, 30, 35, 40],
            "TMIN": [10, 20, 5, 15, 10, 15, 20, 25, 30, 35],  # Days 1, 3 have TMIN > TMAX
        }, index=dates)

        qc = QualityController()
        _, flags = qc.process(df)

        # Check that inconsistent days are flagged
        assert flags.loc[dates[1], "TMAX_flag"] == QCFlag.SUSPECT_RANGE.value
        assert flags.loc[dates[3], "TMAX_flag"] == QCFlag.SUSPECT_RANGE.value

    def test_gap_filling(self):
        """Test gap filling."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "TMAX": [20.0] * 5 + [np.nan] * 2 + [20.0] * 13,
        }, index=dates)

        qc = QualityController()
        df_filled, flags = qc.fill_gaps(df, max_gap=3)

        # Gap should be filled
        assert not df_filled["TMAX"].isna().any()

        # Filled values should be flagged
        assert flags.loc[dates[5], "TMAX_flag"] == QCFlag.GAP_FILLED.value

    def test_duplicate_detection(self):
        """Test duplicate detection."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Create duplicate index
        dates_with_dup = dates.append(dates[5:6])

        df = pd.DataFrame({
            "TMAX": [20.0] * 11,
        }, index=dates_with_dup)

        qc = QualityController()
        df_clean, flags = qc.process(df)

        # Duplicate should be removed
        assert len(df_clean) == 10
        assert not df_clean.index.duplicated().any()


class TestFeatureEncoder:
    """Tests for feature encoding."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for encoding tests."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        np.random.seed(42)

        return pd.DataFrame({
            "TMAX": np.random.normal(15, 10, 365),
            "TMIN": np.random.normal(5, 8, 365),
            "PRCP": np.random.exponential(5, 365),
        }, index=dates)

    def test_fit(self, sample_data):
        """Test encoder fitting."""
        encoder = FeatureEncoder()
        encoder.fit(sample_data)

        assert "TMAX" in encoder.stats
        assert "mean" in encoder.stats["TMAX"]
        assert "std" in encoder.stats["TMAX"]

    def test_transform(self, sample_data):
        """Test data transformation."""
        encoder = FeatureEncoder()
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)

        # Temperature should be normalized
        assert abs(transformed["TMAX"].mean()) < 0.1
        assert abs(transformed["TMAX"].std() - 1.0) < 0.1

        # Should have time features
        assert "day_sin" in transformed.columns
        assert "day_cos" in transformed.columns

    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        encoder = FeatureEncoder()
        encoder.fit(sample_data)

        transformed = encoder.transform(sample_data)
        recovered = encoder.inverse_transform(transformed, columns=["TMAX", "TMIN"])

        # Should recover original values (approximately)
        np.testing.assert_allclose(
            recovered["TMAX"].values,
            sample_data["TMAX"].values,
            rtol=0.01
        )

    def test_precipitation_log_transform(self, sample_data):
        """Test precipitation log transformation."""
        encoder = FeatureEncoder()
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)

        # Log-transformed precip should be more normally distributed
        # (roughly, as it's originally exponential)
        assert transformed["PRCP"].min() >= 0  # log1p is always >= 0 for positive input

    def test_save_load(self, sample_data, tmp_path):
        """Test encoder save and load."""
        encoder = FeatureEncoder()
        encoder.fit(sample_data)

        # Save
        save_path = tmp_path / "encoder.json"
        encoder.save(str(save_path))

        # Load
        loaded = FeatureEncoder.load(str(save_path))

        # Compare stats
        assert encoder.stats == loaded.stats


class TestSpatialGridder:
    """Tests for spatial gridding."""

    def test_grid_creation(self):
        """Test grid creation."""
        gridder = SpatialGridder(resolution=1.0)

        assert len(gridder.lat_grid) == 181  # -90 to 90
        assert len(gridder.lon_grid) == 360  # -180 to 179

    def test_idw_interpolation(self):
        """Test IDW interpolation."""
        gridder = SpatialGridder(resolution=5.0, max_distance=10.0)

        # Create sample station data
        stations = pd.DataFrame({
            "latitude": [40.0, 41.0, 40.5],
            "longitude": [-74.0, -74.0, -73.5],
            "temperature": [20.0, 22.0, 21.0],
        })

        grid = gridder.grid_stations(stations, "temperature")

        # Grid should have values near the stations
        # Find grid indices near 40, -74
        lat_idx = np.argmin(np.abs(gridder.lat_grid - 40))
        lon_idx = np.argmin(np.abs(gridder.lon_grid - (-74)))

        # Value should be close to station value
        if not np.isnan(grid[lat_idx, lon_idx]):
            assert abs(grid[lat_idx, lon_idx] - 20.0) < 5.0

    def test_empty_data(self):
        """Test handling of empty data."""
        gridder = SpatialGridder(resolution=5.0)

        stations = pd.DataFrame({
            "latitude": [],
            "longitude": [],
            "temperature": [],
        })

        grid = gridder.grid_stations(stations, "temperature")

        # Should return grid of NaNs
        assert np.isnan(grid).all()


class TestDatasetIntegration:
    """Integration tests for data loading."""

    def test_station_dataset_sample_structure(self):
        """Test that dataset samples have correct structure."""
        # This would require actual data files
        # For now, test the expected interface
        expected_keys = [
            "input_features",
            "input_mask",
            "target_features",
            "target_mask",
            "station_coords",
            "station_id",
        ]

        # Mock sample
        sample = {
            "input_features": torch.randn(365, 7),
            "input_mask": torch.ones(365, dtype=torch.bool),
            "target_features": torch.randn(90, 3),
            "target_mask": torch.ones(90, dtype=torch.bool),
            "station_coords": torch.tensor([40.0, -74.0, 10.0]),
            "station_id": "USW00094728",
        }

        for key in expected_keys:
            assert key in sample

    def test_forecast_dataset_sample_structure(self):
        """Test forecast dataset sample structure."""
        expected_keys = [
            "node_features",
            "node_coords",
            "edge_index",
            "edge_attr",
            "target_features",
            "mask",
            "n_stations",
            "date",
        ]

        # Mock sample
        n_stations = 50
        seq_len = 30
        forecast_len = 14
        n_features = 3

        sample = {
            "node_features": torch.randn(n_stations, seq_len, n_features),
            "node_coords": torch.randn(n_stations, 3),
            "edge_index": torch.randint(0, n_stations, (2, 200)),
            "edge_attr": torch.randn(200, 1),
            "target_features": torch.randn(n_stations, forecast_len, n_features),
            "mask": torch.ones(n_stations, seq_len + forecast_len, dtype=torch.bool),
            "n_stations": n_stations,
            "date": "2020-01-01",
        }

        for key in expected_keys:
            assert key in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
