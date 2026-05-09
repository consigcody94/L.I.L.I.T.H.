"""Tests for the trivial forecasting baselines.

Beyond unit-checking the math, these tests pin down the regression-gate
behavior: ``assert_model_beats_baselines`` should fail loudly when a
model doesn't beat persistence + climatology. That assertion is the
canary that would have detected the inference-discard bug.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from inference.baselines import (
    assert_model_beats_baselines,
    daily_climatology_forecast,
    evaluate_baselines,
    persistence_forecast,
)


class TestPersistence:
    def test_repeats_last_day(self):
        history = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        out = persistence_forecast(history, forecast_days=4)
        assert out.shape == (4, 3)
        assert np.allclose(out, history[-1])

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="history must"):
            persistence_forecast(np.zeros(5), forecast_days=3)


class TestClimatology:
    def test_returns_correct_shape(self):
        out = daily_climatology_forecast(40.7, -74.0, forecast_days=14)
        assert out.shape == (14, 3)
        assert out.dtype == np.float32

    def test_tmax_above_tmin(self):
        out = daily_climatology_forecast(40.7, -74.0, forecast_days=14)
        assert (out[:, 0] > out[:, 1]).all()

    def test_seasonal_signal_for_temperate_lat(self):
        """Mid-latitude winter should be much colder than summer."""
        winter = daily_climatology_forecast(
            40.0, -74.0, forecast_days=1,
            start_date=datetime(2024, 1, 15),
        )
        summer = daily_climatology_forecast(
            40.0, -74.0, forecast_days=1,
            start_date=datetime(2024, 7, 15),
        )
        # At least 15 °C swing between seasons at 40N
        assert (summer[0, 0] - winter[0, 0]) > 15

    def test_southern_hemisphere_phase_flipped(self):
        # Same calendar day in NH summer (July 15) should be COLD in SH.
        nh = daily_climatology_forecast(
            40.0, -74.0, forecast_days=1,
            start_date=datetime(2024, 7, 15),
        )
        sh = daily_climatology_forecast(
            -40.0, -74.0, forecast_days=1,
            start_date=datetime(2024, 7, 15),
        )
        assert nh[0, 0] > sh[0, 0]

    def test_tropics_have_low_seasonality(self):
        out = daily_climatology_forecast(5.0, 0.0, forecast_days=365)
        season_amp = float(out[:, 0].max() - out[:, 0].min())
        # tropical seasonal amplitude is small
        assert season_amp < 8


class TestEvaluateBaselines:
    @pytest.fixture
    def synthetic_val_set(self):
        rng = np.random.default_rng(0)
        n = 50
        # Targets are normalized; create reasonable X, Y, meta.
        X = rng.standard_normal((n, 30, 3)).astype(np.float32)
        Y = rng.standard_normal((n, 14, 3)).astype(np.float32)
        meta = np.zeros((n, 4), dtype=np.float32)
        meta[:, 0] = rng.uniform(25, 50, n) / 90.0  # normalized lat
        meta[:, 1] = rng.uniform(-125, -67, n) / 180.0  # normalized lon
        meta[:, 2] = 0
        meta[:, 3] = rng.uniform(0, 1, n)
        Y_mean = np.array([10.0, 0.0, 2.0])
        Y_std = np.array([12.0, 10.0, 5.0])
        return X, Y, meta, Y_mean, Y_std

    def test_returns_persistence_and_climatology(self, synthetic_val_set):
        X, Y, meta, Y_mean, Y_std = synthetic_val_set
        out = evaluate_baselines(X, Y, meta, Y_mean=Y_mean, Y_std=Y_std)
        assert "persistence" in out
        assert "climatology" in out
        for b in out.values():
            assert "rmse_temp" in b
            assert "mae_temp" in b
            assert b["rmse_temp"] > 0

    def test_metrics_in_celsius_when_stats_provided(self, synthetic_val_set):
        X, Y, meta, Y_mean, Y_std = synthetic_val_set
        out = evaluate_baselines(X, Y, meta, Y_mean=Y_mean, Y_std=Y_std)
        # On synthetic standard-normal data, denormalized RMSE should be on
        # the order of one std (~12 °C since Y_std[0]=12).
        assert 5 < out["persistence"]["rmse_temp"] < 30


class TestRegressionGate:
    def test_passes_when_model_beats_baselines(self):
        baselines = {
            "persistence": {"rmse_temp": 5.0, "mae_temp": 4.0, "label": "p"},
            "climatology": {"rmse_temp": 4.0, "mae_temp": 3.0, "label": "c"},
        }
        # 3.0 < both 5.0 and 4.0, so should not raise.
        assert_model_beats_baselines(3.0, baselines)

    def test_fails_when_model_no_better_than_climatology(self):
        baselines = {
            "persistence": {"rmse_temp": 5.0, "mae_temp": 4.0, "label": "p"},
            "climatology": {"rmse_temp": 4.0, "mae_temp": 3.0, "label": "c"},
        }
        with pytest.raises(AssertionError, match="not better than"):
            # 5.5 ties with persistence which means the model is no better
            assert_model_beats_baselines(5.5, baselines)

    def test_margin_enforces_meaningful_gap(self):
        baselines = {
            "persistence": {"rmse_temp": 5.0, "mae_temp": 4.0, "label": "p"},
            "climatology": {"rmse_temp": 4.0, "mae_temp": 3.0, "label": "c"},
        }
        # Just barely beating (4.95) without the margin requirement passes.
        assert_model_beats_baselines(4.95, baselines)
        # With a 1.0 °C margin requirement, 4.95 fails (worst is 5.0).
        with pytest.raises(AssertionError):
            assert_model_beats_baselines(4.95, baselines, margin=1.0)
