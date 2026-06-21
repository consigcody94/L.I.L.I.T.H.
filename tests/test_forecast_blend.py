"""
Unit tests for the forecast assembly / bias-correction logic.

These run with NumPy only (no torch), so they exercise the heart of the
serving path without needing a trained checkpoint.
"""

from datetime import date

import numpy as np
import pytest

from inference.forecast_blend import (
    MAX_FORECAST_DAYS,
    TRUSTED_LEAD_DAYS,
    baseline_forecast,
    blend_forecast,
    climatology,
    model_weight,
    nwp_weight,
    synthetic_history,
)


class TestClimatology:
    def test_summer_warmer_than_winter_northern(self):
        # NYC-ish latitude: mid-July (day 196) should beat mid-January (day 15).
        summer = climatology(40.7, -74.0, 196)
        winter = climatology(40.7, -74.0, 15)
        assert summer["mean"] > winter["mean"]
        assert summer["tmax"] > summer["tmin"]

    def test_tropics_have_small_seasonal_swing(self):
        jan = climatology(5.0, 0.0, 15)
        jul = climatology(5.0, 0.0, 196)
        assert abs(jul["mean"] - jan["mean"]) < 6.0

    def test_southern_hemisphere_phase_inverted(self):
        # Southern mid-latitude: January (day 15) is summer, July is winter.
        jan = climatology(-35.0, 150.0, 15)
        jul = climatology(-35.0, 150.0, 196)
        assert jan["mean"] > jul["mean"]


class TestWeights:
    def test_model_weight_full_within_trusted_window(self):
        assert model_weight(1) == 1.0
        assert model_weight(TRUSTED_LEAD_DAYS) == 1.0

    def test_model_weight_decays_and_has_floor(self):
        assert model_weight(45) < 1.0
        assert model_weight(MAX_FORECAST_DAYS) == pytest.approx(0.15, abs=1e-6)
        # Monotonic non-increasing across the horizon.
        weights = [model_weight(d) for d in range(1, MAX_FORECAST_DAYS + 1)]
        assert all(b <= a + 1e-9 for a, b in zip(weights, weights[1:]))

    def test_nwp_weight_starts_high_fades_to_zero(self):
        assert nwp_weight(1, 16) > 0.5
        assert nwp_weight(16, 16) == pytest.approx(0.0, abs=1e-6)
        assert nwp_weight(20, 16) == 0.0
        assert nwp_weight(1, 0) == 0.0


class TestBlendForecast:
    def _pred(self, n, tmax=20.0, tmin=10.0, prcp=0.0):
        arr = np.zeros((n, 3))
        arr[:, 0] = tmax
        arr[:, 1] = tmin
        arr[:, 2] = prcp
        return arr

    def test_shape_and_keys(self):
        out = blend_forecast(self._pred(14), 40.7, -74.0, date(2025, 1, 1))
        assert len(out) == 14
        for f in out:
            assert {
                "date",
                "day",
                "temperature_high",
                "temperature_low",
                "precipitation_mm",
                "precipitation_probability",
                "source",
            } <= set(f)
            assert f["temperature_high"] >= f["temperature_low"]
            assert 0 <= f["precipitation_probability"] <= 100

    def test_is_deterministic(self):
        a = blend_forecast(self._pred(30), 33.9, -118.4, date(2025, 6, 1))
        b = blend_forecast(self._pred(30), 33.9, -118.4, date(2025, 6, 1))
        assert a == b

    def test_uses_model_anomaly_signal(self):
        # A model spike on day 3 must survive into the output (it isn't discarded).
        pred = self._pred(14)
        pred[2, 0] += 8.0  # warm spike on the high
        out = blend_forecast(pred, 40.7, -74.0, date(2025, 1, 1))
        highs = [f["temperature_high"] for f in out]
        assert highs[2] == max(highs)
        assert highs[2] - np.mean(highs) > 3.0

    def test_reference_pulls_near_term_toward_real_values(self):
        # Climatology in deep winter is cold; a warm real reference should pull
        # the first day strongly toward the reference value.
        pred = self._pred(14, tmax=0.0, tmin=-8.0)
        reference = {
            "tmax": [18.0] * 14,
            "tmin": [9.0] * 14,
            "prcp": [0.0] * 14,
            "precip_prob": [5.0] * 14,
        }
        out = blend_forecast(pred, 40.7, -74.0, date(2025, 1, 15), reference=reference)
        assert out[0]["source"] == "reference+model"
        assert out[0]["temperature_high"] > 8.0  # pulled well above the cold climatology
        assert out[0]["precipitation_probability"] == 5

    def test_long_range_reverts_to_climatology(self):
        # With a strong model bias, day 90 should still sit near climatology
        # because the model weight has decayed to its floor.
        pred = self._pred(90, tmax=50.0, tmin=40.0)  # absurdly hot, biased model
        out = blend_forecast(pred, 40.7, -74.0, date(2025, 1, 1))
        # Day 90 high should be far below the biased 50C and within a sane band.
        assert out[-1]["temperature_high"] < 35.0

    def test_empty_pred(self):
        assert blend_forecast(np.zeros((0, 3)), 0.0, 0.0, date(2025, 1, 1)) == []


class TestBaselineForecast:
    def test_full_horizon_no_reference(self):
        out = baseline_forecast(40.7, -74.0, date(2025, 1, 1), days=90)
        assert len(out) == 90
        assert all(f["source"] == "climatology" for f in out)
        assert all(f["temperature_high"] >= f["temperature_low"] for f in out)

    def test_reference_blended_near_term(self):
        reference = {
            "tmax": [25.0] * 16,
            "tmin": [15.0] * 16,
            "prcp": [3.0] * 16,
            "precip_prob": [60.0] * 16,
        }
        out = baseline_forecast(40.7, -74.0, date(2025, 1, 15), days=30, reference=reference)
        assert out[0]["source"] == "reference"
        # Day 1 should be pulled toward the warm reference, away from cold January.
        assert out[0]["temperature_high"] > 12.0
        # Beyond the reference window it reverts to pure climatology.
        assert out[20]["source"] == "climatology"

    def test_deterministic(self):
        a = baseline_forecast(51.5, -0.12, date(2025, 3, 1), days=14)
        b = baseline_forecast(51.5, -0.12, date(2025, 3, 1), days=14)
        assert a == b


class TestSyntheticHistory:
    def test_shape_and_determinism(self):
        a = synthetic_history(40.7, -74.0, days=30)
        b = synthetic_history(40.7, -74.0, days=30)
        assert a.shape == (30, 3)
        assert np.allclose(a, b)
        assert (a[:, 2] >= 0).all()  # precipitation is non-negative
