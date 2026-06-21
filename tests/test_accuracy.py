"""Unit tests for the accuracy aggregation helpers (stdlib only)."""

import math

from web.api.accuracy import aggregate_errors, prediction_errors


def _sample(lead, tmax_err, tmin_err, p_pred=0.0, p_actual=0.0):
    return {
        "lead_days": lead,
        "temp_max_error": tmax_err,
        "temp_min_error": tmin_err,
        "predicted_precipitation": p_pred,
        "actual_precipitation": p_actual,
    }


def test_empty_samples():
    stats = aggregate_errors([])
    assert stats["verified_predictions"] == 0
    assert stats["temp_max_mae"] == 0.0
    assert stats["accuracy_by_lead_day"] == {}


def test_mae_and_rmse():
    samples = [_sample(1, 2.0, -2.0), _sample(1, -4.0, 4.0)]
    stats = aggregate_errors(samples)
    assert stats["verified_predictions"] == 2
    assert stats["temp_max_mae"] == 3.0  # (|2| + |-4|) / 2
    assert stats["temp_max_rmse"] == round(math.sqrt((4 + 16) / 2), 3)


def test_by_lead_day_grouping():
    samples = [_sample(1, 1.0, 1.0), _sample(3, 5.0, 5.0), _sample(3, 1.0, 1.0)]
    stats = aggregate_errors(samples)
    by_lead = stats["accuracy_by_lead_day"]
    assert set(by_lead) == {1, 3}
    assert by_lead[3]["count"] == 2.0
    assert by_lead[3]["temp_max_mae"] == 3.0  # (5 + 1) / 2


def test_precip_occurrence_accuracy():
    # Two correct (wet/wet, dry/dry), one wrong (predicted dry, actually wet).
    samples = [
        _sample(1, 0, 0, p_pred=5.0, p_actual=4.0),  # wet/wet  -> correct
        _sample(1, 0, 0, p_pred=0.0, p_actual=0.0),  # dry/dry  -> correct
        _sample(1, 0, 0, p_pred=0.0, p_actual=3.0),  # dry/wet  -> wrong
    ]
    stats = aggregate_errors(samples)
    assert stats["precip_accuracy"] == round(100.0 * 2 / 3, 1)


def test_prediction_errors():
    err = prediction_errors(10.0, 2.0, 1.0, 8.0, 0.0, 0.5)
    assert err["temp_max_error"] == 2.0
    assert err["temp_min_error"] == 2.0
    assert err["precip_error"] == 0.5
