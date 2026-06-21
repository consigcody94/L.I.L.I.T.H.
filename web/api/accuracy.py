"""
Accuracy aggregation helpers (stdlib only, no FastAPI/torch).

Kept dependency-free so the error-aggregation logic can be unit-tested directly.
A "sample" is a plain dict describing one verified prediction:

    {
        "lead_days": int,
        "temp_max_error": float,   # predicted - actual
        "temp_min_error": float,
        "predicted_precipitation": float,
        "actual_precipitation": float,
    }
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

# A day counts as "wet" above this threshold (mm) for occurrence accuracy.
PRECIP_OCCURRENCE_MM = 0.2


def _mae(values: List[float]) -> float:
    return sum(abs(v) for v in values) / len(values) if values else 0.0


def _rmse(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def _precip_occurrence_accuracy(samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        pred_wet = s["predicted_precipitation"] >= PRECIP_OCCURRENCE_MM
        actual_wet = s["actual_precipitation"] >= PRECIP_OCCURRENCE_MM
        if pred_wet == actual_wet:
            correct += 1
    return 100.0 * correct / len(samples)


def aggregate_errors(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate verified-prediction error samples into summary statistics.

    Returns a dict with the numeric fields expected by ``AccuracyStats`` plus an
    ``accuracy_by_lead_day`` mapping. All values are real, computed from the
    provided samples - no placeholders.
    """
    tmax_err = [s["temp_max_error"] for s in samples]
    tmin_err = [s["temp_min_error"] for s in samples]
    precip_err = [
        s["predicted_precipitation"] - s["actual_precipitation"] for s in samples
    ]

    by_lead: Dict[int, Dict[str, float]] = {}
    leads = sorted({s["lead_days"] for s in samples})
    for lead in leads:
        group = [s for s in samples if s["lead_days"] == lead]
        by_lead[lead] = {
            "temp_max_mae": round(_mae([s["temp_max_error"] for s in group]), 3),
            "temp_min_mae": round(_mae([s["temp_min_error"] for s in group]), 3),
            "count": float(len(group)),
        }

    return {
        "verified_predictions": len(samples),
        "temp_max_mae": round(_mae(tmax_err), 3),
        "temp_max_rmse": round(_rmse(tmax_err), 3),
        "temp_min_mae": round(_mae(tmin_err), 3),
        "temp_min_rmse": round(_rmse(tmin_err), 3),
        "precip_mae": round(_mae(precip_err), 3),
        "precip_accuracy": round(_precip_occurrence_accuracy(samples), 1),
        "accuracy_by_lead_day": by_lead,
    }


def prediction_errors(
    predicted_max: float,
    predicted_min: float,
    predicted_precip: float,
    actual_max: float,
    actual_min: float,
    actual_precip: float,
) -> Dict[str, float]:
    """Per-prediction signed errors (predicted - actual)."""
    return {
        "temp_max_error": round(predicted_max - actual_max, 2),
        "temp_min_error": round(predicted_min - actual_min, 2),
        "precip_error": round(predicted_precip - actual_precip, 2),
    }
