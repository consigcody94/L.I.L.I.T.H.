"""
Forecast assembly and bias correction (NumPy only, no torch).

This module turns raw model output into a calibrated daily forecast. It is kept
free of any deep-learning dependency so the forecasting logic can be unit-tested
in isolation and reused outside the model serving path.

The key design decision: the trained model's day-to-day *signal* is preserved and
anchored to a seasonally-correct climatology, rather than being thrown away. When a
real short-range reference forecast is supplied (e.g. Open-Meteo), it is blended in
for the near term where numerical weather prediction far outperforms any
subseasonal model. Output is fully deterministic - no random perturbations.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

Reference = Mapping[str, Sequence[Optional[float]]]

# Number of model lead days that are treated as "skilful" / fully trusted.
# The SimpleLILITH model is trained with a 14-day target window.
TRUSTED_LEAD_DAYS = 14
MAX_FORECAST_DAYS = 90


def climatology(latitude: float, longitude: float, day_of_year: int) -> Dict[str, float]:
    """
    Expected climatological temperatures for a location on a given day of year.

    Returns a dict with ``tmax``, ``tmin``, ``mean`` (degrees C) and ``diurnal``
    (the typical day-night range). Calibrated to US station normals; serves as the
    bias-correction anchor for model output and the long-range fallback.
    """
    abs_lat = abs(latitude)

    # Base annual mean, seasonal amplitude and diurnal range by climate zone.
    if abs_lat < 15:  # Tropical
        annual_mean, seasonal_amp, diurnal_range = 27.0, 2.0, 10.0
    elif abs_lat < 28:  # Subtropical (Miami, Houston)
        annual_mean, seasonal_amp, diurnal_range = 22.0, 7.0, 11.0
    elif abs_lat < 35:  # Warm temperate (LA, Atlanta)
        annual_mean, seasonal_amp, diurnal_range = 17.0, 10.0, 12.0
    elif abs_lat < 42:  # Mid temperate (NYC, Chicago)
        annual_mean, seasonal_amp, diurnal_range = 11.0, 14.0, 10.0
    elif abs_lat < 48:  # Cool temperate (Minneapolis, Seattle)
        annual_mean, seasonal_amp, diurnal_range = 7.0, 17.0, 11.0
    elif abs_lat < 55:  # Cold
        annual_mean, seasonal_amp, diurnal_range = 2.0, 20.0, 10.0
    else:  # Subarctic / Arctic
        annual_mean, seasonal_amp, diurnal_range = -5.0, 22.0, 9.0

    # Regional adjustments for the US (calibrated to January averages).
    if 24 < latitude < 32 and longitude > -88:  # Florida
        annual_mean += 2.0
        seasonal_amp *= 0.4
    elif 32 < latitude < 42 and longitude < -117:  # California coast
        annual_mean += 3.0
        seasonal_amp *= 0.35
        diurnal_range = 10.0
    elif 32 < latitude < 38 and -117 < longitude < -109:  # Desert SW (Phoenix)
        annual_mean += 2.0
        seasonal_amp *= 0.7
        diurnal_range = 13.0
    elif latitude > 45 and longitude < -122:  # Pacific NW
        seasonal_amp *= 0.5
        annual_mean += 3.0
    elif 28 < latitude < 32 and -98 < longitude < -93:  # Gulf Coast (Houston)
        annual_mean += 3.0
        seasonal_amp *= 0.6
    elif -110 < longitude < -100 and 35 < latitude < 45:  # Mountain West (Denver)
        seasonal_amp *= 0.9
        annual_mean += 2.0

    # Seasonal cycle. Northern hemisphere peaks near day 200 (late July).
    phase_shift = 200 if latitude >= 0 else 15
    seasonal_offset = seasonal_amp * np.cos(2 * np.pi * (day_of_year - phase_shift) / 365.0)
    current_mean = annual_mean + seasonal_offset

    return {
        "tmax": float(current_mean + diurnal_range / 2),
        "tmin": float(current_mean - diurnal_range / 2),
        "mean": float(current_mean),
        "diurnal": float(diurnal_range),
    }


def synthetic_history(
    latitude: float,
    longitude: float,
    days: int = 30,
    reference_date: Optional[date] = None,
) -> np.ndarray:
    """
    Deterministic synthetic recent history [days, 3] = (TMAX, TMIN, PRCP).

    Used only as a last-resort model input when no real observations are
    available. The randomness is seeded by location so results are reproducible.
    """
    reference_date = reference_date or datetime.now().date()
    day_of_year = reference_date.timetuple().tm_yday
    clim = climatology(latitude, longitude, day_of_year)

    rng = np.random.default_rng(int(abs(latitude * 1000 + longitude * 1000)) % 2**31)
    history = np.zeros((days, 3), dtype=np.float32)
    for i in range(days):
        daily_var = float(rng.standard_normal() * 3.0)
        daily_mean = clim["mean"] + daily_var
        history[i, 0] = daily_mean + clim["diurnal"] / 2  # TMAX
        history[i, 1] = daily_mean - clim["diurnal"] / 2  # TMIN
        history[i, 2] = max(0.0, float(rng.exponential(2.0)))  # PRCP (mm)
    return history


def model_weight(lead_day: int) -> float:
    """
    How much of the model's anomaly signal to trust at a given lead day.

    Full trust through the trained horizon, then a linear decay toward a small
    floor (subseasonal skill is real but weak), reaching the floor at day 90.
    """
    if lead_day <= TRUSTED_LEAD_DAYS:
        return 1.0
    span = MAX_FORECAST_DAYS - TRUSTED_LEAD_DAYS
    frac = (lead_day - TRUSTED_LEAD_DAYS) / span
    return float(max(0.15, 1.0 - 0.85 * frac))


def nwp_weight(lead_day: int, reference_len: int) -> float:
    """
    Weight given to a real short-range (NWP) reference forecast.

    High in the first days where numerical models dominate, fading to zero by the
    end of the reference window so the hand-off to the subseasonal model is smooth.
    """
    if reference_len <= 0 or lead_day > reference_len:
        return 0.0
    return float(max(0.0, 0.85 * (1.0 - (lead_day - 1) / max(1, reference_len - 1))))


def baseline_forecast(
    latitude: float,
    longitude: float,
    start_date: date,
    days: int,
    reference: Optional[Reference] = None,
) -> List[Dict[str, Any]]:
    """
    Model-free forecast: per-day climatology with a real short-range reference
    blended into the near term. Used when no trained checkpoint is available, so
    the API can still serve a full-horizon, real-data-anchored forecast with no
    model and no API key. Deterministic.
    """
    days = max(0, int(days))
    if days == 0:
        return []

    ref_tmax = (reference or {}).get("tmax") or []
    ref_tmin = (reference or {}).get("tmin") or []
    ref_prcp = (reference or {}).get("prcp") or []
    ref_prob = (reference or {}).get("precip_prob") or []
    ref_len = len(ref_tmax)

    forecasts: List[Dict[str, Any]] = []
    for i in range(days):
        lead = i + 1
        forecast_date = start_date + timedelta(days=i)
        clim = climatology(latitude, longitude, forecast_date.timetuple().tm_yday)
        tmax, tmin, prcp = clim["tmax"], clim["tmin"], 0.0
        source = "climatology"

        rt = ref_tmax[i] if i < ref_len else None
        rn = ref_tmin[i] if i < ref_len else None
        if rt is not None and rn is not None:
            nw = nwp_weight(lead, ref_len)
            if nw > 0.0:
                tmax = nw * float(rt) + (1.0 - nw) * tmax
                tmin = nw * float(rn) + (1.0 - nw) * tmin
                rp = ref_prcp[i] if i < len(ref_prcp) else None
                if rp is not None:
                    prcp = nw * float(rp)
                source = "reference"

        if tmin > tmax - 1.0:
            tmin = tmax - max(2.0, clim["diurnal"] * 0.5)

        rprob = ref_prob[i] if i < len(ref_prob) else None
        precip_prob = (
            int(round(float(rprob))) if rprob is not None else int(min(100, round(prcp * 12)))
        )

        forecasts.append(
            {
                "date": forecast_date.isoformat(),
                "day": lead,
                "temperature_high": round(tmax, 1),
                "temperature_low": round(tmin, 1),
                "precipitation_mm": round(prcp, 1),
                "precipitation_probability": max(0, min(100, precip_prob)),
                "source": source,
            }
        )

    return forecasts


def blend_forecast(
    pred: np.ndarray,
    latitude: float,
    longitude: float,
    start_date: date,
    reference: Optional[Reference] = None,
) -> List[Dict[str, Any]]:
    """
    Combine model output, climatology and an optional real reference forecast
    into a calibrated daily forecast.

    Args:
        pred: Denormalized model output [n, 3] = (TMAX, TMIN, PRCP).
        latitude, longitude: Location.
        start_date: Date of the first forecast day (lead day 1).
        reference: Optional real short-range forecast with parallel lists keyed by
            ``tmax``, ``tmin``, ``prcp`` and (optionally) ``precip_prob``. Entries
            may be ``None`` where a value is missing.

    Returns:
        List of per-day forecast dicts with stable keys consumed by the API layer.
    """
    pred = np.asarray(pred, dtype=np.float64)
    n = int(pred.shape[0])
    if n == 0:
        return []

    # Anchor the model to its own near-term mean so we keep the *shape* of its
    # prediction (warm spells, cold snaps) while discarding absolute bias.
    baseline_window = min(TRUSTED_LEAD_DAYS, n)
    base_tmax = float(np.mean(pred[:baseline_window, 0]))
    base_tmin = float(np.mean(pred[:baseline_window, 1]))

    ref_tmax = (reference or {}).get("tmax") or []
    ref_tmin = (reference or {}).get("tmin") or []
    ref_prcp = (reference or {}).get("prcp") or []
    ref_prob = (reference or {}).get("precip_prob") or []
    ref_len = len(ref_tmax)

    forecasts: List[Dict[str, Any]] = []
    for i in range(n):
        lead = i + 1
        forecast_date = start_date + timedelta(days=i)
        clim = climatology(latitude, longitude, forecast_date.timetuple().tm_yday)

        # Model anomaly relative to its own baseline, laid over seasonal climatology.
        mw = model_weight(lead)
        tmax = clim["tmax"] + mw * (float(pred[i, 0]) - base_tmax)
        tmin = clim["tmin"] + mw * (float(pred[i, 1]) - base_tmin)
        prcp = max(0.0, float(pred[i, 2]))
        source = "model+climatology"

        # Blend in a real reference forecast for the near term.
        rt = ref_tmax[i] if i < ref_len else None
        rn = ref_tmin[i] if i < ref_len else None
        if rt is not None and rn is not None:
            nw = nwp_weight(lead, ref_len)
            if nw > 0.0:
                tmax = nw * float(rt) + (1.0 - nw) * tmax
                tmin = nw * float(rn) + (1.0 - nw) * tmin
                rp = ref_prcp[i] if i < len(ref_prcp) else None
                if rp is not None:
                    prcp = nw * float(rp) + (1.0 - nw) * prcp
                source = "reference+model"

        # Enforce a physically sane diurnal range (min strictly below max).
        if tmin > tmax - 1.0:
            tmin = tmax - max(2.0, clim["diurnal"] * 0.5)

        # Precipitation probability: prefer the reference's value, else derive it
        # monotonically from the predicted amount (deterministic, no randomness).
        rprob = ref_prob[i] if i < len(ref_prob) else None
        if rprob is not None:
            precip_prob = int(round(float(rprob)))
        else:
            precip_prob = int(min(100, round(prcp * 12)))

        forecasts.append(
            {
                "date": forecast_date.isoformat(),
                "day": lead,
                "temperature_high": round(tmax, 1),
                "temperature_low": round(tmin, 1),
                "precipitation_mm": round(prcp, 1),
                "precipitation_probability": max(0, min(100, precip_prob)),
                "source": source,
            }
        )

    return forecasts
