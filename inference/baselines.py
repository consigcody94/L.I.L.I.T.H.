"""
Trivial forecasting baselines: climatology and persistence.

Why this matters: the climatology-replacement bug we just fixed in
``simple_forecaster.py`` silently shipped for an unknown amount of time
because nobody compared the "model" output against an actual climatology
baseline. If they had, the numbers would have been suspiciously similar
and the bug would have been obvious.

This module gives you that comparison. Run your model AND these baselines
on the same chronological val split and report all three RMSEs side by
side. If your trained model isn't meaningfully beating climatology +
persistence, something is wrong.

References:
    Persistence baseline: tomorrow = today.
    Climatology baseline: predict the long-run average for that
        latitude / day-of-year. The strongest *non-trivial* baseline a
        weather model has to beat to justify its existence.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np


def persistence_forecast(
    history: np.ndarray,
    forecast_days: int,
) -> np.ndarray:
    """tomorrow = today, repeated.

    Args:
        history: (input_days, n_features) — only the last day is used.
        forecast_days: how many days to repeat the last observation.

    Returns:
        (forecast_days, n_features) — the last day, broadcast forward.
    """
    if history.ndim != 2:
        raise ValueError(f"history must be (input_days, n_features), got {history.shape}")
    last = history[-1]
    return np.broadcast_to(last, (forecast_days, last.shape[0])).copy()


def daily_climatology_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int,
    start_date: Optional[datetime] = None,
) -> np.ndarray:
    """Latitude- and day-of-year-based temperature/precip climatology.

    Returns ``(forecast_days, 3)`` for [TMAX, TMIN, PRCP].

    The model is intentionally simple — coastal/continental adjustments
    are deliberately omitted because we want a *trivial* baseline that's
    easy to beat. If the trained model can't beat this, the model is broken.
    """
    if start_date is None:
        start_date = datetime.now().date() + timedelta(days=1)
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    abs_lat = abs(latitude)

    # Latitude bands -> annual mean and seasonal amplitude (°C).
    # Calibrated roughly against US station climatology.
    if abs_lat < 15:
        annual_mean, seasonal_amp, diurnal = 27.0, 2.0, 10.0
    elif abs_lat < 28:
        annual_mean, seasonal_amp, diurnal = 22.0, 7.0, 11.0
    elif abs_lat < 35:
        annual_mean, seasonal_amp, diurnal = 17.0, 10.0, 12.0
    elif abs_lat < 42:
        annual_mean, seasonal_amp, diurnal = 11.0, 14.0, 10.0
    elif abs_lat < 48:
        annual_mean, seasonal_amp, diurnal = 7.0, 17.0, 11.0
    elif abs_lat < 55:
        annual_mean, seasonal_amp, diurnal = 2.0, 20.0, 10.0
    else:
        annual_mean, seasonal_amp, diurnal = -5.0, 22.0, 9.0

    # Northern hemisphere summer peaks ~ day 200; southern ~ day 15.
    phase_shift = 200 if latitude >= 0 else 15

    out = np.zeros((forecast_days, 3), dtype=np.float32)
    for i in range(forecast_days):
        date = start_date + timedelta(days=i)
        doy = date.timetuple().tm_yday
        seasonal = seasonal_amp * np.cos(2 * np.pi * (doy - phase_shift) / 365.0)
        mean = annual_mean + seasonal
        out[i, 0] = mean + diurnal / 2  # TMAX
        out[i, 1] = mean - diurnal / 2  # TMIN
        out[i, 2] = 2.0  # PRCP — flat 2mm/day prior, zero seasonal info
    return out


def evaluate_baselines(
    X_val: np.ndarray,
    Y_val: np.ndarray,
    meta_val: np.ndarray,
    dates_val: Optional[np.ndarray] = None,
    Y_mean: Optional[np.ndarray] = None,
    Y_std: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE / MAE for persistence + climatology on a held-out set.

    Args:
        X_val: (N, input_days, n_features) — input histories (NORMALIZED).
        Y_val: (N, target_days, n_features) — true targets (NORMALIZED).
        meta_val: (N, 4) — lat, lon, elev, day_of_year, all NORMALIZED
                  (meta[:, 0] in [-1, 1], meta[:, 1] in [-1, 1]).
        dates_val: (N,) datetime64[D] — optional, used as climatology start dates.
        Y_mean, Y_std: per-channel denormalization stats. If supplied, errors
                       are reported in original units (°C for temps).

    Returns:
        {"persistence": {"rmse_temp": ..., "mae_temp": ...},
         "climatology": {...}}
    """
    if Y_val.ndim != 3:
        raise ValueError(f"Y_val must be (N, target_days, n_features), got {Y_val.shape}")
    n, target_days, n_features = Y_val.shape

    # Denormalize for human-interpretable metrics.
    if Y_mean is None:
        Y_mean = np.zeros(n_features)
    if Y_std is None:
        Y_std = np.ones(n_features)
    Y_real = Y_val * Y_std + Y_mean

    # --- persistence ---
    last_x = X_val[:, -1, :]  # (N, n_features) — assumed in same normalized space
    persist_norm = np.broadcast_to(last_x[:, None, :], Y_val.shape)
    persist_real = persist_norm * Y_std + Y_mean

    # --- climatology ---
    clim_real = np.zeros_like(Y_real)
    # Reverse meta normalization: lat scaled by 90, lon by 180.
    lats = meta_val[:, 0] * 90.0
    lons = meta_val[:, 1] * 180.0
    for i in range(n):
        start = (
            datetime.utcfromtimestamp(dates_val[i].astype("datetime64[s]").astype(int)).date()
            if dates_val is not None
            else None
        )
        clim_real[i] = daily_climatology_forecast(
            latitude=float(lats[i]),
            longitude=float(lons[i]),
            forecast_days=target_days,
            start_date=start,
        )[:, :n_features]

    def metrics(pred: np.ndarray, label: str) -> Dict[str, float]:
        diff_temp = pred[..., :2] - Y_real[..., :2]
        rmse_temp = float(np.sqrt(np.mean(diff_temp ** 2)))
        mae_temp = float(np.mean(np.abs(diff_temp)))
        out = {"rmse_temp": rmse_temp, "mae_temp": mae_temp, "label": label}
        if n_features >= 3:
            diff_prcp = pred[..., 2] - Y_real[..., 2]
            out["rmse_prcp"] = float(np.sqrt(np.mean(diff_prcp ** 2)))
        return out

    return {
        "persistence": metrics(persist_real, "persistence"),
        "climatology": metrics(clim_real, "climatology"),
    }


def assert_model_beats_baselines(
    model_rmse_temp: float,
    baselines: Dict[str, Dict[str, float]],
    margin: float = 0.0,
) -> None:
    """Loud failure if the trained model is no better than persistence.

    Useful as a CI gate: if a future refactor reintroduces the inference-
    discard bug or breaks training, this assertion catches it before the
    bad model ships.

    Args:
        model_rmse_temp: temperature RMSE of the trained model on val.
        baselines: output of :func:`evaluate_baselines`.
        margin: required RMSE gap below the worst baseline (e.g. 0.5 °C).
                Default 0 just requires *strictly better* than baselines.

    Raises:
        AssertionError if the model fails to beat both baselines.
    """
    persist = baselines["persistence"]["rmse_temp"]
    clim = baselines["climatology"]["rmse_temp"]
    worst = max(persist, clim)
    if model_rmse_temp >= worst - margin:
        raise AssertionError(
            f"Model RMSE {model_rmse_temp:.3f}°C is not better than the worst "
            f"trivial baseline ({worst:.3f}°C — persistence={persist:.3f}, "
            f"climatology={clim:.3f}). Either training is broken, the inference "
            f"path is silently overriding the model, or the data pipeline is "
            f"feeding the model garbage. Investigate before shipping."
        )
