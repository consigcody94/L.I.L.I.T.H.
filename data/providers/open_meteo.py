"""
Open-Meteo data provider - free, keyless real weather data.

Open-Meteo (https://open-meteo.com) serves real observations and forecasts with
no API key and a generous non-commercial free tier. This provider is the project's
bridge from "trained on real data" to "served with real data", supplying:

  * recent_history()      - real recent daily observations to seed the model
  * reference_forecast()  - a real short-range forecast for near-term blending/fallback
  * daily_observations()  - real observed values for verifying past predictions
  * geocode()             - place name -> coordinates

It uses only the Python standard library (urllib) so it adds no dependency and can
run anywhere. Every method fails soft: on any network/parse error it returns None
(or an empty result) and logs, never raising into the caller.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a project dep but keep this usable standalone
    import logging

    logger = logging.getLogger("open_meteo")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Open-Meteo's daily forecast horizon and historical "past_days" lookback limits.
MAX_FORECAST_DAYS = 16
MAX_PAST_DAYS = 92
# The archive (ERA5) lags real time by ~5 days; use the forecast API's past_days
# window for anything more recent than this.
ARCHIVE_LAG_DAYS = 6


class OpenMeteoProvider:
    """Keyless client for Open-Meteo with a small in-memory TTL cache."""

    def __init__(self, timeout: float = 15.0, cache_ttl: float = 900.0):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Low-level HTTP
    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = urllib.parse.urlencode(params, doseq=True)
        full_url = f"{url}?{query}"

        cached = self._cache.get(full_url)
        if cached and (time.time() - cached["ts"]) < self.cache_ttl:
            return cached["data"]

        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "LILITH/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
            self._cache[full_url] = {"ts": time.time(), "data": data}
            return data
        except Exception as exc:  # noqa: BLE001 - fail soft by design
            logger.warning(f"Open-Meteo request failed ({url}): {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recent_history(
        self, latitude: float, longitude: float, days: int = 30
    ) -> Optional[np.ndarray]:
        """
        Real recent daily observations as a [days, 3] array (TMAX, TMIN, PRCP),
        ordered oldest -> newest and ending yesterday. Returns None on failure.
        """
        days = max(1, min(days, MAX_PAST_DAYS))
        data = self._get(
            FORECAST_URL,
            {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "past_days": min(days + 1, MAX_PAST_DAYS),
                "forecast_days": 1,
                "timezone": "auto",
            },
        )
        rows = self._daily_rows(data)
        if not rows:
            return None

        today = datetime.now().date()
        # Keep only complete days strictly before today, then take the last `days`.
        past = [r for r in rows if r["date"] < today]
        past = past[-days:]
        if len(past) < max(7, days // 2):  # need a meaningful window to seed the model
            return None

        arr = np.array(
            [[r["tmax"], r["tmin"], max(0.0, r["prcp"])] for r in past],
            dtype=np.float32,
        )
        return arr

    def reference_forecast(
        self, latitude: float, longitude: float, days: int = MAX_FORECAST_DAYS
    ) -> Optional[Dict[str, List[Optional[float]]]]:
        """
        Real short-range daily forecast starting tomorrow, aligned to model lead
        day 1. Returns parallel lists (tmax, tmin, prcp, precip_prob, dates) or None.
        """
        days = max(1, min(days, MAX_FORECAST_DAYS))
        data = self._get(
            FORECAST_URL,
            {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,precipitation_probability_max"
                ),
                "forecast_days": min(days + 1, MAX_FORECAST_DAYS),
                "timezone": "auto",
            },
        )
        if not data or "daily" not in data:
            return None

        daily = data["daily"]
        times = daily.get("time", [])
        tomorrow = datetime.now().date() + timedelta(days=1)

        out: Dict[str, List[Optional[float]]] = {
            "dates": [],
            "tmax": [],
            "tmin": [],
            "prcp": [],
            "precip_prob": [],
        }
        for idx, day_str in enumerate(times):
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue
            if day < tomorrow:  # align lead day 1 to tomorrow
                continue
            out["dates"].append(day_str)
            out["tmax"].append(_at(daily.get("temperature_2m_max"), idx))
            out["tmin"].append(_at(daily.get("temperature_2m_min"), idx))
            out["prcp"].append(_at(daily.get("precipitation_sum"), idx))
            out["precip_prob"].append(_at(daily.get("precipitation_probability_max"), idx))

        return out if out["dates"] else None

    def daily_observations(
        self, latitude: float, longitude: float, target_day: date
    ) -> Optional[Dict[str, float]]:
        """
        Real observed (TMAX, TMIN, PRCP) for a single past day, used to verify
        predictions. Picks the archive or the forecast past_days window depending
        on how recent the date is. Returns None if unavailable.
        """
        today = datetime.now().date()
        if target_day >= today:
            return None

        age_days = (today - target_day).days
        if age_days <= ARCHIVE_LAG_DAYS:
            data = self._get(
                FORECAST_URL,
                {
                    "latitude": round(latitude, 4),
                    "longitude": round(longitude, 4),
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "past_days": min(age_days + 2, MAX_PAST_DAYS),
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )
        else:
            data = self._get(
                ARCHIVE_URL,
                {
                    "latitude": round(latitude, 4),
                    "longitude": round(longitude, 4),
                    "start_date": target_day.isoformat(),
                    "end_date": target_day.isoformat(),
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "auto",
                },
            )

        for row in self._daily_rows(data):
            if row["date"] == target_day:
                return {"tmax": row["tmax"], "tmin": row["tmin"], "prcp": max(0.0, row["prcp"])}
        return None

    def geocode(self, name: str) -> Optional[Dict[str, Any]]:
        """Resolve a place name to coordinates and a display name."""
        if not name or not name.strip():
            return None
        data = self._get(GEOCODE_URL, {"name": name.strip(), "count": 1, "language": "en"})
        results = (data or {}).get("results") or []
        if not results:
            return None
        r = results[0]
        label = r.get("name", name)
        if r.get("admin1"):
            label = f"{label}, {r['admin1']}"
        if r.get("country_code"):
            label = f"{label}, {r['country_code']}"
        return {
            "name": label,
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "country": r.get("country_code"),
            "admin1": r.get("admin1"),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _daily_rows(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten Open-Meteo's column-oriented ``daily`` block into row dicts."""
        if not data or "daily" not in data:
            return []
        daily = data["daily"]
        times = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        prcp = daily.get("precipitation_sum", [])

        rows: List[Dict[str, Any]] = []
        for idx, day_str in enumerate(times):
            t_max = _at(tmax, idx)
            t_min = _at(tmin, idx)
            if t_max is None or t_min is None:  # skip incomplete days
                continue
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue
            rows.append(
                {
                    "date": day,
                    "tmax": float(t_max),
                    "tmin": float(t_min),
                    "prcp": float(_at(prcp, idx) or 0.0),
                }
            )
        return rows


def _at(seq: Optional[List[Any]], idx: int) -> Optional[float]:
    """Safe indexed access that maps missing/None entries to None."""
    if seq is None or idx >= len(seq):
        return None
    val = seq[idx]
    return None if val is None else val
