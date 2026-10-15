"""Lossless, dependency-free GHCN-Daily decoding for training and verification.

Source format: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
Observation dates are station reporting dates, not guaranteed midnight-to-midnight.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date
import hashlib
from pathlib import Path
import re
import time
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
SCALES = {"TMAX": 0.1, "TMIN": 0.1, "PRCP": 0.1, "SNOW": 1.0, "SNWD": 1.0}
UNITS = {"TMAX": "degC", "TMIN": "degC", "PRCP": "mm", "SNOW": "mm", "SNWD": "mm"}


def download(relative_path: str, cache_dir: Path, max_age: float = 86400) -> tuple[str, str]:
    """Cache NOAA text; return text and SHA256 of the exact downloaded bytes."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", relative_path) or ".." in relative_path:
        raise ValueError("Invalid NOAA path")
    path = Path(cache_dir) / relative_path
    if not path.exists() or time.time() - path.stat().st_mtime > max_age:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(BASE_URL + relative_path, headers={"User-Agent": "Lilith-research/0.1"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    raw = response.read()
                with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    handle.write(raw)
                    temporary = Path(handle.name)
                temporary.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    raw = path.read_bytes()
    return raw.decode("ascii", errors="replace"), hashlib.sha256(raw).hexdigest()


def decode_daily(text: str, station_id: str | None = None) -> list[dict]:
    """Retain missing values and M/Q/S flags; never interpolate verification truth."""
    records = []
    for line in text.splitlines():
        if len(line) < 269:
            continue
        sid, element = line[:11], line[17:21]
        if element not in SCALES or (station_id and sid != station_id):
            continue
        try:
            year, month = int(line[11:15]), int(line[15:17])
            days = monthrange(year, month)[1]
        except ValueError:
            continue
        for day in range(1, days + 1):
            start = 21 + (day - 1) * 8
            try:
                raw_value = int(line[start:start + 5])
            except ValueError:
                continue
            mflag, qflag, sflag = line[start + 5:start + 8]
            usable = raw_value != -9999 and not qflag.strip()
            records.append({
                "station_id": sid, "date": date(year, month, day).isoformat(),
                "element": element, "raw_value": raw_value,
                "value": raw_value * SCALES[element] if usable else None,
                "unit": UNITS[element], "measurement_flag": mflag.strip(),
                "quality_flag": qflag.strip(), "source_flag": sflag.strip(),
                "trace": mflag == "T", "usable": usable,
                "day_definition": "station_reporting_day",
            })
    return records


def station_metadata(text: str) -> dict[str, dict]:
    result = {}
    for line in text.splitlines():
        if len(line) < 71:
            continue
        try:
            result[line[:11]] = {
                "id": line[:11], "latitude": float(line[12:20]),
                "longitude": float(line[21:30]), "elevation": float(line[31:37]),
                "name": line[41:71].strip(), "country_code_ghcn": line[:2],
            }
        except ValueError:
            continue
    return result


def historical_station(station_id: str, start: date, end: date, cache: Path,
                       variables: list[str]) -> dict:
    if not re.fullmatch(r"[A-Z0-9]{11}", station_id):
        raise ValueError("Invalid GHCN station identifier")
    if end < start or (end - start).days > 366 * 300:
        raise ValueError("Date range must be ordered and at most 300 years")
    allowed = {"TMAX": "temperature_max", "TMIN": "temperature_min", "PRCP": "precipitation"}
    if not variables or set(variables) - allowed.keys():
        raise ValueError("Supported variables: TMAX, TMIN, PRCP")
    metadata, _ = download("ghcnd-stations.txt", cache, max_age=7 * 86400)
    station = station_metadata(metadata).get(station_id)
    if station is None:
        raise KeyError(station_id)
    text, digest = download(f"all/{station_id}.dly", cache)
    by_day = {}
    for row in decode_daily(text, station_id):
        if row["element"] not in variables or not start.isoformat() <= row["date"] <= end.isoformat():
            continue
        daily = by_day.setdefault(row["date"], {"date": row["date"]})
        daily[allowed[row["element"]]] = row["value"]
    observations = [by_day[day] for day in sorted(by_day)]
    return {"station_id": station_id, "station_name": station["name"],
            "observations": observations, "total_observations": len(observations),
            "source_url": BASE_URL + f"all/{station_id}.dly", "source_sha256": digest,
            "day_definition": "station_reporting_day", "latitude": station["latitude"],
            "longitude": station["longitude"], "requested_start": start.isoformat(),
            "requested_end": end.isoformat(), "quality_policy": "missing and nonblank QFLAG become null"}
