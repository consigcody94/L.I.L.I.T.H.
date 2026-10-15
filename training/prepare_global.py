"""Build a small global research cohort with calendar-safe, purged time splits.

This is a station experiment, not proof of coverage over the globe or oceans.
Only fully observed windows are used; exclusion counts and geography are reported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from data.providers.ghcn_records import BASE_URL, decode_daily, download, station_metadata

FEATURES = ("TMAX", "TMIN", "PRCP")


def temporal_split(input_start: date, target_start: date, target_end: date,
                   start_year: int = 1980) -> str | None:
    """Entire input+target window stays inside its split, including purge boundaries."""
    if input_start >= date(start_year, 1, 1) and target_end < date(2017, 1, 1):
        return "train"
    if input_start >= date(2017, 1, 1) and target_end < date(2021, 1, 1):
        return "validation"
    if input_start >= date(2021, 1, 1) and target_end < date(2026, 1, 1):
        return "test"
    return None


def choose_stations(metadata: dict, inventory: str, limit: int, start_year: int = 1980) -> list[dict]:
    coverage = defaultdict(set)
    for line in inventory.splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        sid, _, _, element, first, last = parts
        if element in FEATURES and int(first) <= start_year and int(last) >= 2024:
            coverage[sid].add(element)
    cells = defaultdict(list)
    for sid, elements in coverage.items():
        if len(elements) != 3 or sid not in metadata:
            continue
        station = metadata[sid]
        cell = (min(5, int((station["latitude"] + 90) // 30)),
                min(11, int((station["longitude"] + 180) // 30)))
        cells[cell].append(station)
    for candidates in cells.values():
        candidates.sort(key=lambda s: hashlib.sha256(s["id"].encode()).hexdigest())
    chosen = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for cell in sorted(cells):
            if depth < len(cells[cell]) and len(chosen) < limit:
                station = dict(cells[cell][depth])
                station["sampling_cell"] = list(cell)
                chosen.append(station)
                added = True
        if not added:
            break
        depth += 1
    # Freeze station holdouts before downloading/inspecting test outcomes.
    for index, station in enumerate(chosen):
        station["station_holdout"] = index % 5 == 0
    return chosen


def build_dataset(output: Path, cache: Path, limit: int = 72, horizon: int = 14,
                  start_year: int = 1980) -> dict:
    if not 1850 <= start_year <= 2000:
        raise ValueError("Research start year must be between 1850 and 2000")
    output.mkdir(parents=True, exist_ok=True)
    stations_text, stations_hash = download("ghcnd-stations.txt", cache, max_age=7 * 86400)
    inventory, inventory_hash = download("ghcnd-inventory.txt", cache, max_age=7 * 86400)
    stations = choose_stations(station_metadata(stations_text), inventory, limit, start_year)
    first, last = date(start_year, 1, 1), date(2025, 12, 31)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    day_index = {d.isoformat(): i for i, d in enumerate(dates)}
    doy = np.array([d.timetuple().tm_yday - 1 for d in dates])
    splits = {k: defaultdict(list) for k in ("train", "validation", "test", "station_holdout")}
    manifests, failures = [], []
    station_arrays = {}
    # Serial fetching avoids a burst against NOAA. Parsing is bounded by the cohort.
    for station in stations:
        sid = station["id"]
        try:
            text, digest = download(f"all/{sid}.dly", cache)
        except OSError as exc:
            failures.append({"station_id": sid, "reason": type(exc).__name__})
            print(f"Unavailable {sid}: {type(exc).__name__}", flush=True)
            continue
        array = np.full((len(dates), 3), np.nan, dtype=np.float32)
        rejected = 0
        for row in decode_daily(text, sid):
            if row["element"] not in FEATURES or row["date"] not in day_index:
                continue
            if row["value"] is None:
                rejected += 1
            else:
                array[day_index[row["date"]], FEATURES.index(row["element"])] = row["value"]
        station_arrays[sid] = array
        manifests.append({**station, "url": BASE_URL + f"all/{sid}.dly", "sha256": digest,
                          "missing_or_rejected_elements": rejected,
                          "complete_days": int(np.isfinite(array).all(axis=1).sum())})
        print(f"Decoded {sid}: {manifests[-1]['complete_days']} complete days", flush=True)

    # Fit empirical seasonal baselines only on training stations and training dates.
    normals = {}
    for station in manifests:
        if station["station_holdout"]:
            continue
        array = station_arrays[station["id"]]
        train_dates = np.array([d.year < 2017 for d in dates])
        climate = []
        for day in range(366):
            distance = np.minimum(abs(doy - day), 366 - abs(doy - day))
            rows = array[train_dates & (distance <= 15)]
            # Missing normal bins exclude a baseline case rather than creating truth.
            with np.errstate(invalid="ignore"):
                climate.append(np.nanmean(rows, axis=0))
        normals[station["id"]] = np.array(climate, dtype=np.float32)
    if not normals:
        raise RuntimeError("No usable training stations")

    for station in manifests:
        sid = station["id"]
        array = station_arrays[sid]
        if sid in normals:
            normal_sid = sid
        else:
            # Great-circle nearest *training* station; no held-out target statistics.
            lat = np.radians(station["latitude"])
            lon = np.radians(station["longitude"])
            def distance(candidate):
                a = np.radians(candidate["latitude"])
                b = np.radians(candidate["longitude"])
                return np.arccos(np.clip(np.sin(lat)*np.sin(a) + np.cos(lat)*np.cos(a)*np.cos(lon-b), -1, 1))
            normal_sid = min((s for s in manifests if s["id"] in normals), key=distance)["id"]
        station["baseline_normal_station"] = normal_sid
        counts = defaultdict(int)
        for start in range(0, len(dates) - 30 - horizon + 1, 14):
            target = start + 30
            split = temporal_split(dates[start], dates[target], dates[target + horizon - 1], start_year)
            if split is None or (station["station_holdout"] and split != "test"):
                continue
            if station["station_holdout"]:
                split = "station_holdout"
            x, y = array[start:target], array[target:target + horizon]
            climate = normals[normal_sid][doy[target:target + horizon]]
            if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(climate).all():
                counts["excluded_windows"] += 1
                continue
            bucket = splits[split]
            bucket["X"].append(x)
            bucket["Y"].append(y)
            # Day encoding matches inference's *last input date* convention.
            bucket["meta"].append([station["latitude"] / 90, station["longitude"] / 180,
                                   max(-500, station["elevation"]) / 5000,
                                   dates[target - 1].timetuple().tm_yday / 365])
            bucket["station_id"].append(sid)
            bucket["target_start"].append(dates[target].isoformat())
            bucket["climatology"].append(climate)
            counts[split] += 1
        station["window_counts"] = dict(counts)
    counts = {}
    for split, values in splits.items():
        if not values["X"]:
            raise RuntimeError(f"Empty split: {split}; inspect cohort completeness")
        arrays = {k: np.asarray(v, dtype=np.float32 if k in {"X", "Y", "meta", "climatology"} else str)
                  for k, v in values.items()}
        np.savez_compressed(output / f"{split}.npz", **arrays)
        counts[split] = len(values["X"])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "features": list(FEATURES),
        "units": ["degC", "degC", "mm"], "input_days": 30, "target_days": horizon,
        "stride_days": 14, "station_sampling": "round_robin_30_degree_cells_hash_order",
        "requested_stations": limit, "stations": manifests, "download_failures": failures,
        "metadata_sha256": stations_hash, "inventory_sha256": inventory_hash,
        "counts": counts, "split_policy": {"train": f"{start_year}-2016", "validation": "2017-2020",
        "test": "2021-2025", "purge": "entire input and target must remain in one split"},
        "limitations": ["Land station sample, no ocean or uniform global coverage",
                         "Historical data revisions; not an operational as-of replay",
                         "Complete-window selection favors well-instrumented stations",
                         "GHCN station reporting days may differ from local calendar days",
                         "Station holdout is not a contiguous-region holdout"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stations", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--start-year", type=int, default=1980)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, args.cache, args.stations, args.horizon, args.start_year)["counts"]))
