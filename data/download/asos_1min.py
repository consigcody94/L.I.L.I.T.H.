"""
ASOS 1-minute observations from NCEI.

ASOS (Automated Surface Observing System) 1-min data is the highest-resolution
publicly available U.S. surface observation feed: ~900 active stations, 1-min
cadence, going back to ~2000 for most stations. It contains precipitation,
visibility, temperature, dew point, wind, gust, ceiling, MSLP, and station
pressure — about 4x the variables in GHCN-Daily and ~1440x the cadence.

API:
    https://www.ncei.noaa.gov/access/services/data/v1
    dataset: 'asos-1min'

This downloader pulls month-chunks (NCEI throttles long requests), saves raw
CSV per station-month, and produces a unified parquet for training.

Why bother:
    - GHCN-Daily forces the model to predict daily aggregates (TMAX/TMIN). The
      1-min stream lets the model learn diurnal cycles, gust statistics, and
      sub-daily precipitation timing — all of which are commercially valuable
      forecast targets that the daily-only model literally cannot produce.

Usage:
    python -m data.download.asos_1min --stations KORD KJFK --start 2020-01 --end 2020-12
"""
from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


NCEI_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"

# Variables exposed by the ASOS 1-min feed that we care about for forecasting.
DEFAULT_VARS = [
    "tmpc",        # temperature (°C)
    "dwpc",        # dew point (°C)
    "relh",        # relative humidity (%)
    "drct",        # wind direction (deg)
    "sknt",        # wind speed (knots)
    "gust",        # gust speed (knots)
    "alti",        # altimeter setting (inHg)
    "mslp",        # MSLP (hPa)
    "p01i",        # 1-hr precip (inches)
    "vsby",        # visibility (miles)
    "skyc1", "skyl1",  # sky condition / cloud base lvl 1
]


@dataclass
class FetchSpec:
    station: str          # 4-letter ICAO id, e.g. "KORD"
    start: str            # "YYYY-MM"
    end: str              # "YYYY-MM" inclusive
    variables: List[str]


def _month_iter(start_ym: str, end_ym: str) -> Iterable[pd.Timestamp]:
    start = pd.Timestamp(start_ym + "-01")
    end = pd.Timestamp(end_ym + "-01")
    cur = start
    while cur <= end:
        yield cur
        cur = (cur + pd.DateOffset(months=1)).normalize()


class ASOS1MinDownloader:
    """Hits the NCEI 'data/v1' service for asos-1min. Polite by default."""

    def __init__(
        self,
        output_dir: Path | str = "data/raw/asos_1min",
        timeout: float = 60.0,
        sleep_between: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.sleep_between = sleep_between

    def _request_month(
        self,
        client: httpx.Client,
        station: str,
        month_start: pd.Timestamp,
        variables: List[str],
    ) -> Optional[pd.DataFrame]:
        # NCEI accepts inclusive start/end dates.
        end_of_month = (month_start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
        params = {
            "dataset": "asos-1min",
            "stations": station,
            "startDate": month_start.strftime("%Y-%m-%dT00:00:00"),
            "endDate": end_of_month.strftime("%Y-%m-%dT23:59:00"),
            "dataTypes": ",".join(variables),
            "format": "csv",
            "units": "metric",
        }
        try:
            response = client.get(NCEI_BASE, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"{station} {month_start:%Y-%m}: {exc}")
            return None

        text = response.text
        if not text or text.startswith("<") or "DATE" not in text.split("\n", 1)[0].upper():
            return None

        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            logger.warning(f"{station} {month_start:%Y-%m}: parse failed: {exc}")
            return None

        if df.empty:
            return None

        # Normalize the timestamp column name (varies across NCEI datasets).
        date_col = next((c for c in df.columns if c.upper() in {"DATE", "TIME", "OBSERVATION_DATE"}), None)
        if date_col:
            df = df.rename(columns={date_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

        return df

    def download(
        self,
        stations: List[str],
        start_ym: str,
        end_ym: str,
        variables: Optional[List[str]] = None,
        force: bool = False,
    ) -> List[Path]:
        """Download ASOS 1-min data for every station × month in [start_ym, end_ym].

        One CSV per station per month is cached under output_dir/<STATION>/.
        Returns the list of cached paths (existing + new).
        """
        variables = variables or DEFAULT_VARS
        paths: List[Path] = []

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            total = sum(1 for _ in _month_iter(start_ym, end_ym)) * len(stations)
            with tqdm(total=total, desc="ASOS 1-min", unit="month") as pbar:
                for station in stations:
                    station_dir = self.output_dir / station
                    station_dir.mkdir(parents=True, exist_ok=True)

                    for month in _month_iter(start_ym, end_ym):
                        cache = station_dir / f"{month:%Y-%m}.csv"
                        if cache.exists() and not force:
                            paths.append(cache)
                            pbar.update(1)
                            continue

                        df = self._request_month(client, station, month, variables)
                        if df is not None and not df.empty:
                            df.to_csv(cache, index=False)
                            paths.append(cache)
                        pbar.update(1)
                        time.sleep(self.sleep_between)  # be polite to NCEI

        return paths

    def aggregate_to_hourly(
        self,
        cached_paths: List[Path],
        output: Path | str = "data/raw/asos_1min/hourly.parquet",
    ) -> pd.DataFrame:
        """Aggregate raw 1-min observations into hourly mean/max/min summaries.

        The model trains on hourly tokens so we don't need to drag the full
        1-min stream through the dataloader — the high-frequency value is in
        statistics like gust max and precip totals, which we can pre-compute.
        """
        frames: List[pd.DataFrame] = []
        for path in tqdm(cached_paths, desc="Aggregating to hourly"):
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue

            station = path.parent.name
            df_indexed = df.set_index("timestamp")
            numeric = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                continue

            # Per-station resample. Doing the groupby+resample at once produces
            # a 3-level column MultiIndex on pandas 3.x (groupby_key, agg_func,
            # base_column) which breaks the simple 2-tuple unpack — cleaner to
            # resample one station at a time and append the station tag after.
            agg = df_indexed[numeric].resample("1h").agg(["mean", "max", "min"])
            # 2-level MultiIndex now: (base_column, agg_func)
            agg.columns = [f"{base}_{stat}" for base, stat in agg.columns]
            agg = agg.reset_index()
            agg["station"] = station
            frames.append(agg)

        if not frames:
            logger.warning("No data aggregated.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out, index=False)
        logger.success(f"Aggregated {len(combined):,} hourly rows -> {out}")
        return combined


def main():
    parser = argparse.ArgumentParser(description="Download ASOS 1-min observations from NCEI")
    parser.add_argument("--stations", nargs="+", required=True,
                        help="ICAO station ids (e.g. KORD KJFK KLAX). Use --station-list-file to pass many.")
    parser.add_argument("--station-list-file", type=str, default=None,
                        help="Optional file with one station ID per line; appended to --stations.")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM")
    parser.add_argument("--output-dir", type=str, default="data/raw/asos_1min")
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARS)
    parser.add_argument("--no-aggregate", action="store_true", help="Skip hourly-aggregation step")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stations = list(args.stations)
    if args.station_list_file:
        with open(args.station_list_file) as f:
            stations += [line.strip().upper() for line in f if line.strip()]
    stations = sorted(set(stations))

    dl = ASOS1MinDownloader(args.output_dir)
    paths = dl.download(stations, args.start, args.end, args.variables, force=args.force)
    if paths and not args.no_aggregate:
        dl.aggregate_to_hourly(paths)


if __name__ == "__main__":
    main()
