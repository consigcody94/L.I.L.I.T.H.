"""
Synoptic Data PBC mesonet observations (formerly MesoWest).

Synoptic aggregates real-time and historical surface obs from ~75,000 stations
across CWOP, RAWS, AWOS, agricultural mesonets, DOT, and ASOS. Most networks
report at 5- to 15-minute cadence; some (NWS ASOS, RAWS) at 1-min via this API.

The free tier requires a `token` query parameter — sign up at
https://customer.synopticdata.com/. Set the token via env var SYNOPTIC_TOKEN
or pass --token. Without a token, this module is non-functional (we don't
ship a key).

Endpoints used:
    /v2/stations/timeseries  — historical bulk pull
    /v2/stations/latest      — most recent obs (for live forecasts)
    /v2/stations/metadata    — station inventory

This complements the GHCN station network: synoptic adds dense urban coverage,
fire weather (RAWS), and agricultural mesonets that GHCN ignores. For long-
range forecasts the extra spatial density is the main win.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm


SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
DEFAULT_VARS = "air_temp,dew_point_temperature,relative_humidity,wind_speed,wind_direction,wind_gust,altimeter,pressure,precip_accum_one_hour,visibility"


@dataclass
class SynopticConfig:
    token: str
    timeout: float = 60.0
    request_pause: float = 0.5  # seconds between requests; respect rate limits


class SynopticDownloader:
    """Pulls historical mesonet observations from Synoptic Data PBC."""

    def __init__(self, config: SynopticConfig, output_dir: Path | str = "data/raw/synoptic"):
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        url = f"{SYNOPTIC_BASE}/{endpoint.lstrip('/')}"
        params = {"token": self.cfg.token, **params}
        try:
            with httpx.Client(timeout=self.cfg.timeout, follow_redirects=True) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning(f"{endpoint}: {exc}")
            return None

        # Synoptic packs error info into a SUMMARY block; surface it loudly.
        summary = payload.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE", 1) != 1:
            logger.warning(f"Synoptic API error: {summary.get('RESPONSE_MESSAGE')}")
            return None
        return payload

    def metadata(
        self,
        bbox: Optional[List[float]] = None,
        state: Optional[str] = None,
        network: Optional[str] = None,
        complete: bool = True,
    ) -> pd.DataFrame:
        """Look up station metadata. `complete` adds period-of-record info."""
        params: Dict = {"complete": "1" if complete else "0"}
        if bbox:
            # Synoptic bbox is W,S,E,N
            params["bbox"] = ",".join(str(x) for x in bbox)
        if state:
            params["state"] = state
        if network:
            params["network"] = network

        payload = self._get("stations/metadata", params)
        if not payload:
            return pd.DataFrame()

        stations = payload.get("STATION", [])
        if not stations:
            return pd.DataFrame()

        # Flatten the most useful fields. The ``or "nan"`` idiom is wrong for
        # numeric 0 (an equator/Greenwich station would be coerced to NaN), so
        # we explicitly check for None and empty strings instead.
        def _to_float(v) -> float:
            if v is None or v == "":
                return float("nan")
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        rows = []
        for s in stations:
            rows.append({
                "stid": s.get("STID"),
                "name": s.get("NAME"),
                "latitude": _to_float(s.get("LATITUDE")),
                "longitude": _to_float(s.get("LONGITUDE")),
                "elevation": _to_float(s.get("ELEVATION")),
                "state": s.get("STATE"),
                "network": (s.get("MNET") or {}).get("SHORTNAME"),
                "period_start": (s.get("PERIOD_OF_RECORD") or {}).get("start"),
                "period_end": (s.get("PERIOD_OF_RECORD") or {}).get("end"),
            })
        return pd.DataFrame(rows)

    def timeseries(
        self,
        stations: List[str],
        start: str,
        end: str,
        variables: str = DEFAULT_VARS,
        force: bool = False,
    ) -> List[Path]:
        """Pull a time range for a set of stations.

        Args:
            stations: Synoptic station IDs (e.g. ["KORD", "C5984"]).
            start: "YYYYMMDDHHMM" UTC (Synoptic's required format).
            end:   "YYYYMMDDHHMM" UTC.
            variables: comma-separated Synoptic variable names.

        Cached as one parquet per station spanning the full requested range.
        """
        paths: List[Path] = []
        for stid in tqdm(stations, desc="Synoptic", unit="station"):
            cache = self.output_dir / f"{stid}_{start}_{end}.parquet"
            if cache.exists() and not force:
                paths.append(cache)
                continue

            payload = self._get(
                "stations/timeseries",
                {
                    "stid": stid,
                    "start": start,
                    "end": end,
                    "vars": variables,
                    "obtimezone": "utc",
                    "output": "json",
                },
            )
            time.sleep(self.cfg.request_pause)
            if not payload:
                continue

            stations_data = payload.get("STATION", [])
            if not stations_data:
                continue

            df = self._station_payload_to_df(stations_data[0])
            if df.empty:
                continue

            df.to_parquet(cache, index=False)
            paths.append(cache)

        return paths

    @staticmethod
    def _station_payload_to_df(station_block: Dict) -> pd.DataFrame:
        observations = station_block.get("OBSERVATIONS", {})
        ts = observations.get("date_time")
        if not ts:
            return pd.DataFrame()

        df = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True)})
        for key, values in observations.items():
            if key == "date_time":
                continue
            if not isinstance(values, list) or len(values) != len(ts):
                continue
            # Strip Synoptic's "_set_1" suffix for cleanliness.
            clean = key.replace("_set_1", "").replace("_set_1d", "")
            df[clean] = pd.to_numeric(values, errors="coerce")

        df["stid"] = station_block.get("STID")
        # See _to_float in metadata() for why we don't use ``or "nan"`` here.
        def _to_float(v) -> float:
            if v is None or v == "":
                return float("nan")
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
        df["latitude"] = _to_float(station_block.get("LATITUDE"))
        df["longitude"] = _to_float(station_block.get("LONGITUDE"))
        df["elevation"] = _to_float(station_block.get("ELEVATION"))
        return df

    def latest(self, bbox: List[float], variables: str = DEFAULT_VARS) -> pd.DataFrame:
        """Most-recent observations within a bbox — for live inference inputs."""
        payload = self._get(
            "stations/latest",
            {
                "bbox": ",".join(str(x) for x in bbox),
                "vars": variables,
                "obtimezone": "utc",
                "output": "json",
            },
        )
        if not payload:
            return pd.DataFrame()

        rows = []
        for station in payload.get("STATION", []):
            obs = station.get("OBSERVATIONS", {})
            if not obs:
                continue
            row = {
                "stid": station.get("STID"),
                "latitude": float(station.get("LATITUDE", "nan") or "nan"),
                "longitude": float(station.get("LONGITUDE", "nan") or "nan"),
                "elevation": float(station.get("ELEVATION", "nan") or "nan"),
            }
            for key, value in obs.items():
                if isinstance(value, dict) and "value" in value:
                    clean = key.replace("_value_1", "").replace("_value_1d", "")
                    row[clean] = pd.to_numeric(value["value"], errors="coerce")
                    if "date_time" in value:
                        row[f"{clean}_at"] = value["date_time"]
            rows.append(row)
        return pd.DataFrame(rows)


def get_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve a Synoptic token from CLI / env / a tokenfile, in that order."""
    if cli_token:
        return cli_token
    env = os.environ.get("SYNOPTIC_TOKEN")
    if env:
        return env
    token_file = Path.home() / ".synoptic_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Download Synoptic mesonet observations")
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--start", type=str, required=True, help="YYYYMMDDHHMM UTC")
    parser.add_argument("--end", type=str, required=True, help="YYYYMMDDHHMM UTC")
    parser.add_argument("--output-dir", type=str, default="data/raw/synoptic")
    parser.add_argument("--variables", type=str, default=DEFAULT_VARS)
    parser.add_argument("--token", type=str, default=None,
                        help="Synoptic API token. If omitted, read from $SYNOPTIC_TOKEN or ~/.synoptic_token.")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Just print station metadata and exit")
    args = parser.parse_args()

    token = get_token(args.token)
    if not token:
        raise SystemExit(
            "No Synoptic token found. Sign up at https://customer.synopticdata.com/, "
            "then either pass --token, set $SYNOPTIC_TOKEN, or write the token to ~/.synoptic_token."
        )

    dl = SynopticDownloader(SynopticConfig(token=token), output_dir=args.output_dir)

    if args.metadata_only:
        meta = dl.metadata(network=None)
        logger.info(f"{len(meta)} stations available")
        print(meta.head(20))
        return

    dl.timeseries(args.stations, args.start, args.end, args.variables)


if __name__ == "__main__":
    main()
