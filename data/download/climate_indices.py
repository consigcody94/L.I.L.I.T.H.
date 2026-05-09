"""
Climate index downloader.

Pulls the major teleconnection indices used for sub-seasonal-to-seasonal (S2S)
forecasting from NOAA's Climate Prediction Center, NCEI, and ESRL/PSL. These
indices are kilobytes each and contain most of the long-range predictability
signal for surface variables — wiring them into the model is the cheapest
sub-monthly skill improvement available.

Indices supported:
    enso  : ONI (Oceanic Niño Index 3-month running SST anomaly)        — monthly
    mei   : Multivariate ENSO Index v2                                   — bi-monthly
    nao   : North Atlantic Oscillation                                   — daily + monthly
    pdo   : Pacific Decadal Oscillation                                  — monthly
    ao    : Arctic Oscillation                                           — daily + monthly
    mjo   : Madden-Julian Oscillation (RMM1, RMM2 from Wheeler-Hendon)   — daily
    pna   : Pacific/North American pattern                               — daily + monthly
    soi   : Southern Oscillation Index                                   — monthly

Output:
    data/raw/climate/<index>.csv  with columns: date, value(s).
    data/raw/climate/all.parquet  combined daily index for fast loading.
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class IndexSource:
    """Where to fetch one index from and how to parse it."""

    name: str
    url: str
    parser: Callable[[str], pd.DataFrame]
    cadence: str  # "daily" | "monthly" | "bimonthly"


def _parse_oni(text: str) -> pd.DataFrame:
    """ONI from CPC: 'SEAS YR TOTAL ANOM' columns where SEAS is e.g. 'DJF'.

    CPC's convention is that YR labels the *last* month of the rolling
    3-month season. So 'DJF 2020' = Dec 2019, Jan 2020, Feb 2020 — center
    month is January 2020 (the listed year, not year-1). Likewise 'NDJ
    2019' = Nov/Dec 2019 + Jan 2020, centered on December 2019.

    We pin each season to its center month so monthly forecasts line up
    with daily series naturally.
    """
    seas_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in seas_to_month:
            continue
        try:
            year = int(parts[1])
            anom = float(parts[3])
        except ValueError:
            continue
        month = seas_to_month[parts[0]]
        rows.append({"date": pd.Timestamp(year=year, month=month, day=15), "oni": anom})
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def _parse_cpc_monthly_table(value_col: str) -> Callable[[str], pd.DataFrame]:
    """Factory for parsers of CPC's classic 'YEAR Jan Feb ... Dec' tables (NAO, AO, PNA)."""

    def parse(text: str) -> pd.DataFrame:
        rows: List[Dict] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            for month, val in enumerate(parts[1:13], start=1):
                try:
                    v = float(val)
                except ValueError:
                    continue
                if v == -99.9 or v == -999.0:
                    continue
                rows.append({"date": pd.Timestamp(year=year, month=month, day=15), value_col: v})
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return parse


def _parse_cpc_daily(value_col: str) -> Callable[[str], pd.DataFrame]:
    """Factory for CPC's daily index tables (NAO, AO, PNA daily): YYYY MM DD value."""

    def parse(text: str) -> pd.DataFrame:
        rows: List[Dict] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                date = pd.Timestamp(year=int(parts[0]), month=int(parts[1]), day=int(parts[2]))
                v = float(parts[3])
            except (ValueError, TypeError):
                continue
            if v == -999.0 or v == -99.9:
                continue
            rows.append({"date": date, value_col: v})
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return parse


def _parse_pdo(text: str) -> pd.DataFrame:
    """NCEI ERSST PDO: same format as CPC NAO monthly table."""
    return _parse_cpc_monthly_table("pdo")(text)


def _parse_mei(text: str) -> pd.DataFrame:
    """MEI v2 from PSL: header rows, then 'YYYY DJ JF FM ... ND'."""
    seasons = ["DJ", "JF", "FM", "MA", "AM", "MJ", "JJ", "JA", "AS", "SO", "ON", "ND"]
    rows: List[Dict] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month, val in enumerate(parts[1:13], start=1):
            try:
                v = float(val)
            except ValueError:
                continue
            if abs(v) > 9:  # MEI is bounded ~ -3 to +3; -999 sentinel
                continue
            rows.append({"date": pd.Timestamp(year=year, month=month, day=15), "mei": v})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _parse_mjo(text: str) -> pd.DataFrame:
    """BoM Wheeler-Hendon RMM: 'YYYY MM DD RMM1 RMM2 phase amplitude ...'."""
    rows: List[Dict] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            date = pd.Timestamp(year=int(parts[0]), month=int(parts[1]), day=int(parts[2]))
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
        except (ValueError, TypeError):
            continue
        if abs(rmm1) > 100 or abs(rmm2) > 100:  # missing flag
            continue
        rows.append({"date": date, "mjo_rmm1": rmm1, "mjo_rmm2": rmm2})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _parse_soi(text: str) -> pd.DataFrame:
    """CPC SOI: 'YYYYMM value' per line, sometimes with header noise."""
    rows: List[Dict] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            ym = parts[0]
            if len(ym) != 6:
                continue
            year = int(ym[:4])
            month = int(ym[4:])
            v = float(parts[1])
        except ValueError:
            continue
        if v == -999.9:
            continue
        rows.append({"date": pd.Timestamp(year=year, month=month, day=15), "soi": v})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


SOURCES: Dict[str, IndexSource] = {
    "enso": IndexSource(
        name="ONI",
        url="https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
        parser=_parse_oni,
        cadence="monthly",
    ),
    "mei": IndexSource(
        name="MEI v2",
        url="https://psl.noaa.gov/enso/mei/data/meiv2.data",
        parser=_parse_mei,
        cadence="bimonthly",
    ),
    "nao_monthly": IndexSource(
        name="NAO monthly",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
        parser=_parse_cpc_monthly_table("nao"),
        cadence="monthly",
    ),
    "nao": IndexSource(
        name="NAO daily",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.daily.nao.cdas.z500.20200101_current.csv",
        parser=_parse_cpc_daily("nao"),
        cadence="daily",
    ),
    "ao": IndexSource(
        name="AO daily",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        parser=_parse_cpc_monthly_table("ao"),
        cadence="monthly",
    ),
    "pdo": IndexSource(
        name="PDO",
        url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        parser=_parse_pdo,
        cadence="monthly",
    ),
    "mjo": IndexSource(
        name="MJO RMM",
        url="http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
        parser=_parse_mjo,
        cadence="daily",
    ),
    "pna": IndexSource(
        name="PNA monthly",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table",
        parser=_parse_cpc_monthly_table("pna"),
        cadence="monthly",
    ),
    "soi": IndexSource(
        name="SOI",
        url="https://www.cpc.ncep.noaa.gov/data/indices/soi",
        parser=_parse_soi,
        cadence="monthly",
    ),
}


class ClimateIndexDownloader:
    """Downloads and parses CPC/NCEI climate indices.

    Network is best-effort: any URL that 404s or changes format is logged and
    skipped — these endpoints get reorganized every few years. The merged frame
    only includes indices that successfully downloaded.
    """

    def __init__(self, output_dir: Path | str = "data/raw/climate", timeout: float = 30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _fetch(self, url: str) -> Optional[str]:
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning(f"Failed to fetch {url}: {exc}")
            return None

    def download(self, indices: Optional[List[str]] = None, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Download the named indices; default = all."""
        if indices is None:
            indices = list(SOURCES.keys())

        results: Dict[str, pd.DataFrame] = {}
        for key in indices:
            if key not in SOURCES:
                logger.warning(f"Unknown index: {key} — skipping")
                continue
            src = SOURCES[key]
            cache_path = self.output_dir / f"{key}.csv"

            if cache_path.exists() and not force:
                logger.info(f"Using cached {key} -> {cache_path}")
                results[key] = pd.read_csv(cache_path, parse_dates=["date"])
                continue

            logger.info(f"Downloading {src.name} from {src.url}")
            text = self._fetch(src.url)
            if text is None:
                continue

            try:
                df = src.parser(text)
            except Exception as exc:  # parser breakage is recoverable
                logger.warning(f"Failed to parse {key}: {exc}")
                continue

            if df.empty:
                logger.warning(f"Empty parse for {key}")
                continue

            df.to_csv(cache_path, index=False)
            results[key] = df
            logger.success(f"{src.name}: {len(df):,} rows -> {cache_path}")

        return results

    def to_daily_frame(self, dfs: Dict[str, pd.DataFrame], start: str = "1950-01-01") -> pd.DataFrame:
        """Forward-fill all indices onto a daily date index from `start` -> today.

        Monthly indices are held constant within their month. The resulting frame
        is the canonical input the model's ClimateEmbedding consumes.
        """
        dates = pd.date_range(start, pd.Timestamp.today(), freq="D")
        merged = pd.DataFrame({"date": dates}).set_index("date")

        for key, df in dfs.items():
            if df.empty:
                continue
            df = df.set_index("date").sort_index()
            value_cols = [c for c in df.columns]
            # Reindex onto the daily axis with forward-fill (limited to 31 days
            # so a stale monthly value from a year ago doesn't pretend to be current).
            reidx = df.reindex(merged.index, method="ffill", limit=31)
            for col in value_cols:
                merged[col] = reidx[col]

        merged = merged.dropna(how="all").reset_index()
        return merged

    def save_daily_parquet(self, dfs: Dict[str, pd.DataFrame], path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.output_dir / "all_daily.parquet"
        daily = self.to_daily_frame(dfs)
        daily.to_parquet(path, index=False)
        logger.success(f"Combined daily indices -> {path} ({len(daily):,} rows, {len(daily.columns)-1} indices)")
        return path


def main():
    parser = argparse.ArgumentParser(description="Download climate indices for L.I.L.I.T.H.")
    parser.add_argument(
        "--indices",
        type=str,
        default="enso,nao,pdo,mjo,ao",
        help="Comma-separated list. Choices: " + ",".join(SOURCES.keys()),
    )
    parser.add_argument("--output-dir", type=str, default="data/raw/climate")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--start", type=str, default="1950-01-01", help="Start date for daily frame")
    args = parser.parse_args()

    indices = [i.strip() for i in args.indices.split(",") if i.strip()]
    dl = ClimateIndexDownloader(args.output_dir)
    dfs = dl.download(indices, force=args.force)
    if dfs:
        dl.save_daily_parquet(dfs)
    else:
        logger.error("No indices downloaded successfully.")


if __name__ == "__main__":
    main()
