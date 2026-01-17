"""
GHCN-Hourly Data Downloader

Downloads and parses GHCN-Hourly (formerly ISD) data from NOAA NCEI.
https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly

This dataset includes wind, temperature, pressure, humidity, clouds, and more
at hourly resolution from 20,000+ stations worldwide.
"""

import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm


@dataclass
class HourlyStation:
    """GHCN-Hourly station metadata."""

    usaf: str  # USAF station ID
    wban: str  # WBAN station ID
    station_name: str
    country: str
    state: Optional[str]
    latitude: float
    longitude: float
    elevation: float
    begin_date: datetime
    end_date: datetime

    @property
    def id(self) -> str:
        """Combined station ID."""
        return f"{self.usaf}-{self.wban}"


@dataclass
class HourlyObservation:
    """Single hourly observation record."""

    station_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    elevation: float

    # Wind
    wind_direction: Optional[float]  # degrees
    wind_speed: Optional[float]  # m/s
    wind_gust: Optional[float]  # m/s

    # Temperature
    temperature: Optional[float]  # °C
    dew_point: Optional[float]  # °C

    # Pressure
    sea_level_pressure: Optional[float]  # hPa
    station_pressure: Optional[float]  # hPa

    # Humidity
    relative_humidity: Optional[float]  # %

    # Visibility
    visibility: Optional[float]  # meters

    # Precipitation
    precipitation_1h: Optional[float]  # mm
    precipitation_6h: Optional[float]  # mm

    # Sky condition
    cloud_ceiling: Optional[float]  # meters
    cloud_coverage: Optional[str]  # e.g., "CLR", "FEW", "SCT", "BKN", "OVC"

    # Quality
    quality_control: str


class GHCNHourlyDownloader:
    """
    Downloads and parses GHCN-Hourly (ISD-Lite) data.

    GHCN-Hourly provides sub-daily observations including wind, temperature,
    pressure, and humidity from global surface stations.

    We use the ISD-Lite format which is a simplified version containing the
    most essential variables.

    Example usage:
        downloader = GHCNHourlyDownloader(output_dir="data/raw/ghcn_hourly")
        downloader.download_station_list()

        # Download data for specific stations and years
        for station in downloader.get_stations(country="US", min_years=30):
            downloader.download_station_year(station.usaf, station.wban, 2023)
    """

    # ISD-Lite base URL (simplified hourly format)
    BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
    STATION_LIST_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

    def __init__(
        self,
        output_dir: str | Path = "data/raw/ghcn_hourly",
        timeout: float = 60.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def __enter__(self) -> "GHCNHourlyDownloader":
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()

    def download_station_list(self, force: bool = False) -> Path:
        """Download the station history/metadata file."""
        output_path = self.output_dir / "isd-history.csv"

        if output_path.exists() and not force:
            logger.info(f"Station list already exists: {output_path}")
            return output_path

        logger.info(f"Downloading station list from {self.STATION_LIST_URL}")
        response = self.client.get(self.STATION_LIST_URL)
        response.raise_for_status()

        output_path.write_text(response.text)
        logger.success(f"Downloaded station list to {output_path}")
        return output_path

    def parse_stations(self, path: Optional[Path] = None) -> list[HourlyStation]:
        """Parse the station history CSV file."""
        if path is None:
            path = self.output_dir / "isd-history.csv"

        if not path.exists():
            self.download_station_list()

        df = pd.read_csv(path, low_memory=False)

        stations = []
        for _, row in df.iterrows():
            try:
                # Skip stations with missing coordinates
                if pd.isna(row.get("LAT")) or pd.isna(row.get("LON")):
                    continue

                station = HourlyStation(
                    usaf=str(row["USAF"]).zfill(6),
                    wban=str(row["WBAN"]).zfill(5),
                    station_name=str(row.get("STATION NAME", "")),
                    country=str(row.get("CTRY", "")),
                    state=str(row.get("STATE", "")) if pd.notna(row.get("STATE")) else None,
                    latitude=float(row["LAT"]),
                    longitude=float(row["LON"]),
                    elevation=float(row.get("ELEV(M)", 0)) if pd.notna(row.get("ELEV(M)")) else 0.0,
                    begin_date=pd.to_datetime(str(row.get("BEGIN", "19000101")), format="%Y%m%d"),
                    end_date=pd.to_datetime(str(row.get("END", "20991231")), format="%Y%m%d"),
                )
                stations.append(station)
            except Exception as e:
                logger.debug(f"Skipping station: {e}")
                continue

        logger.info(f"Parsed {len(stations)} stations")
        return stations

    def get_stations(
        self,
        country: Optional[str] = None,
        min_years: int = 0,
        bbox: Optional[tuple[float, float, float, float]] = None,
        active_only: bool = True,
    ) -> list[HourlyStation]:
        """
        Get stations matching criteria.

        Args:
            country: 2-letter country code
            min_years: Minimum years of data required
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            active_only: Only include stations with data through 2023+

        Returns:
            List of matching stations
        """
        stations = self.parse_stations()

        if country:
            stations = [s for s in stations if s.country == country]

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            stations = [
                s
                for s in stations
                if min_lon <= s.longitude <= max_lon and min_lat <= s.latitude <= max_lat
            ]

        if min_years > 0:
            stations = [
                s
                for s in stations
                if (s.end_date - s.begin_date).days / 365 >= min_years
            ]

        if active_only:
            cutoff = datetime(2023, 1, 1)
            stations = [s for s in stations if s.end_date >= cutoff]

        logger.info(f"Found {len(stations)} matching stations")
        return stations

    def download_station_year(
        self,
        usaf: str,
        wban: str,
        year: int,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download ISD-Lite data for a station-year.

        ISD-Lite files are organized by year: {year}/{usaf}-{wban}-{year}.gz
        """
        filename = f"{usaf}-{wban}-{year}.gz"
        url = f"{self.BASE_URL}{year}/{filename}"
        output_path = self.output_dir / "data" / str(year) / filename

        if output_path.exists() and not force:
            logger.debug(f"Data already exists: {output_path}")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self.client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return output_path
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"No data for {usaf}-{wban} in {year}")
                return None
            raise

    def parse_isd_lite(
        self,
        usaf: str,
        wban: str,
        year: int,
    ) -> Generator[HourlyObservation, None, None]:
        """
        Parse an ISD-Lite file and yield observations.

        ISD-Lite format (fixed-width, space-separated):
        Field 1: Year
        Field 2: Month
        Field 3: Day
        Field 4: Hour
        Field 5: Air Temperature (°C * 10)
        Field 6: Dew Point Temperature (°C * 10)
        Field 7: Sea Level Pressure (hPa * 10)
        Field 8: Wind Direction (degrees)
        Field 9: Wind Speed (m/s * 10)
        Field 10: Sky Condition Total Coverage Code
        Field 11: Liquid Precipitation Depth 1-Hour (mm * 10)
        Field 12: Liquid Precipitation Depth 6-Hour (mm * 10)

        Missing values are represented as -9999.
        """
        path = self.output_dir / "data" / str(year) / f"{usaf}-{wban}-{year}.gz"

        if not path.exists():
            result = self.download_station_year(usaf, wban, year)
            if result is None:
                return

        station_id = f"{usaf}-{wban}"

        with gzip.open(path, "rt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 12:
                    continue

                try:
                    year_val = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])

                    timestamp = datetime(year_val, month, day, hour)

                    # Parse values (-9999 = missing)
                    def parse_val(idx: int, scale: float = 10.0) -> Optional[float]:
                        val = int(parts[idx])
                        return val / scale if val != -9999 else None

                    yield HourlyObservation(
                        station_id=station_id,
                        timestamp=timestamp,
                        latitude=0.0,  # Need to lookup from station metadata
                        longitude=0.0,
                        elevation=0.0,
                        wind_direction=parse_val(7, 1.0),
                        wind_speed=parse_val(8, 10.0),
                        wind_gust=None,
                        temperature=parse_val(4, 10.0),
                        dew_point=parse_val(5, 10.0),
                        sea_level_pressure=parse_val(6, 10.0),
                        station_pressure=None,
                        relative_humidity=None,  # Computed from temp/dew point
                        visibility=None,
                        precipitation_1h=parse_val(10, 10.0),
                        precipitation_6h=parse_val(11, 10.0),
                        cloud_ceiling=None,
                        cloud_coverage=str(int(parts[9])) if int(parts[9]) != -9999 else None,
                        quality_control="",
                    )
                except (ValueError, IndexError) as e:
                    logger.debug(f"Parse error: {e}")
                    continue

    def station_year_to_dataframe(
        self,
        usaf: str,
        wban: str,
        year: int,
    ) -> pd.DataFrame:
        """Load station-year data as a pandas DataFrame."""
        observations = list(self.parse_isd_lite(usaf, wban, year))

        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame([vars(o) for o in observations])
        df = df.set_index("timestamp").sort_index()

        return df

    def download_station_range(
        self,
        usaf: str,
        wban: str,
        start_year: int,
        end_year: int,
    ) -> list[Path]:
        """Download multiple years of data for a station."""
        paths = []
        for year in range(start_year, end_year + 1):
            result = self.download_station_year(usaf, wban, year)
            if result:
                paths.append(result)
        return paths

    def download_all(
        self,
        stations: Optional[list[HourlyStation]] = None,
        years: Optional[list[int]] = None,
        max_stations: Optional[int] = None,
        **filter_kwargs,
    ) -> int:
        """
        Download data for multiple stations and years.

        Returns count of files downloaded.
        """
        if stations is None:
            stations = self.get_stations(**filter_kwargs)

        if max_stations:
            stations = stations[:max_stations]

        if years is None:
            years = list(range(2000, 2024))

        count = 0
        for station in tqdm(stations, desc="Downloading stations"):
            for year in years:
                try:
                    result = self.download_station_year(station.usaf, station.wban, year)
                    if result:
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to download {station.id}/{year}: {e}")

        logger.success(f"Downloaded {count} station-year files")
        return count


def main():
    """CLI entry point for downloading GHCN-Hourly data."""
    import argparse

    parser = argparse.ArgumentParser(description="Download GHCN-Hourly (ISD-Lite) data")
    parser.add_argument(
        "--output-dir",
        default="data/raw/ghcn_hourly",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Filter by country code (e.g., US, CA, GB)",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=20,
        help="Minimum years of data required",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Maximum number of stations to download",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Start year for data download",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for data download",
    )

    args = parser.parse_args()

    with GHCNHourlyDownloader(output_dir=args.output_dir) as downloader:
        downloader.download_station_list()

        years = list(range(args.start_year, args.end_year + 1))
        downloader.download_all(
            country=args.country,
            min_years=args.min_years,
            max_stations=args.max_stations,
            years=years,
        )


if __name__ == "__main__":
    main()
