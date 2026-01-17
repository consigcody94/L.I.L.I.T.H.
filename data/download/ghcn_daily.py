"""
GHCN-Daily Data Downloader

Downloads and parses GHCN-Daily data from NOAA NCEI.
https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

Data format documentation:
https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
"""

import gzip
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm


@dataclass
class Station:
    """GHCN Station metadata."""

    id: str
    latitude: float
    longitude: float
    elevation: float
    state: Optional[str]
    name: str
    gsn_flag: Optional[str]
    hcn_flag: Optional[str]
    wmo_id: Optional[str]


@dataclass
class DailyObservation:
    """Single daily observation record."""

    station_id: str
    date: datetime
    element: str  # TMAX, TMIN, PRCP, SNOW, SNWD, etc.
    value: float
    m_flag: Optional[str]  # Measurement flag
    q_flag: Optional[str]  # Quality flag
    s_flag: Optional[str]  # Source flag


class GHCNDailyDownloader:
    """
    Downloads and parses GHCN-Daily data.

    GHCN-Daily contains daily climate summaries from land surface stations
    across the globe, with records from over 100,000 stations in 180 countries.

    Example usage:
        downloader = GHCNDailyDownloader(output_dir="data/raw/ghcn_daily")
        downloader.download_stations()
        downloader.download_inventory()

        # Download data for specific stations
        for station in downloader.get_stations(country="US", min_years=50):
            downloader.download_station_data(station.id)
    """

    BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"

    # Element codes we care about
    ELEMENTS = {
        "TMAX": "Maximum temperature (tenths of degrees C)",
        "TMIN": "Minimum temperature (tenths of degrees C)",
        "PRCP": "Precipitation (tenths of mm)",
        "SNOW": "Snowfall (mm)",
        "SNWD": "Snow depth (mm)",
        "AWND": "Average daily wind speed (tenths of m/s)",
        "TAVG": "Average temperature (tenths of degrees C)",
        "RHAV": "Average relative humidity (%)",
        "RHMX": "Maximum relative humidity (%)",
        "RHMN": "Minimum relative humidity (%)",
    }

    def __init__(
        self,
        output_dir: str | Path = "data/raw/ghcn_daily",
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

    def __enter__(self) -> "GHCNDailyDownloader":
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()

    def download_stations(self, force: bool = False) -> Path:
        """
        Download station metadata file (ghcnd-stations.txt).

        Returns path to the downloaded file.
        """
        url = f"{self.BASE_URL}ghcnd-stations.txt"
        output_path = self.output_dir / "ghcnd-stations.txt"

        if output_path.exists() and not force:
            logger.info(f"Stations file already exists: {output_path}")
            return output_path

        logger.info(f"Downloading stations from {url}")
        response = self.client.get(url)
        response.raise_for_status()

        output_path.write_text(response.text)
        logger.success(f"Downloaded stations to {output_path}")
        return output_path

    def download_inventory(self, force: bool = False) -> Path:
        """
        Download station inventory file (ghcnd-inventory.txt).

        The inventory shows which elements are available for each station
        and the period of record.
        """
        url = f"{self.BASE_URL}ghcnd-inventory.txt"
        output_path = self.output_dir / "ghcnd-inventory.txt"

        if output_path.exists() and not force:
            logger.info(f"Inventory file already exists: {output_path}")
            return output_path

        logger.info(f"Downloading inventory from {url}")
        response = self.client.get(url)
        response.raise_for_status()

        output_path.write_text(response.text)
        logger.success(f"Downloaded inventory to {output_path}")
        return output_path

    def parse_stations(self, path: Optional[Path] = None) -> list[Station]:
        """
        Parse the stations metadata file.

        Format (fixed-width):
        ID            1-11   Character
        LATITUDE     13-20   Real
        LONGITUDE    22-30   Real
        ELEVATION    32-37   Real
        STATE        39-40   Character
        NAME         42-71   Character
        GSN FLAG     73-75   Character
        HCN/CRN FLAG 77-79   Character
        WMO ID       81-85   Character
        """
        if path is None:
            path = self.output_dir / "ghcnd-stations.txt"

        if not path.exists():
            self.download_stations()

        stations = []
        with open(path) as f:
            for line in f:
                if len(line.strip()) < 40:
                    continue

                station = Station(
                    id=line[0:11].strip(),
                    latitude=float(line[12:20].strip()),
                    longitude=float(line[21:30].strip()),
                    elevation=float(line[31:37].strip()) if line[31:37].strip() else 0.0,
                    state=line[38:40].strip() or None,
                    name=line[41:71].strip(),
                    gsn_flag=line[72:75].strip() or None,
                    hcn_flag=line[76:79].strip() or None,
                    wmo_id=line[80:85].strip() or None,
                )
                stations.append(station)

        logger.info(f"Parsed {len(stations)} stations")
        return stations

    def parse_inventory(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Parse the inventory file.

        Format (fixed-width):
        ID            1-11   Character
        LATITUDE     13-20   Real
        LONGITUDE    22-30   Real
        ELEMENT      32-35   Character
        FIRSTYEAR    37-40   Integer
        LASTYEAR     42-45   Integer
        """
        if path is None:
            path = self.output_dir / "ghcnd-inventory.txt"

        if not path.exists():
            self.download_inventory()

        records = []
        with open(path) as f:
            for line in f:
                if len(line.strip()) < 45:
                    continue

                records.append(
                    {
                        "station_id": line[0:11].strip(),
                        "latitude": float(line[12:20].strip()),
                        "longitude": float(line[21:30].strip()),
                        "element": line[31:35].strip(),
                        "first_year": int(line[36:40].strip()),
                        "last_year": int(line[41:45].strip()),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} inventory records")
        return df

    def get_stations(
        self,
        country: Optional[str] = None,
        min_years: int = 0,
        elements: Optional[list[str]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> list[Station]:
        """
        Get stations matching criteria.

        Args:
            country: 2-letter country code (first 2 chars of station ID)
            min_years: Minimum years of data required
            elements: Required elements (e.g., ["TMAX", "TMIN", "PRCP"])
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            List of matching stations
        """
        stations = self.parse_stations()
        inventory = self.parse_inventory()

        # Filter by country
        if country:
            stations = [s for s in stations if s.id.startswith(country)]

        # Filter by bounding box
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            stations = [
                s
                for s in stations
                if min_lon <= s.longitude <= max_lon and min_lat <= s.latitude <= max_lat
            ]

        # Filter by data availability
        if min_years > 0 or elements:
            valid_ids = set()
            elements = elements or list(self.ELEMENTS.keys())

            for station in stations:
                station_inv = inventory[inventory["station_id"] == station.id]

                # Check each required element
                has_all_elements = True
                max_years = 0

                for element in elements:
                    elem_inv = station_inv[station_inv["element"] == element]
                    if elem_inv.empty:
                        has_all_elements = False
                        break

                    years = elem_inv.iloc[0]["last_year"] - elem_inv.iloc[0]["first_year"]
                    max_years = max(max_years, years)

                if has_all_elements and max_years >= min_years:
                    valid_ids.add(station.id)

            stations = [s for s in stations if s.id in valid_ids]

        logger.info(f"Found {len(stations)} matching stations")
        return stations

    def download_station_data(
        self,
        station_id: str,
        force: bool = False,
    ) -> Path:
        """
        Download data file for a single station.

        The data is stored in .dly format (one file per station).
        """
        # Station data is in the 'all' subdirectory as .dly.gz files
        url = f"{self.BASE_URL}all/{station_id}.dly"
        output_path = self.output_dir / "stations" / f"{station_id}.dly"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not force:
            logger.debug(f"Station data already exists: {output_path}")
            return output_path

        logger.debug(f"Downloading {station_id}")

        try:
            response = self.client.get(url)
            response.raise_for_status()
            output_path.write_text(response.text)
        except httpx.HTTPStatusError:
            # Try gzipped version
            url_gz = f"{url}.gz"
            response = self.client.get(url_gz)
            response.raise_for_status()

            # Decompress
            content = gzip.decompress(response.content)
            output_path.write_bytes(content)

        return output_path

    def parse_station_data(self, station_id: str) -> Generator[DailyObservation, None, None]:
        """
        Parse a station's .dly file and yield observations.

        Format (fixed-width, one line per station-year-month-element):
        ID            1-11   Character
        YEAR         12-15   Integer
        MONTH        16-17   Integer
        ELEMENT      18-21   Character
        VALUE1       22-26   Integer (day 1)
        MFLAG1       27-27   Character
        QFLAG1       28-28   Character
        SFLAG1       29-29   Character
        ... repeated for days 2-31
        """
        path = self.output_dir / "stations" / f"{station_id}.dly"
        if not path.exists():
            self.download_station_data(station_id)

        with open(path) as f:
            for line in f:
                if len(line) < 269:
                    continue

                station = line[0:11].strip()
                year = int(line[11:15])
                month = int(line[15:17])
                element = line[17:21].strip()

                # Skip elements we don't care about
                if element not in self.ELEMENTS:
                    continue

                # Parse each day's value (31 days max)
                for day in range(1, 32):
                    offset = 21 + (day - 1) * 8
                    value_str = line[offset : offset + 5].strip()
                    m_flag = line[offset + 5 : offset + 6].strip() or None
                    q_flag = line[offset + 6 : offset + 7].strip() or None
                    s_flag = line[offset + 7 : offset + 8].strip() or None

                    # -9999 indicates missing value
                    if value_str == "-9999" or not value_str:
                        continue

                    try:
                        date = datetime(year, month, day)
                    except ValueError:
                        # Invalid date (e.g., Feb 30)
                        continue

                    # Convert value (stored as tenths for most elements)
                    value = float(value_str)
                    if element in ("TMAX", "TMIN", "TAVG", "PRCP", "AWND"):
                        value /= 10.0

                    yield DailyObservation(
                        station_id=station,
                        date=date,
                        element=element,
                        value=value,
                        m_flag=m_flag,
                        q_flag=q_flag,
                        s_flag=s_flag,
                    )

    def station_to_dataframe(self, station_id: str) -> pd.DataFrame:
        """
        Load station data as a pandas DataFrame.

        Returns a DataFrame with columns for each element and a datetime index.
        """
        observations = list(self.parse_station_data(station_id))

        if not observations:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([vars(o) for o in observations])

        # Pivot to have elements as columns
        df = df.pivot_table(
            index="date",
            columns="element",
            values="value",
            aggfunc="first",
        )

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def download_all(
        self,
        stations: Optional[list[Station]] = None,
        max_stations: Optional[int] = None,
        **filter_kwargs,
    ) -> list[Path]:
        """
        Download data for multiple stations.

        Args:
            stations: List of stations to download (or use filter_kwargs)
            max_stations: Maximum number of stations to download
            **filter_kwargs: Arguments passed to get_stations()

        Returns:
            List of paths to downloaded files
        """
        if stations is None:
            stations = self.get_stations(**filter_kwargs)

        if max_stations:
            stations = stations[:max_stations]

        paths = []
        for station in tqdm(stations, desc="Downloading stations"):
            try:
                path = self.download_station_data(station.id)
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to download {station.id}: {e}")

        logger.success(f"Downloaded {len(paths)} station files")
        return paths


def main():
    """CLI entry point for downloading GHCN-Daily data."""
    import argparse

    parser = argparse.ArgumentParser(description="Download GHCN-Daily data")
    parser.add_argument(
        "--output-dir",
        default="data/raw/ghcn_daily",
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
        default=30,
        help="Minimum years of data required",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Maximum number of stations to download",
    )
    parser.add_argument(
        "--stations-only",
        action="store_true",
        help="Only download station metadata, not observation data",
    )

    args = parser.parse_args()

    with GHCNDailyDownloader(output_dir=args.output_dir) as downloader:
        # Always download metadata
        downloader.download_stations()
        downloader.download_inventory()

        if not args.stations_only:
            downloader.download_all(
                country=args.country,
                min_years=args.min_years,
                max_stations=args.max_stations,
            )


if __name__ == "__main__":
    main()
