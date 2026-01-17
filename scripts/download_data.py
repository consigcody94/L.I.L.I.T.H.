#!/usr/bin/env python3
"""
LILITH Data Download Script

Downloads GHCN weather station data from NOAA NCEI.

Usage:
    python scripts/download_data.py --max-stations 1000 --min-years 30
"""

import argparse
from pathlib import Path
from loguru import logger

from data.download.ghcn_daily import GHCNDailyDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download GHCN weather station data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/ghcn_daily",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Maximum number of stations to download",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=30,
        help="Minimum years of data required per station",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=["TMAX", "TMIN", "PRCP"],
        help="Weather elements to require",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Filter by country code (e.g., US, UK, JP)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, just list matching stations",
    )

    args = parser.parse_args()

    # Setup logging
    logger.info("LILITH Data Download Script")
    logger.info(f"Output directory: {args.output_dir}")

    # Create downloader
    downloader = GHCNDailyDownloader(output_dir=args.output_dir)

    # Download station metadata
    logger.info("Downloading station metadata...")
    downloader.download_stations()
    downloader.download_inventory()

    # Get qualifying stations
    logger.info(f"Finding stations with {args.min_years}+ years of data...")
    stations = downloader.get_stations(
        min_years=args.min_years,
        elements=args.elements,
    )

    # Filter by country if specified
    if args.country:
        stations = [s for s in stations if s.id.startswith(args.country)]
        logger.info(f"Filtered to {len(stations)} stations in {args.country}")

    # Limit number of stations
    if args.max_stations:
        stations = stations[:args.max_stations]

    logger.info(f"Found {len(stations)} qualifying stations")

    if args.skip_download:
        # Just list stations
        for i, station in enumerate(stations[:20]):
            logger.info(f"  {station.id}: {station.name} ({station.latitude:.2f}, {station.longitude:.2f})")
        if len(stations) > 20:
            logger.info(f"  ... and {len(stations) - 20} more")
        return

    # Download station data
    logger.info("Downloading station data...")
    success_count = 0
    fail_count = 0

    for i, station in enumerate(stations):
        try:
            downloader.download_station_data(station.id)
            success_count += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(stations)} stations downloaded")

        except Exception as e:
            logger.warning(f"Failed to download {station.id}: {e}")
            fail_count += 1

    logger.success(f"Download complete: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    main()
