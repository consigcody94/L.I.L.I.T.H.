#!/usr/bin/env python3
"""
LILITH Data Processing Script

Processes raw GHCN data through the quality control and encoding pipeline.

Usage:
    python scripts/process_data.py --input data/raw/ghcn_daily --output data/storage/parquet
"""

import argparse
from pathlib import Path
from loguru import logger

from data.processing.pipeline import DataPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Process GHCN data through the LILITH pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/ghcn_daily",
        help="Input directory with raw GHCN data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/storage/parquet",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--tensor-output",
        type=str,
        default="data/storage/zarr",
        help="Output directory for tensor data",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Maximum stations to process",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=30,
        help="Minimum years of data required",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"],
        help="Variables to process",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download missing data",
    )
    parser.add_argument(
        "--create-tensors",
        action="store_true",
        help="Create training tensors after processing",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1950,
        help="Start year for tensor creation",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for tensor creation",
    )

    args = parser.parse_args()

    # Setup logging
    logger.info("LILITH Data Processing Pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Create pipeline config
    config = PipelineConfig(
        raw_dir=args.input,
        output_dir=args.output,
        tensor_dir=args.tensor_output,
        min_years=args.min_years,
        target_variables=args.variables,
    )

    # Create and run pipeline
    pipeline = DataPipeline(config)

    logger.info("Running data processing pipeline...")
    pipeline.run(
        max_stations=args.max_stations,
        download=not args.no_download,
    )

    # Create training tensors if requested
    if args.create_tensors:
        logger.info("Creating training tensors...")
        pipeline.create_training_tensors(
            start_year=args.start_year,
            end_year=args.end_year,
        )

    logger.success("Processing complete!")


if __name__ == "__main__":
    main()
