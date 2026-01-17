#!/usr/bin/env python3
"""
LILITH Inference Script

Run inference with a trained LILITH model.

Usage:
    python scripts/run_inference.py --checkpoint checkpoints/best.pt --lat 40.7128 --lon -74.006
"""

import argparse
import json
from pathlib import Path
from loguru import logger

import torch

from inference.forecast import Forecaster, ForecastRequest


def main():
    parser = argparse.ArgumentParser(
        description="Run LILITH weather forecast inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude of forecast location",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude of forecast location",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to forecast",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Path to feature encoder JSON",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default=None,
        help="Path to stations parquet file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=10,
        help="Number of ensemble members for uncertainty",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "table"],
        default="table",
        help="Output format",
    )

    args = parser.parse_args()

    # Load forecaster
    logger.info(f"Loading model from {args.checkpoint}")
    forecaster = Forecaster.from_pretrained(
        checkpoint_path=args.checkpoint,
        device=args.device,
        encoder_path=args.encoder,
        stations_path=args.stations,
    )

    # Run forecast
    logger.info(f"Generating {args.days}-day forecast for ({args.lat}, {args.lon})")
    response = forecaster.forecast(
        latitude=args.lat,
        longitude=args.lon,
        forecast_days=args.days,
        include_uncertainty=True,
        ensemble_members=args.ensemble,
    )

    # Output results
    if args.format == "json":
        output = json.dumps(response.to_dict(), indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            logger.info(f"Forecast saved to {args.output}")
        else:
            print(output)

    else:  # table format
        print(f"\n{'='*70}")
        print(f"LILITH 90-Day Forecast")
        print(f"Location: {args.lat:.4f}°, {args.lon:.4f}°")
        print(f"Generated: {response.generated_at}")
        print(f"Model: {response.model_version}")
        print(f"{'='*70}\n")

        print(f"{'Date':<12} {'High':>8} {'Low':>8} {'Precip':>8} {'Prob':>6}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

        for forecast in response.forecasts[:14]:  # Show first 14 days
            print(
                f"{forecast.date:<12} "
                f"{forecast.temperature_max:>7.1f}° "
                f"{forecast.temperature_min:>7.1f}° "
                f"{forecast.precipitation:>7.1f}mm "
                f"{forecast.precipitation_probability*100:>5.0f}%"
            )

        if len(response.forecasts) > 14:
            print(f"\n... and {len(response.forecasts) - 14} more days")

        # Summary statistics
        temps_max = [f.temperature_max for f in response.forecasts]
        temps_min = [f.temperature_min for f in response.forecasts]
        precips = [f.precipitation for f in response.forecasts]

        print(f"\n{'Summary Statistics':}")
        print(f"  Temperature range: {min(temps_min):.1f}° to {max(temps_max):.1f}°")
        print(f"  Total precipitation: {sum(precips):.1f}mm")
        print(f"  Rainy days: {sum(1 for p in precips if p > 0.1)}")


if __name__ == "__main__":
    main()
