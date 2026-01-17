#!/usr/bin/env python3
"""
LILITH API Server Startup Script

Starts the FastAPI server for the LILITH weather prediction API.

Usage:
    python scripts/start_api.py --checkpoint checkpoints/best.pt --port 8000
"""

import argparse
import os
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Start LILITH API server"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, runs in demo mode without)",
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
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Set environment variables for the API
    if args.checkpoint:
        os.environ["LILITH_CHECKPOINT"] = args.checkpoint
    if args.encoder:
        os.environ["LILITH_ENCODER"] = args.encoder
    if args.stations:
        os.environ["LILITH_STATIONS"] = args.stations

    logger.info("Starting LILITH API Server")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Checkpoint: {args.checkpoint or 'None (demo mode)'}")

    # Import and run uvicorn
    import uvicorn

    uvicorn.run(
        "web.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
