"""Operator CLI: python -m scripts.verify_forecasts --ledger PATH --limit 100."""

import argparse
from pathlib import Path
import json

from data.providers.open_meteo import OpenMeteoProvider
from web.api.storage.ledger import ForecastLedger
from web.api.verification import compare_pending


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error("limit must be between 1 and 1000")
    ledger = ForecastLedger(args.ledger)
    count = compare_pending(ledger, OpenMeteoProvider(timeout=10), limit=args.limit)
    report = ledger.report()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"new_reference_comparisons": count, "independent_station_verified": 0}))
