"""Bounded, idempotent modeled-reference comparisons (not independent truth)."""

from datetime import date, datetime, timezone
import time


def compare_pending(ledger, provider, *, today: date | None = None,
                    limit: int = 100, max_seconds: float = 45) -> int:
    today = today or datetime.now(timezone.utc).date()
    started, count, fetched = time.monotonic(), 0, {}
    for record in ledger.pending(today, limit=limit):
        if time.monotonic() - started > max_seconds:
            break
        target = date.fromisoformat(record["target_date"])
        key = (record["latitude"], record["longitude"], target)
        if key not in fetched:
            fetched[key] = provider.daily_observations(*key)
        values = fetched[key]
        if values is None:
            continue
        from data.providers.open_meteo import ARCHIVE_LAG_DAYS
        recent = (today - target).days <= ARCHIVE_LAG_DAYS
        source = "open_meteo_forecast_past_days" if recent else "open_meteo_archive"
        basis = "past_forecast" if recent else "gridded_analysis"
        count += ledger.compare(record["id"], values, source=source, basis=basis,
                                retrieved_at=datetime.now(timezone.utc))
    return count
