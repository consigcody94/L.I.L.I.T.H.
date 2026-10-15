from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone

import pytest

from web.api.storage.ledger import ForecastLedger


def issuance(issued="2026-01-01T12:00:00+00:00", days=90, offset=0):
    start = datetime.fromisoformat(issued).date() + timedelta(days=1)
    return {"location": {"latitude": 40.7, "longitude": -74.0}, "generated_at": issued,
            "model_version": "research-test", "forecast_days": days,
            "forecasts": [{"date": (start + timedelta(days=i)).isoformat(),
                           "temperature_max": 10.0 + offset, "temperature_min": 2.0,
                           "precipitation": 1.0, "precipitation_probability": 0.2,
                           "source": "reference+model"} for i in range(days)]}


def test_full_horizon_survives_restart_and_exact_replay(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    payload = issuance()
    run_id = ForecastLedger(path).archive(payload)
    restarted = ForecastLedger(path)
    assert restarted.replay(run_id) == {**payload, "run_id": run_id}
    assert len(restarted.records()) == 90
    assert restarted.records()[-1]["lead_days"] == 90


def test_retry_is_idempotent_and_new_issue_does_not_overwrite(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    first = ledger.archive(issuance(days=1))
    assert ledger.archive(issuance(days=1)) == first
    second = ledger.archive(issuance(issued="2026-01-01T13:00:00+00:00", days=1, offset=4))
    assert first != second
    assert len(ledger.records()) == 2
    assert ledger.replay(first)["forecasts"][0]["temperature_max"] == 10


def test_concurrent_duplicate_writes_are_atomic(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: ledger.archive(issuance()), range(8)))
    assert len(set(ids)) == 1
    assert len(ledger.records()) == 90


def test_comparison_revisions_preserve_predictions_and_old_truth(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    run_id = ledger.archive(issuance(days=1))
    record_id = ledger.records()[0]["id"]
    kwargs = dict(source="archive", basis="gridded_analysis", retrieved_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert ledger.compare(record_id, {"tmax": 8, "tmin": 1, "prcp": 1}, **kwargs)
    assert not ledger.compare(record_id, {"tmax": 8, "tmin": 1, "prcp": 1}, **kwargs)
    assert ledger.compare(record_id, {"tmax": 9, "tmin": 1, "prcp": 1}, **kwargs)
    assert ledger.records()[0]["temp_max_error"] == 1
    assert ledger.replay(run_id)["forecasts"][0]["temperature_max"] == 10
    with ledger.connect() as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM verifications").fetchone()[0] == 2
    assert ledger.compare(record_id, {"tmax": 8, "tmin": 1, "prcp": 1}, **kwargs)
    assert ledger.records()[0]["temp_max_error"] == 2
    with ledger.connect() as db:
        assert db.execute("SELECT count(*) FROM verifications").fetchone()[0] == 3


def test_future_and_partial_truth_never_become_perfect_scores(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    ledger.archive(issuance(days=1))
    identifier = ledger.records()[0]["id"]
    assert not ledger.compare(identifier, {"tmax": 8}, source="archive", basis="gridded_analysis",
                              retrieved_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    ledger.compare(identifier, {"tmax": 8}, source="archive", basis="gridded_analysis",
                   retrieved_at=datetime(2026, 1, 3, tzinfo=timezone.utc))
    record = ledger.records()[0]
    assert record["actual_precipitation"] is None
    report = ledger.report(now=datetime(2026, 1, 5, tzinfo=timezone.utc))
    assert report["stats"]["verified_predictions"] == 0
    assert report["independent_station_verified"] == 0
    assert len(ledger.pending(date(2026, 1, 5))) == 1


def test_report_applies_time_and_exact_location_filter(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    ledger.archive(issuance(issued="2025-01-01T12:00:00+00:00", days=1))
    ledger.archive(issuance(days=1))
    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    assert ledger.report(days=30, now=now)["stats"]["total_predictions"] == 1
    assert ledger.report(lat=40.8, lon=-74, now=now)["stats"]["total_predictions"] == 0
    with pytest.raises(ValueError):
        ledger.report(lat=40.7)


def test_no_station_truth_without_matching_contract(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    ledger.archive(issuance(days=1))
    with pytest.raises(ValueError, match="Station"):
        ledger.compare(ledger.records()[0]["id"], {"tmax": 8}, source="ghcn", basis="station")


def test_pending_batch_is_bounded_and_skips_future(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite3")
    ledger.archive(issuance())
    assert len(ledger.pending(date(2026, 2, 1), limit=5)) == 5
    assert ledger.pending(date(2026, 1, 1)) == []


def test_hourly_payload_survives_restart_without_daily_scoring(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = ForecastLedger(path)
    payload = {"location": {"latitude": 1, "longitude": 2},
               "generated_at": "2026-01-01T12:00:00Z", "model_version": "reference",
               "forecast_hours": 1, "forecasts": [{"datetime": "2026-01-01T13:00:00Z",
                                                     "wind_speed": 8.7, "humidity": 76}]}
    identifier = ledger.archive_hourly(payload)
    assert ForecastLedger(path).replay(identifier) == {**payload, "run_id": identifier}
    assert ledger.records() == []
