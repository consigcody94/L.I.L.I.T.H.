"""SQLite pilot ledger. Single host only; global production uses object storage.

Forecasts are immutable. Corrected comparison data produce new observation and
verification versions. Model estimates are explicitly distinguished from stations.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone, timedelta, date
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from web.api.accuracy import aggregate_errors


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    # Existing forecasters emit naive server timestamps. Preserve the assumption in
    # the payload; deployments must set TZ=UTC until provider timezones are explicit.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class ForecastLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO schema_version VALUES (1);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, issued_at TEXT NOT NULL,
                    latitude REAL NOT NULL, longitude REAL NOT NULL,
                    model_version TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_time ON runs(issued_at);
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    target_date TEXT NOT NULL, lead_days INTEGER NOT NULL,
                    tmax REAL NOT NULL, tmin REAL NOT NULL, precip REAL NOT NULL,
                    precip_prob REAL NOT NULL, source TEXT NOT NULL,
                    UNIQUE(run_id, target_date)
                );
                CREATE INDEX IF NOT EXISTS predictions_target ON predictions(target_date);
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY, latitude REAL NOT NULL, longitude REAL NOT NULL,
                    target_date TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    source TEXT NOT NULL, basis TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verifications (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL REFERENCES predictions(id),
                    observation_id TEXT NOT NULL REFERENCES observations(id),
                    algorithm TEXT NOT NULL, verified_at TEXT NOT NULL,
                    UNIQUE(prediction_id, observation_id, algorithm)
                );
                CREATE INDEX IF NOT EXISTS verification_prediction ON verifications(prediction_id, sequence);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def archive(self, payload: dict) -> str:
        """Transactionally store every returned daily forecast, before serving it."""
        body = dict(payload)
        body.pop("run_id", None)
        run_id = hashlib.sha256(canonical(body).encode()).hexdigest()
        location = body["location"]
        issued = utc(body["generated_at"]).isoformat()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                       (run_id, issued, location["latitude"], location["longitude"],
                        body["model_version"], canonical({**body, "run_id": run_id})))
            for forecast in body["forecasts"]:
                target = date.fromisoformat(forecast["date"])
                lead = (target - utc(body["generated_at"]).date()).days
                db.execute("INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (f"{run_id}:{target.isoformat()}", run_id, target.isoformat(), lead,
                            forecast["temperature_max"], forecast["temperature_min"],
                            forecast["precipitation"], forecast["precipitation_probability"],
                            forecast.get("source") or "unspecified"))
        return run_id

    def replay(self, run_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def archive_hourly(self, payload: dict) -> str:
        """Retain all hourly fields exactly; hourly scoring is a separate contract."""
        body = dict(payload)
        body.pop("run_id", None)
        run_id = hashlib.sha256(canonical(body).encode()).hexdigest()
        location = body["location"]
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                       (run_id, utc(body["generated_at"]).isoformat(), location["latitude"],
                        location["longitude"], body["model_version"], canonical({**body, "run_id": run_id})))
        return run_id

    def pending(self, today: date, limit: int = 200) -> list[dict]:
        """Fetch a bounded group; no remote I/O is performed under a DB lock."""
        with self.connect() as db:
            rows = db.execute("""
                SELECT p.*, r.latitude, r.longitude FROM predictions p
                JOIN runs r ON r.id=p.run_id
                WHERE p.target_date < ? AND NOT EXISTS (
                    SELECT 1 FROM verifications v JOIN observations o ON o.id=v.observation_id
                    WHERE v.sequence=(SELECT MAX(v2.sequence) FROM verifications v2 WHERE v2.prediction_id=p.id)
                    AND json_extract(o.payload, '$.values.tmax') IS NOT NULL
                    AND json_extract(o.payload, '$.values.tmin') IS NOT NULL
                    AND json_extract(o.payload, '$.values.prcp') IS NOT NULL
                ) ORDER BY p.target_date DESC, p.id LIMIT ?
            """, (today.isoformat(), limit)).fetchall()
        return [dict(row) for row in rows]

    def compare(self, prediction_id: str, values: dict, *, source: str, basis: str,
                retrieved_at: datetime | None = None) -> bool:
        """Append a comparison, preserving original truth revisions and partial data.

        Station matching is deliberately NOT inferred from coordinates. This pilot
        supports modeled reference comparisons; station verification needs a separate
        station-id, elevation, distance and day-window matching contract.
        """
        if basis not in {"gridded_analysis", "past_forecast"}:
            raise ValueError("Station observation matching is not implemented in this pilot")
        for field in ("tmax", "tmin", "prcp"):
            value = values.get(field)
            if value is not None and not math.isfinite(value):
                raise ValueError("Comparison values must be finite or null")
        now = retrieved_at or datetime.now(timezone.utc)
        now = utc(now)
        with self.connect() as db:
            row = db.execute("""SELECT p.*, r.latitude, r.longitude FROM predictions p
                                JOIN runs r ON r.id=p.run_id WHERE p.id=?""", (prediction_id,)).fetchone()
            if row is None:
                raise KeyError(prediction_id)
            if date.fromisoformat(row["target_date"]) >= now.date():
                return False
            content = {"values": {k: values.get(k) for k in ("tmax", "tmin", "prcp")},
                       "latitude": row["latitude"], "longitude": row["longitude"],
                       "target_date": row["target_date"], "source": source, "basis": basis,
                       "day_definition": "provider_local_day_unresolved"}
            previous = db.execute("""SELECT o.id, o.payload FROM verifications v
                                    JOIN observations o ON o.id=v.observation_id
                                    WHERE v.prediction_id=? ORDER BY v.sequence DESC LIMIT 1""",
                                  (prediction_id,)).fetchone()
            if previous:
                old = json.loads(previous["payload"])
                old.pop("revision_parent", None)
                if canonical(old) == canonical(content):
                    return False
            # The same numeric value can legitimately reappear after a correction.
            # A predecessor link distinguishes that revision from an idempotent retry.
            content["revision_parent"] = previous["id"] if previous else None
            observation_id = hashlib.sha256(canonical(content).encode()).hexdigest()
            db.execute("INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (observation_id, row["latitude"], row["longitude"], row["target_date"],
                        now.isoformat(), source, basis, canonical(content)))
            cursor = db.execute("INSERT OR IGNORE INTO verifications "
                                "(prediction_id, observation_id, algorithm, verified_at) VALUES (?, ?, ?, ?)",
                                (prediction_id, observation_id, "daily_difference_v1", now.isoformat()))
            return cursor.rowcount == 1

    def records(self, *, lat=None, lon=None, days: int | None = None,
                now: datetime | None = None, limit: int | None = None,
                verified_only: bool = False) -> list[dict]:
        now = utc(now or datetime.now(timezone.utc))
        clauses, parameters = [], []
        if (lat is None) != (lon is None):
            raise ValueError("Supply both latitude and longitude")
        if lat is not None:
            # An exact persisted forecast point, not a broad half-degree box.
            clauses.extend(["abs(r.latitude-?) < 0.00005", "abs(r.longitude-?) < 0.00005"])
            parameters.extend([lat, lon])
        if days is not None:
            clauses.extend(["r.issued_at >= ?", "r.issued_at <= ?"])
            parameters.extend([(now - timedelta(days=days)).isoformat(), now.isoformat()])
        if verified_only:
            clauses.append("o.id IS NOT NULL")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"""
            SELECT p.*, r.issued_at, r.latitude, r.longitude, r.model_version,
                   o.payload AS observation, o.source AS verification_source,
                   o.basis AS verification_basis, v.verified_at
            FROM predictions p JOIN runs r ON r.id=p.run_id
            LEFT JOIN verifications v ON v.sequence=(
                SELECT MAX(v2.sequence) FROM verifications v2 WHERE v2.prediction_id=p.id
            ) LEFT JOIN observations o ON o.id=v.observation_id
            {where} ORDER BY r.issued_at DESC, p.target_date, p.id
        """
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self.connect() as db:
            rows = db.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            actual = json.loads(row["observation"])["values"] if row["observation"] else {}
            maximum, minimum, precip = actual.get("tmax"), actual.get("tmin"), actual.get("prcp")
            record = {
                "id": row["id"], "run_id": row["run_id"],
                "location": {"latitude": row["latitude"], "longitude": row["longitude"]},
                "predicted_at": row["issued_at"], "target_date": row["target_date"],
                "predicted_temp_max": row["tmax"], "predicted_temp_min": row["tmin"],
                "predicted_precipitation": row["precip"], "predicted_precip_prob": row["precip_prob"],
                "lead_days": row["lead_days"], "source": row["source"],
                "model_version": row["model_version"], "actual_temp_max": maximum,
                "actual_temp_min": minimum, "actual_precipitation": precip,
                "temp_max_error": row["tmax"] - maximum if maximum is not None else None,
                "temp_min_error": row["tmin"] - minimum if minimum is not None else None,
                "precip_error": row["precip"] - precip if precip is not None else None,
                "verification_source": row["verification_source"],
                "verification_basis": row["verification_basis"], "verified_at": row["verified_at"],
            }
            result.append(record)
        return result

    def report(self, *, lat=None, lon=None, days=30, now=None) -> dict:
        now = utc(now or datetime.now(timezone.utc))
        records = self.records(lat=lat, lon=lon, days=days, now=now)
        complete = [r for r in records if all(r[k] is not None for k in
                    ("actual_temp_max", "actual_temp_min", "actual_precipitation"))]
        # Keep existing v1 numeric response compatibility. The UI gates on count;
        # zero counts mean unavailable, not perfect accuracy.
        stats = aggregate_errors(complete)
        stats["total_predictions"] = len(records)
        return {"generated_at": now, "period_start": (now - timedelta(days=days)).date().isoformat(),
                "period_end": now.date().isoformat(), "stats": stats,
                "recent_predictions": records[:50],
                "location_filter": f"{lat:.4f},{lon:.4f}" if lat is not None else None,
                "comparison_basis": "modeled_reference_not_station_observations",
                "period_basis": "forecast_issue_time", "independent_station_verified": 0}
