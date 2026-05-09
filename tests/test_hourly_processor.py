"""Tests for the sub-daily -> daily features bridge processor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.processing.hourly_processor import HourlyProcessor, DAILY_FEATURE_MAP


@pytest.fixture
def synthetic_hourly_parquet(tmp_path):
    """Build a small parquet that mimics the output of asos_1min.aggregate_to_hourly."""
    n_days = 100
    hours_per_day = 24
    rows = []
    for station in ("KORD", "KJFK"):
        base_time = pd.Timestamp("2020-01-01 00:00", tz="UTC")
        for d in range(n_days):
            for h in range(hours_per_day):
                ts = base_time + pd.Timedelta(days=d, hours=h)
                # Sine wave temp + per-hour mean/max/min channels.
                temp = 10 + 5 * np.sin(2 * np.pi * h / 24) + d * 0.05
                rows.append({
                    "station": station,
                    "timestamp": ts,
                    "tmpc_mean": temp,
                    "tmpc_max": temp + 0.5,
                    "tmpc_min": temp - 0.5,
                    "relh_min": 50.0 + (h % 3),
                    "sknt_mean": 5.0 + 0.5 * h,
                    "gust_max": 15.0 + 0.2 * h,
                    "mslp_min": 1013.0 - 0.1 * h,
                    "p01i_mean": 0.0 if h % 12 else 1.5,
                    "p01i_max": 0.0 if h % 12 else 2.0,
                })
    df = pd.DataFrame(rows)
    path = tmp_path / "hourly.parquet"
    df.to_parquet(path, index=False)
    return path


class TestHourlyProcessor:
    def test_loads_parquet(self, synthetic_hourly_parquet):
        proc = HourlyProcessor(synthetic_hourly_parquet)
        assert not proc.df.empty
        assert "timestamp" in proc.df.columns

    def test_to_daily_features_shape(self, synthetic_hourly_parquet):
        proc = HourlyProcessor(synthetic_hourly_parquet)
        daily = proc.to_daily_features()
        # 100 days × 2 stations
        assert len(daily) == 200
        for name, *_ in DAILY_FEATURE_MAP:
            assert name in daily.columns, f"missing {name}"

    def test_quartile_of_day_aggregation(self, synthetic_hourly_parquet):
        proc = HourlyProcessor(synthetic_hourly_parquet)
        daily = proc.to_daily_features()
        # tmpc at hour 0 should equal the synthetic temp at h=0 (= base + d*0.05)
        first_day = daily[(daily["station_id"] == "KORD")
                          & (daily["date"] == pd.Timestamp("2020-01-01"))]
        assert not first_day.empty
        # 10 + 5*sin(0) + 0 = 10
        assert first_day["t_00z"].iloc[0] == pytest.approx(10.0, abs=0.1)

    def test_create_training_sequences_contract(self, synthetic_hourly_parquet):
        proc = HourlyProcessor(synthetic_hourly_parquet)
        daily = proc.to_daily_features()
        # Manually fill in lat/lon since the synthetic parquet has no station_meta.
        daily["latitude"] = 40.0
        daily["longitude"] = -74.0
        daily["elevation"] = 10.0

        X, Y, meta, dates = proc.create_training_sequences(
            daily, input_days=10, target_days=10, stride=5
        )
        n_features = len(proc.feature_names)
        assert X.shape[1:] == (10, n_features)
        assert Y.shape[1:] == (10, n_features)
        assert meta.shape[1] == 4
        assert dates.dtype.kind == "M"  # datetime64

    def test_skips_short_stations(self, tmp_path):
        # One station with only 5 days of data — should be dropped.
        rows = []
        for h in range(24 * 5):
            ts = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=h)
            rows.append({
                "station": "KSHORT",
                "timestamp": ts,
                "tmpc_mean": 10.0,
                "tmpc_max": 11.0,
                "tmpc_min": 9.0,
                "relh_min": 50.0,
                "sknt_mean": 5.0,
                "gust_max": 10.0,
                "mslp_min": 1013.0,
                "p01i_mean": 0.0,
                "p01i_max": 0.0,
            })
        path = tmp_path / "short.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)

        proc = HourlyProcessor(path)
        daily = proc.to_daily_features()
        daily["latitude"] = 40.0
        daily["longitude"] = -74.0
        daily["elevation"] = 10.0
        X, Y, meta, dates = proc.create_training_sequences(
            daily, input_days=30, target_days=90, stride=7
        )
        # Not enough days — empty output, not a crash.
        assert len(X) == 0

    def test_feature_names_match_map(self, synthetic_hourly_parquet):
        proc = HourlyProcessor(synthetic_hourly_parquet)
        assert proc.feature_names == [name for name, *_ in DAILY_FEATURE_MAP]
        assert len(proc.feature_names) == 12


class TestAsosAggregatorIntegration:
    """End-to-end pipeline test: raw 1-min CSV -> ASOS aggregator -> HourlyProcessor.

    Catches the kind of bug we shipped in round 1: the aggregator produced a
    3-level MultiIndex on modern pandas, which broke the unpack into ``base``
    and ``stat``. Without this test the failure mode is a runtime error the
    first time anyone actually runs ``lilith download asos-1min``.
    """

    def test_aggregator_produces_columns_processor_consumes(self, tmp_path):
        from data.download.asos_1min import ASOS1MinDownloader

        # Simulate one station-month of NCEI v1 1-min CSVs.
        rows = []
        for h in range(24):
            for m in range(60):
                ts = pd.Timestamp(f"2024-01-01 {h:02d}:{m:02d}:00")
                rows.append({
                    "timestamp": ts,
                    "tmpc": 10.0 + h * 0.1,
                    "dwpc": 5.0,
                    "relh": 50.0,
                    "sknt": 5.0,
                    "gust": 8.0,
                    "mslp": 1013.0,
                    "p01i": 0.0,
                })
        station_dir = tmp_path / "KORD"
        station_dir.mkdir()
        csv = station_dir / "2024-01.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)

        out_pq = tmp_path / "hourly.parquet"
        agg_df = ASOS1MinDownloader(tmp_path).aggregate_to_hourly([csv], output=out_pq)

        # Aggregator must produce the columns HourlyProcessor reads from.
        required = {"tmpc_mean", "tmpc_max", "tmpc_min", "gust_max",
                    "mslp_min", "p01i_mean", "p01i_max", "relh_min",
                    "sknt_mean", "station", "timestamp"}
        missing = required - set(agg_df.columns)
        assert not missing, f"Aggregator missing required columns: {missing}"

        # Pipeline through the processor should yield 12 daily features.
        proc = HourlyProcessor(out_pq)
        daily = proc.to_daily_features()
        for name, *_ in DAILY_FEATURE_MAP:
            assert name in daily.columns, f"daily missing {name}"
        assert len(daily) == 1  # one day of synthetic data
