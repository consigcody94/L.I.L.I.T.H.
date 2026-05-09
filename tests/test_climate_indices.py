"""Unit tests for the climate indices downloader and parsers."""
from __future__ import annotations

import pandas as pd
import pytest

from data.download.climate_indices import (
    ClimateIndexDownloader,
    SOURCES,
    _parse_oni,
    _parse_cpc_monthly_table,
    _parse_cpc_daily,
    _parse_mei,
    _parse_mjo,
    _parse_pdo,
    _parse_soi,
)


class TestParsers:
    def test_oni_parser(self):
        text = (
            "DJF 2020 26.6 0.5\n"
            "JFM 2020 26.7 0.4\n"
            "FMA 2020 27.0 0.4\n"
            "garbage line\n"
            "MAM 2020 27.4 0.3\n"
        )
        df = _parse_oni(text)
        assert len(df) == 4
        assert {"date", "oni"}.issubset(df.columns)
        assert df["oni"].iloc[0] == pytest.approx(0.5)

    def test_oni_djf_pinned_to_january_of_listed_year(self):
        """CPC labels DJF by the YR of February (the last month of the
        season). Center month is January of the same year — NOT year-1."""
        df = _parse_oni("DJF 2020 26.6 0.5\n")
        assert len(df) == 1
        # Expected: January 15, 2020 (center of Dec 2019 / Jan 2020 / Feb 2020).
        assert df["date"].iloc[0] == pd.Timestamp("2020-01-15")

    def test_oni_ndj_pinned_to_december(self):
        """NDJ 2019 = Nov/Dec 2019 + Jan 2020, centered on December 2019."""
        df = _parse_oni("NDJ 2019 26.6 0.5\n")
        assert len(df) == 1
        assert df["date"].iloc[0] == pd.Timestamp("2019-12-15")

    def test_oni_jja_centers_on_july(self):
        """Sanity check the simple seasons."""
        df = _parse_oni("JJA 2020 28.0 0.3\n")
        assert df["date"].iloc[0] == pd.Timestamp("2020-07-15")

    def test_oni_skips_invalid_year(self):
        text = "JFM ABCD 26.7 0.4\nFMA 2020 27.0 0.4\n"
        df = _parse_oni(text)
        assert len(df) == 1

    def test_cpc_monthly_factory(self):
        parser = _parse_cpc_monthly_table("nao")
        text = (
            "1950 -0.4 -1.2 -0.5 0.4 -0.1 -0.3 -0.7 0.6 0.5 0.4 -1.4 -2.0\n"
            "1951 -0.3  0.7 -0.2 0.6  0.4  0.1  0.7 0.6 0.4 0.0  0.6 -0.7\n"
        )
        df = parser(text)
        assert len(df) == 24
        assert df["nao"].min() == pytest.approx(-2.0)
        assert df["nao"].max() == pytest.approx(0.7)

    def test_cpc_monthly_skips_missing_sentinel(self):
        parser = _parse_cpc_monthly_table("nao")
        text = "2020 0.5 -99.9 -999.0 0.3 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n"
        df = parser(text)
        # Missing sentinels filtered out
        assert len(df) == 10

    def test_cpc_daily_factory(self):
        parser = _parse_cpc_daily("ao")
        text = (
            "2020 1 1 -0.5\n"
            "2020 1 2 -0.4\n"
            "2020 1 3 -999.0\n"
            "2020 1 4 0.2\n"
        )
        df = parser(text)
        assert len(df) == 3
        assert df["ao"].iloc[-1] == pytest.approx(0.2)

    def test_mei_parser(self):
        text = (
            "1979 -0.34 -0.45 0.12 0.34 0.45 0.56 -0.12 -0.34 0.0 0.1 0.2 0.3\n"
            "1980 0.5 -999 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0\n"
        )
        df = _parse_mei(text)
        # 12 from year 1, 11 from year 2 (one -999 sentinel dropped)
        assert len(df) == 23
        assert "mei" in df.columns

    def test_mjo_parser(self):
        text = (
            "1974 6 1 -0.123 0.234 5 1.456 garbage extra\n"
            "1974 6 2 0.456 -0.123 4 0.987 garbage extra\n"
            "1974 6 3 999 999 0 0 garbage extra\n"
        )
        df = _parse_mjo(text)
        # 999 is the missing flag
        assert len(df) == 2
        assert {"mjo_rmm1", "mjo_rmm2"}.issubset(df.columns)

    def test_pdo_parser(self):
        # PDO uses the same monthly table parser as NAO
        text = "1950 0.5 -0.3 0.2 0.1 0.0 -0.1 0.2 0.3 0.4 0.5 0.6 0.7\n"
        df = _parse_pdo(text)
        assert len(df) == 12
        assert df["pdo"].iloc[0] == pytest.approx(0.5)

    def test_soi_parser(self):
        text = (
            "199001 0.5\n"
            "199002 -0.3\n"
            "199003 -999.9\n"
            "199004 0.2\n"
        )
        df = _parse_soi(text)
        assert len(df) == 3


class TestDownloader:
    def test_to_daily_frame_forwards_monthly(self, tmp_path):
        dl = ClimateIndexDownloader(output_dir=tmp_path)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-15", "2020-02-15"]),
            "oni": [0.5, 0.7],
        })
        daily = dl.to_daily_frame({"enso": df}, start="2020-01-01")
        # Forward-fill within 31 days. Jan 15 covers 1/15..1/31..2/15 fine.
        jan_20 = daily[daily["date"] == pd.Timestamp("2020-01-20")]
        assert not jan_20.empty
        assert jan_20["oni"].iloc[0] == pytest.approx(0.5)
        feb_20 = daily[daily["date"] == pd.Timestamp("2020-02-20")]
        assert feb_20["oni"].iloc[0] == pytest.approx(0.7)

    def test_to_daily_frame_drops_pre_data(self, tmp_path):
        dl = ClimateIndexDownloader(output_dir=tmp_path)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-06-15"]),
            "oni": [0.5],
        })
        daily = dl.to_daily_frame({"enso": df}, start="2020-01-01")
        # Rows before data starts are NaN-only and dropped.
        assert daily["date"].min() >= pd.Timestamp("2020-06-15")

    def test_save_daily_parquet_roundtrip(self, tmp_path):
        dl = ClimateIndexDownloader(output_dir=tmp_path)
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-15", periods=3, freq="MS"),
            "oni": [0.5, 0.7, -0.1],
        })
        path = dl.save_daily_parquet({"enso": df})
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert "oni" in loaded.columns
        assert len(loaded) > 0

    def test_unknown_source_skipped(self, tmp_path):
        dl = ClimateIndexDownloader(output_dir=tmp_path)
        # Bogus key: should warn and return empty result rather than crash.
        result = dl.download(["this_index_does_not_exist"])
        assert result == {}

    def test_sources_have_required_fields(self):
        for key, src in SOURCES.items():
            assert src.name
            assert src.url.startswith("http"), key
            assert callable(src.parser), key
            assert src.cadence in {"daily", "monthly", "bimonthly"}
