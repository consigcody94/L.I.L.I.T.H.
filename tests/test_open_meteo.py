"""
Integration tests for the keyless Open-Meteo provider.

Network-gated: each test is skipped if Open-Meteo is unreachable, so the suite
still passes offline / in CI without network access.
"""

from datetime import datetime, timedelta

import pytest

from data.providers.open_meteo import OpenMeteoProvider


@pytest.fixture(scope="module")
def provider():
    return OpenMeteoProvider(timeout=20.0)


def _require(value):
    if value is None:
        pytest.skip("Open-Meteo unreachable (offline)")
    return value


def test_recent_history_shape(provider):
    hist = _require(provider.recent_history(40.71, -74.01, days=30))
    assert hist.shape[1] == 3
    assert hist.shape[0] >= 15  # at least a couple of weeks of real data
    assert (hist[:, 2] >= 0).all()  # precipitation non-negative


def test_reference_forecast_alignment(provider):
    ref = _require(provider.reference_forecast(40.71, -74.01, days=16))
    assert len(ref["dates"]) >= 7
    # Lead day 1 should be tomorrow.
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
    assert ref["dates"][0] == tomorrow
    assert ref["tmax"][0] is not None and ref["tmin"][0] is not None


def test_daily_observations_recent_and_archive(provider):
    recent = _require(
        provider.daily_observations(40.71, -74.01, datetime.now().date() - timedelta(days=10))
    )
    assert recent["tmax"] >= recent["tmin"]

    archived = _require(
        provider.daily_observations(40.71, -74.01, datetime.now().date() - timedelta(days=40))
    )
    assert archived["tmax"] >= archived["tmin"]


def test_future_date_returns_none(provider):
    assert provider.daily_observations(40.71, -74.01, datetime.now().date() + timedelta(days=5)) is None


def test_geocode(provider):
    result = _require(provider.geocode("Chicago"))
    assert "latitude" in result and "longitude" in result
    assert 41 < result["latitude"] < 42

    assert provider.geocode("zzzqx___nowhere_at_all") is None
