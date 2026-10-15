from datetime import date, timedelta

from fastapi.testclient import TestClient
import pytest

import web.api.main as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    from data.providers.open_meteo import OpenMeteoProvider
    monkeypatch.setenv("LILITH_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
    monkeypatch.setenv("LILITH_CHECKPOINT", str(tmp_path / "no-model.pt"))
    monkeypatch.setattr(api, "_verify_predictions_with_actuals", lambda: 0)
    def reference(*_):
        tomorrow = date.today() + timedelta(days=1)
        return {"dates": [(tomorrow + timedelta(days=i)).isoformat() for i in range(14)],
                "tmax": [10] * 14, "tmin": [2] * 14, "prcp": [1] * 14, "precip_prob": [20] * 14}
    monkeypatch.setattr(OpenMeteoProvider, "reference_forecast", reference)
    monkeypatch.setattr(OpenMeteoProvider, "hourly_forecast", lambda *_: None)
    with TestClient(api.app) as c:
        yield c


def test_issued_90_day_forecast_is_replayable(client):
    response = client.post("/v1/forecast", json={"latitude": 40.7, "longitude": -74, "days": 90})
    assert response.status_code == 200
    result = response.json()
    assert result["run_id"]
    replay = client.get(f"/v1/forecast/runs/{result['run_id']}")
    assert replay.json() == result
    assert len(result["forecasts"]) == 90
    assert result["forecasts"][0]["source"]
    assert result["forecasts"][0]["uncertainty_method"] == "heuristic_not_calibrated"
    report = client.get("/v1/accuracy").json()
    assert report["stats"]["total_predictions"] == 90
    assert report["independent_station_verified"] == 0


def test_archive_failure_does_not_return_untracked_success(client, monkeypatch):
    def fail(*_):
        raise OSError("disk full")
    monkeypatch.setattr(api._ledger, "archive", fail)
    response = client.post("/v1/forecast", json={"latitude": 40.7, "longitude": -74, "days": 1})
    assert response.status_code == 503


def test_hourly_outage_returns_503_not_fake_weather(client):
    assert client.post("/v1/forecast/hourly", json={"latitude": 0, "longitude": 0}).status_code == 503


def test_reporting_is_read_only_and_requires_coordinate_pair(client):
    assert client.get("/v1/accuracy?latitude=40").status_code == 422
    assert client.post("/v1/accuracy/verify").status_code == 405
    assert client.get("/v1/forecast/runs/missing").status_code == 404
