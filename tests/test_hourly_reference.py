from datetime import datetime, timedelta, timezone

from data.providers.open_meteo import OpenMeteoProvider


def test_hourly_reference_preserves_values_units_and_optional_missing_uv(monkeypatch):
    provider = OpenMeteoProvider()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    data = {"hourly": {"time": [(now + timedelta(hours=i)).replace(tzinfo=None).isoformat() for i in range(24)]}}
    hourly = data["hourly"]
    names = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
             "precipitation_probability", "wind_speed_10m", "wind_direction_10m", "cloud_cover", "surface_pressure"]
    for name, value in zip(names, [10, 8, 76, 1.2, 30, 8.7, 180, 45, 995]):
        hourly[name] = [value] * 24
    def fetch(url, params):
        assert params["timezone"] == "GMT"
        assert params["wind_speed_unit"] == "ms"
        return data
    monkeypatch.setattr(provider, "_get", fetch)
    rows = provider.hourly_forecast(1, 2, 24)
    assert len(rows) == 24
    assert rows[0]["wind_speed"] == 8.7
    assert rows[0]["precipitation_probability"] == 0.3
    assert rows[0]["uv_index"] is None
    assert rows[0]["datetime"].endswith("+00:00")
    hourly["relative_humidity_2m"][0] = None
    assert provider.hourly_forecast(1, 2, 24) is None
