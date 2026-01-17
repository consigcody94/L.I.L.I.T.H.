"""
Tests for LILITH API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import date, datetime

from web.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns valid response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "version" in data


class TestForecastEndpoint:
    """Tests for forecast endpoint."""

    def test_forecast_basic(self, client):
        """Test basic forecast request."""
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 40.7128,
                "longitude": -74.006,
                "days": 14,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "location" in data
        assert "forecasts" in data
        assert len(data["forecasts"]) == 14

    def test_forecast_full_90_days(self, client):
        """Test 90-day forecast."""
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 51.5074,
                "longitude": -0.1278,
                "days": 90,
                "include_uncertainty": True,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["forecasts"]) == 90

        # Check uncertainty bounds present
        first_forecast = data["forecasts"][0]
        assert "temperature_max_lower" in first_forecast
        assert "temperature_max_upper" in first_forecast

    def test_forecast_validation(self, client):
        """Test input validation."""
        # Invalid latitude
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 100,  # Invalid
                "longitude": 0,
            }
        )
        assert response.status_code == 422

        # Invalid longitude
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 0,
                "longitude": 200,  # Invalid
            }
        )
        assert response.status_code == 422

        # Invalid days
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 0,
                "longitude": 0,
                "days": 100,  # > 90
            }
        )
        assert response.status_code == 422

    def test_forecast_response_structure(self, client):
        """Test forecast response has correct structure."""
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 35.6762,
                "longitude": 139.6503,
                "days": 7,
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level fields
        assert "location" in data
        assert "generated_at" in data
        assert "model_version" in data
        assert "forecast_days" in data
        assert "forecasts" in data

        # Check location
        assert data["location"]["latitude"] == pytest.approx(35.6762)
        assert data["location"]["longitude"] == pytest.approx(139.6503)

        # Check forecast structure
        forecast = data["forecasts"][0]
        assert "date" in forecast
        assert "temperature_max" in forecast
        assert "temperature_min" in forecast
        assert "precipitation" in forecast
        assert "precipitation_probability" in forecast

        # Check temperature relationship
        assert forecast["temperature_max"] >= forecast["temperature_min"]

        # Check precipitation probability range
        assert 0 <= forecast["precipitation_probability"] <= 1


class TestBatchForecastEndpoint:
    """Tests for batch forecast endpoint."""

    def test_batch_forecast(self, client):
        """Test batch forecast for multiple locations."""
        response = client.post(
            "/v1/forecast/batch",
            json={
                "locations": [
                    {"latitude": 40.7128, "longitude": -74.006},
                    {"latitude": 51.5074, "longitude": -0.1278},
                    {"latitude": 35.6762, "longitude": 139.6503},
                ],
                "days": 14,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "forecasts" in data
        assert "total_locations" in data
        assert "processing_time_ms" in data

        assert data["total_locations"] == 3
        assert len(data["forecasts"]) == 3

    def test_batch_forecast_empty(self, client):
        """Test batch forecast with empty locations."""
        response = client.post(
            "/v1/forecast/batch",
            json={
                "locations": [],
                "days": 14,
            }
        )

        # Should fail validation (min_length=1)
        assert response.status_code == 422


class TestStationsEndpoint:
    """Tests for stations endpoint."""

    def test_list_stations(self, client):
        """Test station listing."""
        response = client.get("/v1/stations")

        assert response.status_code == 200
        data = response.json()

        assert "stations" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    def test_list_stations_with_location_filter(self, client):
        """Test station listing with location filter."""
        response = client.get(
            "/v1/stations",
            params={
                "latitude": 40.7128,
                "longitude": -74.006,
                "radius": 2.0,
            }
        )

        assert response.status_code == 200

    def test_get_station_not_found(self, client):
        """Test getting non-existent station."""
        response = client.get("/v1/stations/NONEXISTENT123")

        assert response.status_code == 404


class TestHistoricalEndpoint:
    """Tests for historical data endpoint."""

    def test_historical_not_implemented(self, client):
        """Test historical endpoint returns not implemented."""
        response = client.post(
            "/v1/historical",
            json={
                "station_id": "USW00094728",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            }
        )

        # Should return 501 Not Implemented
        assert response.status_code == 501


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/v1/forecast",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/v1/forecast",
            json={
                "latitude": 40.7128,
                # Missing longitude
            }
        )

        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
