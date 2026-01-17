"""
Pydantic schemas for LILITH API.

Defines request and response models for all API endpoints.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """Geographic location."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")


class ForecastRequest(BaseModel):
    """Request for weather forecast."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    start_date: Optional[date] = Field(None, description="Start date (defaults to today)")
    days: int = Field(90, ge=1, le=90, description="Number of days to forecast")
    include_uncertainty: bool = Field(True, description="Include uncertainty bounds")
    ensemble_members: int = Field(10, ge=1, le=50, description="Number of ensemble members")
    variables: List[str] = Field(
        default=["temperature_max", "temperature_min", "precipitation"],
        description="Variables to forecast",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "days": 90,
                "include_uncertainty": True,
            }
        }


class DailyForecast(BaseModel):
    """Single day forecast."""

    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    temperature_max: float = Field(..., description="Maximum temperature (°C)")
    temperature_min: float = Field(..., description="Minimum temperature (°C)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    precipitation_probability: float = Field(..., ge=0, le=1, description="Probability of precipitation")

    # Uncertainty bounds (95% confidence interval)
    temperature_max_lower: Optional[float] = Field(None, description="Lower bound of max temp")
    temperature_max_upper: Optional[float] = Field(None, description="Upper bound of max temp")
    temperature_min_lower: Optional[float] = Field(None, description="Lower bound of min temp")
    temperature_min_upper: Optional[float] = Field(None, description="Upper bound of min temp")


class ForecastResponse(BaseModel):
    """Complete forecast response."""

    location: Location = Field(..., description="Forecast location")
    generated_at: datetime = Field(..., description="Generation timestamp")
    model_version: str = Field(..., description="Model version used")
    forecast_days: int = Field(..., description="Number of forecast days")
    forecasts: List[DailyForecast] = Field(..., description="Daily forecasts")

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 40.7128, "longitude": -74.0060},
                "generated_at": "2024-01-15T12:00:00Z",
                "model_version": "lilith-base-v1",
                "forecast_days": 90,
                "forecasts": [
                    {
                        "date": "2024-01-16",
                        "temperature_max": 5.2,
                        "temperature_min": -2.1,
                        "precipitation": 0.0,
                        "precipitation_probability": 0.1,
                    }
                ],
            }
        }


class BatchForecastRequest(BaseModel):
    """Request for multiple location forecasts."""

    locations: List[Location] = Field(..., min_length=1, max_length=100)
    days: int = Field(90, ge=1, le=90)
    include_uncertainty: bool = Field(True)


class BatchForecastResponse(BaseModel):
    """Response for multiple location forecasts."""

    forecasts: List[ForecastResponse]
    total_locations: int
    processing_time_ms: float


class StationInfo(BaseModel):
    """Weather station information."""

    station_id: str = Field(..., description="GHCN station ID")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    elevation: float = Field(..., description="Elevation (m)")
    country: str = Field(..., description="Country code")
    start_date: Optional[str] = Field(None, description="First observation date")
    end_date: Optional[str] = Field(None, description="Last observation date")


class StationListResponse(BaseModel):
    """Response for station list."""

    stations: List[StationInfo]
    total: int
    page: int
    page_size: int


class HistoricalRequest(BaseModel):
    """Request for historical data."""

    station_id: str = Field(..., description="GHCN station ID")
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")
    variables: List[str] = Field(
        default=["TMAX", "TMIN", "PRCP"],
        description="Variables to retrieve",
    )


class HistoricalObservation(BaseModel):
    """Single historical observation."""

    date: str
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    precipitation: Optional[float] = None


class HistoricalResponse(BaseModel):
    """Response for historical data."""

    station_id: str
    station_name: str
    observations: List[HistoricalObservation]
    total_observations: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


# ===== Hourly Forecast Schemas =====

class HourlyForecast(BaseModel):
    """Single hour forecast."""

    datetime: str = Field(..., description="Forecast datetime (ISO 8601)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    temperature: float = Field(..., description="Temperature (°C)")
    feels_like: float = Field(..., description="Feels like temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    precipitation_probability: float = Field(..., ge=0, le=1, description="Probability of precipitation")
    wind_speed: float = Field(..., ge=0, description="Wind speed (m/s)")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction (degrees)")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    pressure: float = Field(..., description="Atmospheric pressure (hPa)")
    uv_index: Optional[float] = Field(None, ge=0, description="UV index")

    # Uncertainty
    temperature_lower: Optional[float] = Field(None, description="Lower bound of temperature")
    temperature_upper: Optional[float] = Field(None, description="Upper bound of temperature")


class HourlyForecastRequest(BaseModel):
    """Request for hourly forecast."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    hours: int = Field(48, ge=1, le=168, description="Number of hours to forecast (max 7 days)")
    include_uncertainty: bool = Field(True, description="Include uncertainty bounds")


class HourlyForecastResponse(BaseModel):
    """Hourly forecast response."""

    location: Location = Field(..., description="Forecast location")
    generated_at: datetime = Field(..., description="Generation timestamp")
    model_version: str = Field(..., description="Model version used")
    forecast_hours: int = Field(..., description="Number of forecast hours")
    forecasts: List[HourlyForecast] = Field(..., description="Hourly forecasts")
    daily_summary: Optional[DailyForecast] = Field(None, description="Summary for first day")


# ===== Prediction Accuracy Schemas =====

class PredictionRecord(BaseModel):
    """Stored prediction for accuracy tracking."""

    id: str = Field(..., description="Unique prediction ID")
    location: Location = Field(..., description="Forecast location")
    location_name: Optional[str] = Field(None, description="Location name")
    predicted_at: datetime = Field(..., description="When prediction was made")
    target_date: str = Field(..., description="Date being predicted (YYYY-MM-DD)")

    # Predictions
    predicted_temp_max: float = Field(..., description="Predicted max temperature")
    predicted_temp_min: float = Field(..., description="Predicted min temperature")
    predicted_precipitation: float = Field(..., description="Predicted precipitation")
    predicted_precip_prob: float = Field(..., description="Predicted precipitation probability")

    # Actual observations (filled in once available)
    actual_temp_max: Optional[float] = Field(None, description="Actual max temperature")
    actual_temp_min: Optional[float] = Field(None, description="Actual min temperature")
    actual_precipitation: Optional[float] = Field(None, description="Actual precipitation")

    # Accuracy metrics (calculated once actuals are available)
    temp_max_error: Optional[float] = Field(None, description="Max temp prediction error")
    temp_min_error: Optional[float] = Field(None, description="Min temp prediction error")
    precip_error: Optional[float] = Field(None, description="Precipitation prediction error")
    lead_days: int = Field(..., description="Days ahead this was predicted")


class AccuracyStats(BaseModel):
    """Aggregated accuracy statistics."""

    total_predictions: int = Field(..., description="Total predictions tracked")
    verified_predictions: int = Field(..., description="Predictions with actuals")

    # Temperature accuracy
    temp_max_mae: float = Field(..., description="Mean Absolute Error for max temp (°C)")
    temp_max_rmse: float = Field(..., description="Root Mean Square Error for max temp (°C)")
    temp_min_mae: float = Field(..., description="Mean Absolute Error for min temp (°C)")
    temp_min_rmse: float = Field(..., description="Root Mean Square Error for min temp (°C)")

    # Precipitation accuracy
    precip_mae: float = Field(..., description="Mean Absolute Error for precipitation (mm)")
    precip_accuracy: float = Field(..., description="Precipitation occurrence accuracy (%)")

    # By lead time
    accuracy_by_lead_day: Dict[int, Dict[str, float]] = Field(
        ..., description="Accuracy metrics broken down by forecast lead time"
    )


class AccuracyReportRequest(BaseModel):
    """Request for accuracy report."""

    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Filter by latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Filter by longitude")
    days_back: int = Field(30, ge=1, le=365, description="Number of days to look back")
    location_name: Optional[str] = Field(None, description="Filter by location name")


class AccuracyReportResponse(BaseModel):
    """Accuracy report response."""

    generated_at: datetime = Field(..., description="Report generation timestamp")
    period_start: str = Field(..., description="Start of analysis period")
    period_end: str = Field(..., description="End of analysis period")
    stats: AccuracyStats = Field(..., description="Aggregated statistics")
    recent_predictions: List[PredictionRecord] = Field(..., description="Recent prediction records")
    location_filter: Optional[str] = Field(None, description="Location filter applied")


# ===== METAR Monitoring Schemas =====

class MetarStation(BaseModel):
    """METAR station information."""

    icao: str = Field(..., description="ICAO airport code (e.g., KJFK)")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    elevation_m: Optional[float] = Field(None, description="Elevation in meters")
    raw_metar: Optional[str] = Field(None, description="Raw METAR string")
    observation_time: Optional[str] = Field(None, description="Observation time")
    is_flagged: bool = Field(False, description="Whether METAR has $ flag (maintenance needed)")
    is_missing: bool = Field(False, description="Whether METAR is missing/stale")
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    dewpoint_c: Optional[float] = Field(None, description="Dewpoint in Celsius")
    wind_speed_kt: Optional[float] = Field(None, description="Wind speed in knots")
    wind_dir: Optional[float] = Field(None, description="Wind direction in degrees")
    visibility_sm: Optional[float] = Field(None, description="Visibility in statute miles")
    altimeter_inhg: Optional[float] = Field(None, description="Altimeter setting in inHg")
    weather: Optional[str] = Field(None, description="Weather phenomena")
    clouds: Optional[str] = Field(None, description="Cloud layers")
    last_checked: Optional[datetime] = Field(None, description="Last time this station was checked")


class MetarMonitorResponse(BaseModel):
    """Response for METAR monitoring."""

    generated_at: datetime = Field(..., description="Response timestamp")
    total_stations: int = Field(..., description="Total stations monitored")
    flagged_count: int = Field(..., description="Stations with $ flag")
    missing_count: int = Field(..., description="Stations with missing/stale data")
    healthy_count: int = Field(..., description="Healthy stations")
    stations: List[MetarStation] = Field(..., description="Station data")
    next_update_seconds: int = Field(300, description="Seconds until next update")
