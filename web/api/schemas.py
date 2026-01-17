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
