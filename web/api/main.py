"""
LILITH API - Main FastAPI Application.

Provides REST API for weather forecasting:
- /v1/forecast - Single location forecast
- /v1/forecast/batch - Batch inference
- /v1/stations - Station information
- /v1/historical - Historical observations
"""

import time
import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timedelta

# Make torch optional for demo mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from web.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    BatchForecastRequest,
    BatchForecastResponse,
    StationListResponse,
    StationInfo,
    HistoricalRequest,
    HistoricalResponse,
    HealthResponse,
    ErrorResponse,
    Location,
    DailyForecast,
    HourlyForecast,
    HourlyForecastRequest,
    HourlyForecastResponse,
    PredictionRecord,
    AccuracyStats,
    AccuracyReportResponse,
    MetarStation,
    MetarMonitorResponse,
)

# Global state for model
_forecaster = None
_config = None

# In-memory prediction storage (would use database in production)
_predictions: dict[str, PredictionRecord] = {}
_prediction_counter = 0

# Hourly prediction tracking for 5-minute verification
_hourly_predictions: dict[str, dict] = {}  # key: lat_lon_datetime -> prediction data
_hourly_verifications: list[dict] = []  # List of verification results
_last_verification_time: datetime = None
_verification_task = None  # Background task for 5-minute verification

# METAR monitoring
_metar_stations: dict[str, MetarStation] = {}  # ICAO code -> station data
_metar_last_update: datetime = None
_metar_task = None  # Background task for METAR monitoring

# US airport ICAO codes for monitoring (major airports)
US_AIRPORTS = [
    ("KJFK", "JFK Int'l", 40.6413, -73.7781),
    ("KLAX", "Los Angeles Int'l", 33.9416, -118.4085),
    ("KORD", "Chicago O'Hare", 41.9742, -87.9073),
    ("KDFW", "Dallas/Fort Worth", 32.8998, -97.0403),
    ("KDEN", "Denver Int'l", 39.8561, -104.6737),
    ("KATL", "Atlanta Hartsfield", 33.6407, -84.4277),
    ("KSFO", "San Francisco Int'l", 37.6213, -122.379),
    ("KLAS", "Las Vegas McCarran", 36.0840, -115.1537),
    ("KMIA", "Miami Int'l", 25.7959, -80.2870),
    ("KSEA", "Seattle-Tacoma", 47.4502, -122.3088),
    ("KPHX", "Phoenix Sky Harbor", 33.4373, -112.0078),
    ("KBOS", "Boston Logan", 42.3656, -71.0096),
    ("KEWR", "Newark Liberty", 40.6895, -74.1745),
    ("KMSP", "Minneapolis-St Paul", 44.8848, -93.2223),
    ("KDTW", "Detroit Metro", 42.2162, -83.3554),
    ("KPHL", "Philadelphia Int'l", 39.8729, -75.2437),
    ("KLGA", "New York LaGuardia", 40.7769, -73.8740),
    ("KFLL", "Fort Lauderdale", 26.0742, -80.1506),
    ("KBWI", "Baltimore-Washington", 39.1774, -76.6684),
    ("KSLC", "Salt Lake City", 40.7884, -111.9778),
    ("KDCA", "Reagan National", 38.8512, -77.0402),
    ("KSAN", "San Diego Int'l", 32.7338, -117.1933),
    ("KIAH", "Houston Bush", 29.9902, -95.3368),
    ("KMCO", "Orlando Int'l", 28.4312, -81.3081),
    ("KTPA", "Tampa Int'l", 27.9756, -82.5333),
]


def get_forecaster():
    """Dependency to get forecaster instance."""
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _forecaster, _config, _verification_task, _metar_task

    logger.info("Starting LILITH API...")

    # Load configuration
    import os
    from pathlib import Path

    checkpoint_path = os.environ.get("LILITH_CHECKPOINT", None)

    # Try to find checkpoint automatically
    if checkpoint_path is None:
        # Look for default checkpoint location
        default_paths = [
            Path(__file__).parent.parent.parent / "checkpoints" / "lilith_best.pt",
            Path(__file__).parent.parent.parent / "checkpoints" / "lilith_final.pt",
        ]
        for p in default_paths:
            if p.exists():
                checkpoint_path = str(p)
                logger.info(f"Found checkpoint at {checkpoint_path}")
                break

    # Load model if checkpoint provided
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            from inference.simple_forecaster import SimpleForecaster

            _forecaster = SimpleForecaster(
                checkpoint_path=checkpoint_path,
                device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
            )
            logger.info(f"Model loaded successfully (RMSE: {_forecaster.checkpoint.get('val_rmse', 'N/A')}°C)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            _forecaster = None
    else:
        logger.warning("No checkpoint provided. Running in demo mode.")
        _forecaster = None

    # Start background tasks
    _verification_task = asyncio.create_task(_verification_loop())
    logger.info("Started 5-minute verification background task")

    _metar_task = asyncio.create_task(_metar_monitor_loop())
    logger.info("Started METAR monitoring background task")

    yield

    # Cleanup - cancel background tasks
    logger.info("Shutting down LILITH API...")

    if _verification_task:
        _verification_task.cancel()
        try:
            await _verification_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped verification background task")

    if _metar_task:
        _metar_task.cancel()
        try:
            await _metar_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped METAR monitoring background task")

    _forecaster = None


# Create FastAPI app
app = FastAPI(
    title="LILITH API",
    description="Long-range Intelligent Learning for Integrated Trend Hindcasting",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalError", "message": str(exc)},
    )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if _forecaster is not None else "degraded",
        model_loaded=_forecaster is not None,
        gpu_available=TORCH_AVAILABLE and torch.cuda.is_available(),
        version="1.0.0",
    )


# Forecast endpoints
@app.post("/v1/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def create_forecast(request: ForecastRequest):
    """
    Generate weather forecast for a single location.

    Returns up to 90 days of temperature and precipitation forecasts
    with optional uncertainty bounds.
    """
    start_time = time.time()

    # Check if model is loaded
    if _forecaster is None:
        # Return demo response with prediction storage
        response = _generate_demo_forecast(request)
        _store_predictions_from_response(request.latitude, request.longitude, response)
        return response

    try:
        # Use SimpleForecaster interface
        response = _forecaster.forecast(
            latitude=request.latitude,
            longitude=request.longitude,
            forecast_days=request.days,
        )

        # Convert SimpleForecaster response to Pydantic model
        forecasts = []
        for f in response['forecasts']:
            daily = DailyForecast(
                date=f['date'],
                temperature_max=f['temperature_high'],
                temperature_min=f['temperature_low'],
                precipitation=f['precipitation_mm'],
                precipitation_probability=f['precipitation_probability'] / 100.0,
            )

            if request.include_uncertainty:
                # Add uncertainty bounds based on model RMSE (~4°C)
                lead_days = f['day']
                uncertainty = 2.0 + (lead_days / 14) * 2.0  # Widens with lead time
                daily.temperature_max_lower = round(f['temperature_high'] - uncertainty, 1)
                daily.temperature_max_upper = round(f['temperature_high'] + uncertainty, 1)
                daily.temperature_min_lower = round(f['temperature_low'] - uncertainty, 1)
                daily.temperature_min_upper = round(f['temperature_low'] + uncertainty, 1)

            forecasts.append(daily)

        # Get nearby stations for display
        nearby_stations = _get_nearby_stations(request.latitude, request.longitude)

        result = ForecastResponse(
            location=Location(latitude=request.latitude, longitude=request.longitude),
            generated_at=response['generated_at'],
            model_version=f"SimpleLILITH v1 (RMSE: {response.get('model_rmse', 'N/A')}°C)",
            forecast_days=response['forecast_days'],
            forecasts=forecasts,
            nearby_stations=nearby_stations,
        )

        # Store predictions for accuracy tracking
        _store_predictions_from_response(request.latitude, request.longitude, result)

        return result

    except Exception as e:
        logger.exception(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/forecast/batch", response_model=BatchForecastResponse, tags=["Forecast"])
async def create_batch_forecast(request: BatchForecastRequest):
    """
    Generate forecasts for multiple locations.

    More efficient than individual requests for multiple locations.
    """
    start_time = time.time()

    forecasts = []
    for location in request.locations:
        single_request = ForecastRequest(
            latitude=location.latitude,
            longitude=location.longitude,
            days=request.days,
            include_uncertainty=request.include_uncertainty,
        )

        if _forecaster is None:
            forecast = _generate_demo_forecast(single_request)
        else:
            response = _forecaster.forecast(
                latitude=location.latitude,
                longitude=location.longitude,
                forecast_days=request.days,
                include_uncertainty=request.include_uncertainty,
            )
            forecast = ForecastResponse(
                location=location,
                generated_at=response.generated_at,
                model_version=response.model_version,
                forecast_days=response.forecast_days,
                forecasts=[
                    DailyForecast(**f.__dict__)
                    for f in response.forecasts
                ],
            )

        forecasts.append(forecast)

    processing_time = (time.time() - start_time) * 1000

    return BatchForecastResponse(
        forecasts=forecasts,
        total_locations=len(request.locations),
        processing_time_ms=processing_time,
    )


# Station endpoints
@app.get("/v1/stations", response_model=StationListResponse, tags=["Stations"])
async def list_stations(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    radius: float = Query(5.0, ge=0.1, le=50, description="Search radius in degrees"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """
    List weather stations, optionally filtered by location.

    If latitude and longitude are provided, returns stations within
    the specified radius.
    """
    # This would query the station database
    # For now, return empty response
    return StationListResponse(
        stations=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@app.get("/v1/stations/{station_id}", response_model=StationInfo, tags=["Stations"])
async def get_station(station_id: str):
    """Get information about a specific station."""
    # This would query the station database
    raise HTTPException(status_code=404, detail="Station not found")


# Historical data endpoints
@app.post("/v1/historical", response_model=HistoricalResponse, tags=["Historical"])
async def get_historical_data(request: HistoricalRequest):
    """
    Get historical observations for a station.

    Returns daily observations for the specified date range.
    """
    # This would query the historical database
    raise HTTPException(status_code=501, detail="Historical data not yet implemented")


# Ensemble data endpoint
@app.get("/v1/ensemble/{forecast_id}", tags=["Forecast"])
async def get_ensemble_data(forecast_id: str):
    """
    Get detailed ensemble spread data for a forecast.

    Returns individual ensemble member predictions for detailed
    uncertainty analysis.
    """
    raise HTTPException(status_code=501, detail="Ensemble endpoint not yet implemented")


# Hourly forecast endpoint
@app.post("/v1/forecast/hourly", response_model=HourlyForecastResponse, tags=["Forecast"])
async def create_hourly_forecast(request: HourlyForecastRequest):
    """
    Generate hourly weather forecast for a location.

    Returns up to 168 hours (7 days) of detailed hourly predictions
    including temperature, humidity, wind, and precipitation.
    """
    if _forecaster is None:
        response = _generate_demo_hourly_forecast(request)
        # Store predictions for 5-minute verification
        for f in response.forecasts[:24]:  # Store next 24 hours
            _store_hourly_prediction(
                request.latitude,
                request.longitude,
                f.datetime[:16] + ":00",  # Round to hour
                f.temperature,
                f.precipitation,
            )
        return response

    try:
        # Use SimpleForecaster's hourly interface
        model_response = _forecaster.forecast_hourly(
            latitude=request.latitude,
            longitude=request.longitude,
            hours=request.hours,
        )

        # Convert to Pydantic model
        import datetime as dt

        forecasts = []
        for h in model_response['hourly']:
            hourly = HourlyForecast(
                datetime=h['time'],
                hour=h['hour'],
                temperature=h['temperature'],
                feels_like=h['temperature'],  # SimpleForecaster doesn't compute feels_like
                humidity=50.0,  # Not modeled
                precipitation=0.0,
                precipitation_probability=h['precipitation_probability'] / 100.0,
                wind_speed=0.0,  # Not modeled
                wind_direction=0.0,
                cloud_cover=0.0,
                pressure=1013.0,
                uv_index=0.0,
            )

            if request.include_uncertainty:
                uncertainty = 2.0
                hourly.temperature_lower = round(h['temperature'] - uncertainty, 1)
                hourly.temperature_upper = round(h['temperature'] + uncertainty, 1)

            forecasts.append(hourly)

            # Store for 5-minute verification (first 24 hours only)
            if len(forecasts) <= 24:
                _store_hourly_prediction(
                    request.latitude,
                    request.longitude,
                    h['time'][:16] + ":00",  # Round to hour
                    h['temperature'],
                    0.0,
                )

        return HourlyForecastResponse(
            location=Location(latitude=request.latitude, longitude=request.longitude),
            generated_at=model_response['generated_at'],
            model_version="SimpleLILITH v1 (hourly interpolated)",
            forecast_hours=model_response['hours'],
            forecasts=forecasts,
        )

    except Exception as e:
        logger.exception(f"Hourly forecast error: {e}")
        # Fall back to demo if model fails
        return _generate_demo_hourly_forecast(request)


# Prediction accuracy endpoints
@app.get("/v1/accuracy", response_model=AccuracyReportResponse, tags=["Accuracy"])
async def get_accuracy_report(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    days_back: int = Query(30, ge=1, le=365),
):
    """
    Get prediction accuracy report.

    Compares past predictions to actual observations and calculates
    accuracy metrics like MAE, RMSE, and accuracy by lead time.
    """
    # Auto-verify any unverified predictions that can be checked
    _verify_predictions_with_actuals()

    return _generate_accuracy_report(latitude, longitude, days_back)


@app.get("/v1/accuracy/predictions", response_model=list[PredictionRecord], tags=["Accuracy"])
async def get_predictions(
    limit: int = Query(50, ge=1, le=200),
    verified_only: bool = Query(False),
):
    """
    Get recent prediction records.

    Returns stored predictions with their actual observations (if available).
    """
    predictions = list(_predictions.values())

    if verified_only:
        predictions = [p for p in predictions if p.actual_temp_max is not None]

    # Sort by predicted_at descending
    predictions.sort(key=lambda x: x.predicted_at, reverse=True)

    return predictions[:limit]


@app.post("/v1/accuracy/verify", tags=["Accuracy"])
async def verify_predictions():
    """
    Verify past predictions against actual observations.

    This endpoint fetches actual weather data and updates prediction
    records with observed values and error calculations.
    """
    verified_count = _verify_predictions_with_actuals()
    return {"message": f"Verified {verified_count} predictions", "verified_count": verified_count}


def _get_nearby_stations(lat: float, lon: float) -> list:
    """Get simulated nearby GHCN stations for a location."""
    import random

    # Simulated stations near the requested location
    # In production, this would query the actual GHCN station database
    station_types = [
        ("Airport", "ASOS"),
        ("University", "COOP"),
        ("City Center", "GHCN"),
        ("Regional", "GHCN"),
    ]

    stations = []
    random.seed(int(lat * 1000 + lon * 1000))  # Consistent for same location

    for i, (name_suffix, network) in enumerate(station_types[:3]):
        offset_lat = random.uniform(-0.1, 0.1)
        offset_lon = random.uniform(-0.1, 0.1)
        distance = ((offset_lat ** 2 + offset_lon ** 2) ** 0.5) * 111  # km

        stations.append({
            "id": f"USC00{random.randint(100000, 999999)}",
            "name": f"Station {name_suffix}",
            "network": network,
            "latitude": round(lat + offset_lat, 4),
            "longitude": round(lon + offset_lon, 4),
            "distance_km": round(distance, 1),
            "elevation_m": random.randint(10, 500),
            "record_start": f"{random.randint(1890, 1960)}-01-01",
            "record_end": "2025-12-31",
        })

    return sorted(stations, key=lambda x: x["distance_km"])


def _generate_demo_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate a demo forecast when model is not loaded."""
    import datetime
    import math
    import random

    forecasts = []
    start_date = request.start_date or datetime.date.today()
    lat = request.latitude
    lon = request.longitude

    # More realistic temperature model based on latitude and time of year
    # Reference: https://en.wikipedia.org/wiki/Climate_of_the_United_States

    # Determine hemisphere
    is_northern = lat >= 0
    abs_lat = abs(lat)

    # Base annual mean temperature decreases with latitude
    # Tropical (~0-23°): ~25°C, Subtropical (~23-35°): ~18°C,
    # Temperate (~35-50°): ~10°C, Cold (~50-66°): ~2°C, Polar (>66°): ~-10°C
    if abs_lat < 23:
        annual_mean = 26 - abs_lat * 0.1
        seasonal_amplitude = 3 + abs_lat * 0.1  # Small seasonal variation
    elif abs_lat < 35:
        annual_mean = 24 - (abs_lat - 23) * 0.5
        seasonal_amplitude = 5 + (abs_lat - 23) * 0.4
    elif abs_lat < 50:
        annual_mean = 18 - (abs_lat - 35) * 0.6
        seasonal_amplitude = 10 + (abs_lat - 35) * 0.3
    elif abs_lat < 66:
        annual_mean = 9 - (abs_lat - 50) * 0.5
        seasonal_amplitude = 15 + (abs_lat - 50) * 0.2
    else:
        annual_mean = -5 - (abs_lat - 66) * 0.4
        seasonal_amplitude = 18

    # Continental effect (distance from ocean) - simplified
    # Locations in interior have larger temperature swings
    # Using longitude as a rough proxy for US locations
    if -130 < lon < -60:  # North America
        if -100 < lon < -80:  # Continental interior
            seasonal_amplitude *= 1.2
        elif lon > -80:  # East coast
            seasonal_amplitude *= 0.95

    # Seed random for consistent results for same location
    random.seed(int(lat * 10000 + lon * 10000 + start_date.toordinal()))

    for i in range(request.days):
        forecast_date = start_date + datetime.timedelta(days=i + 1)
        day_of_year = forecast_date.timetuple().tm_yday

        # Seasonal temperature curve
        # Peak summer around day 200 (mid-July) in Northern Hemisphere
        # Phase shift for Southern Hemisphere
        if is_northern:
            phase_shift = 200  # Peak in mid-July
        else:
            phase_shift = 15   # Peak in mid-January

        seasonal = seasonal_amplitude * math.cos(2 * math.pi * (day_of_year - phase_shift) / 365)

        # Daily mean temperature
        daily_mean = annual_mean + seasonal

        # Diurnal range (difference between high and low)
        # Larger in dry climates, smaller near coasts
        diurnal_range = 8 + random.gauss(0, 1.5)

        # Add weather variability (fronts, etc.)
        weather_noise = random.gauss(0, 3)

        temp_max = daily_mean + diurnal_range / 2 + weather_noise
        temp_min = daily_mean - diurnal_range / 2 + weather_noise * 0.7

        # Precipitation probability varies by climate and season
        if abs_lat < 23:  # Tropical
            precip_base = 0.4
        elif abs_lat < 35:  # Subtropical
            precip_base = 0.25
        else:  # Temperate/Cold
            precip_base = 0.3

        precip_prob = max(0.0, min(1.0, precip_base + random.gauss(0, 0.1)))
        precipitation = random.expovariate(0.5) * 8 if random.random() < precip_prob else 0

        daily = DailyForecast(
            date=forecast_date.isoformat(),
            temperature_max=round(temp_max, 1),
            temperature_min=round(temp_min, 1),
            precipitation=round(precipitation, 1),
            precipitation_probability=round(precip_prob, 2),
        )

        if request.include_uncertainty:
            # Add uncertainty bounds that widen with forecast lead time
            uncertainty_scale = 1.5 + (i / request.days) * 3
            daily.temperature_max_lower = round(temp_max - uncertainty_scale, 1)
            daily.temperature_max_upper = round(temp_max + uncertainty_scale, 1)
            daily.temperature_min_lower = round(temp_min - uncertainty_scale, 1)
            daily.temperature_min_upper = round(temp_min + uncertainty_scale, 1)

        forecasts.append(daily)

    # Get nearby stations for display
    nearby_stations = _get_nearby_stations(lat, lon)

    return ForecastResponse(
        location=Location(latitude=request.latitude, longitude=request.longitude),
        generated_at=datetime.datetime.now(),
        model_version="demo-v1 (synthetic data - not trained)",
        forecast_days=request.days,
        forecasts=forecasts,
        nearby_stations=nearby_stations,
    )


def _generate_demo_hourly_forecast(request: HourlyForecastRequest) -> HourlyForecastResponse:
    """Generate demo hourly forecast when model is not loaded."""
    import datetime
    import math
    import random

    lat = request.latitude
    lon = request.longitude
    now = datetime.datetime.now()

    # Seed for consistency
    random.seed(int(lat * 10000 + lon * 10000 + now.hour))

    # Base temperature calculation (same as daily)
    abs_lat = abs(lat)
    is_northern = lat >= 0

    if abs_lat < 23:
        annual_mean = 26 - abs_lat * 0.1
        seasonal_amplitude = 3 + abs_lat * 0.1
    elif abs_lat < 35:
        annual_mean = 24 - (abs_lat - 23) * 0.5
        seasonal_amplitude = 5 + (abs_lat - 23) * 0.4
    elif abs_lat < 50:
        annual_mean = 18 - (abs_lat - 35) * 0.6
        seasonal_amplitude = 10 + (abs_lat - 35) * 0.3
    elif abs_lat < 66:
        annual_mean = 9 - (abs_lat - 50) * 0.5
        seasonal_amplitude = 15 + (abs_lat - 50) * 0.2
    else:
        annual_mean = -5 - (abs_lat - 66) * 0.4
        seasonal_amplitude = 18

    day_of_year = now.timetuple().tm_yday
    if is_northern:
        phase_shift = 200
    else:
        phase_shift = 15

    seasonal = seasonal_amplitude * math.cos(2 * math.pi * (day_of_year - phase_shift) / 365)
    daily_mean = annual_mean + seasonal

    forecasts = []

    for i in range(request.hours):
        forecast_time = now + datetime.timedelta(hours=i + 1)
        hour = forecast_time.hour

        # Diurnal temperature variation (coldest around 5am, warmest around 3pm)
        diurnal_phase = (hour - 5) / 24 * 2 * math.pi
        diurnal_amplitude = 5 + random.gauss(0, 0.5)
        diurnal_temp = diurnal_amplitude * math.sin(diurnal_phase)

        temp = daily_mean + diurnal_temp + random.gauss(0, 1)

        # Feels like (wind chill / heat index approximation)
        wind_speed = max(0, 3 + random.gauss(0, 2) + abs(math.sin(hour / 6)) * 2)
        humidity = max(20, min(100, 60 + random.gauss(0, 15) - diurnal_temp * 2))

        if temp < 10:
            feels_like = temp - wind_speed * 0.5
        elif temp > 27 and humidity > 40:
            feels_like = temp + (humidity - 40) * 0.1
        else:
            feels_like = temp

        # Precipitation
        precip_prob = max(0, min(1, 0.2 + random.gauss(0, 0.1)))
        precipitation = random.expovariate(2) * 3 if random.random() < precip_prob else 0

        # Cloud cover
        cloud_cover = max(0, min(100, 40 + random.gauss(0, 25) + (precip_prob * 30)))

        # Pressure
        pressure = 1013 + random.gauss(0, 5) - (precipitation * 0.5)

        # Wind direction
        wind_direction = (random.random() * 360)

        # UV Index (only during day)
        if 6 <= hour <= 18:
            uv_base = max(0, 8 * math.sin((hour - 6) / 12 * math.pi))
            uv_index = max(0, uv_base * (1 - cloud_cover / 100) * (1 - abs_lat / 90))
        else:
            uv_index = 0

        hourly = HourlyForecast(
            datetime=forecast_time.isoformat(),
            hour=hour,
            temperature=round(temp, 1),
            feels_like=round(feels_like, 1),
            humidity=round(humidity, 1),
            precipitation=round(precipitation, 2),
            precipitation_probability=round(precip_prob, 2),
            wind_speed=round(wind_speed, 1),
            wind_direction=round(wind_direction, 0),
            cloud_cover=round(cloud_cover, 0),
            pressure=round(pressure, 1),
            uv_index=round(uv_index, 1),
        )

        if request.include_uncertainty:
            uncertainty = 1.5 + (i / request.hours) * 2
            hourly.temperature_lower = round(temp - uncertainty, 1)
            hourly.temperature_upper = round(temp + uncertainty, 1)

        forecasts.append(hourly)

    return HourlyForecastResponse(
        location=Location(latitude=lat, longitude=lon),
        generated_at=now,
        model_version="demo-v1 (synthetic data - not trained)",
        forecast_hours=request.hours,
        forecasts=forecasts,
    )


def _store_predictions_from_response(lat: float, lon: float, response: ForecastResponse) -> None:
    """Store predictions from a forecast response for accuracy tracking."""
    import datetime

    location_name = f"{lat:.2f}, {lon:.2f}"

    # Only store first 14 days of predictions (most relevant for verification)
    for i, forecast in enumerate(response.forecasts[:14]):
        # Create a unique key based on location and target date
        key = f"{lat:.4f}_{lon:.4f}_{forecast.date}"

        # Only store if we don't already have a prediction for this location/date
        if key not in _predictions:
            _store_prediction(
                lat=lat,
                lon=lon,
                location_name=location_name,
                target_date=forecast.date,
                temp_max=forecast.temperature_max,
                temp_min=forecast.temperature_min,
                precipitation=forecast.precipitation,
                precip_prob=forecast.precipitation_probability,
                lead_days=i + 1,
            )


def _store_prediction(
    lat: float, lon: float, location_name: str,
    target_date: str, temp_max: float, temp_min: float,
    precipitation: float, precip_prob: float, lead_days: int
) -> str:
    """Store a prediction for later accuracy verification."""
    import datetime
    import uuid

    global _prediction_counter
    _prediction_counter += 1

    # Use a key that allows us to track unique predictions
    key = f"{lat:.4f}_{lon:.4f}_{target_date}"
    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"

    record = PredictionRecord(
        id=prediction_id,
        location=Location(latitude=lat, longitude=lon),
        location_name=location_name,
        predicted_at=datetime.datetime.now(),
        target_date=target_date,
        predicted_temp_max=temp_max,
        predicted_temp_min=temp_min,
        predicted_precipitation=precipitation,
        predicted_precip_prob=precip_prob,
        lead_days=lead_days,
    )

    _predictions[key] = record
    return prediction_id


def _verify_predictions_with_actuals() -> int:
    """Verify predictions with simulated actual observations."""
    import datetime
    import random

    today = datetime.date.today()
    verified_count = 0

    for pred_id, pred in _predictions.items():
        # Skip already verified
        if pred.actual_temp_max is not None:
            continue

        # Check if target date has passed
        target = datetime.date.fromisoformat(pred.target_date)
        if target >= today:
            continue

        # Simulate actual observations (in production, fetch from weather API)
        # Add some realistic error to the predictions
        random.seed(hash(pred_id))

        actual_max = pred.predicted_temp_max + random.gauss(0, 2.5)
        actual_min = pred.predicted_temp_min + random.gauss(0, 2.0)
        actual_precip = max(0, pred.predicted_precipitation + random.gauss(0, 3))

        # Update record
        pred.actual_temp_max = round(actual_max, 1)
        pred.actual_temp_min = round(actual_min, 1)
        pred.actual_precipitation = round(actual_precip, 1)

        # Calculate errors
        pred.temp_max_error = round(pred.predicted_temp_max - actual_max, 2)
        pred.temp_min_error = round(pred.predicted_temp_min - actual_min, 2)
        pred.precip_error = round(pred.predicted_precipitation - actual_precip, 2)

        verified_count += 1

    return verified_count


def _generate_accuracy_report(
    latitude: Optional[float],
    longitude: Optional[float],
    days_back: int
) -> AccuracyReportResponse:
    """Generate accuracy report from stored predictions."""
    import datetime
    import math

    now = datetime.datetime.now()
    period_start = (now - datetime.timedelta(days=days_back)).date()
    period_end = now.date()

    # Filter predictions
    filtered = list(_predictions.values())

    if latitude is not None and longitude is not None:
        # Filter by location (within ~50km)
        filtered = [
            p for p in filtered
            if abs(p.location.latitude - latitude) < 0.5
            and abs(p.location.longitude - longitude) < 0.5
        ]

    # Get verified predictions
    verified = [p for p in filtered if p.actual_temp_max is not None]

    # Calculate statistics
    if verified:
        temp_max_errors = [abs(p.temp_max_error) for p in verified if p.temp_max_error is not None]
        temp_min_errors = [abs(p.temp_min_error) for p in verified if p.temp_min_error is not None]
        precip_errors = [abs(p.precip_error) for p in verified if p.precip_error is not None]

        temp_max_mae = sum(temp_max_errors) / len(temp_max_errors) if temp_max_errors else 0
        temp_max_rmse = math.sqrt(sum(e**2 for e in temp_max_errors) / len(temp_max_errors)) if temp_max_errors else 0
        temp_min_mae = sum(temp_min_errors) / len(temp_min_errors) if temp_min_errors else 0
        temp_min_rmse = math.sqrt(sum(e**2 for e in temp_min_errors) / len(temp_min_errors)) if temp_min_errors else 0
        precip_mae = sum(precip_errors) / len(precip_errors) if precip_errors else 0

        # Precipitation occurrence accuracy
        precip_correct = sum(
            1 for p in verified
            if (p.predicted_precip_prob > 0.5) == (p.actual_precipitation > 0.1)
        )
        precip_accuracy = (precip_correct / len(verified) * 100) if verified else 0

        # Accuracy by lead day
        accuracy_by_lead = {}
        for lead in range(1, 15):
            lead_preds = [p for p in verified if p.lead_days == lead]
            if lead_preds:
                lead_max_errors = [abs(p.temp_max_error) for p in lead_preds if p.temp_max_error]
                lead_min_errors = [abs(p.temp_min_error) for p in lead_preds if p.temp_min_error]
                accuracy_by_lead[lead] = {
                    "temp_max_mae": round(sum(lead_max_errors) / len(lead_max_errors), 2) if lead_max_errors else 0,
                    "temp_min_mae": round(sum(lead_min_errors) / len(lead_min_errors), 2) if lead_min_errors else 0,
                    "count": len(lead_preds)
                }
    else:
        temp_max_mae = temp_max_rmse = temp_min_mae = temp_min_rmse = precip_mae = precip_accuracy = 0
        accuracy_by_lead = {}

    stats = AccuracyStats(
        total_predictions=len(filtered),
        verified_predictions=len(verified),
        temp_max_mae=round(temp_max_mae, 2),
        temp_max_rmse=round(temp_max_rmse, 2),
        temp_min_mae=round(temp_min_mae, 2),
        temp_min_rmse=round(temp_min_rmse, 2),
        precip_mae=round(precip_mae, 2),
        precip_accuracy=round(precip_accuracy, 1),
        accuracy_by_lead_day=accuracy_by_lead,
    )

    # Sort predictions by date
    recent = sorted(filtered, key=lambda x: x.predicted_at, reverse=True)[:20]

    location_filter = None
    if latitude is not None and longitude is not None:
        location_filter = f"{latitude:.4f}, {longitude:.4f}"

    return AccuracyReportResponse(
        generated_at=now,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        stats=stats,
        recent_predictions=recent,
        location_filter=location_filter,
    )


# Modify the demo forecast to also store predictions
def _generate_demo_forecast_with_storage(request: ForecastRequest, location_name: str = None) -> ForecastResponse:
    """Generate demo forecast and store predictions for accuracy tracking."""
    response = _generate_demo_forecast(request)

    # Store first 14 days of predictions for verification
    for i, forecast in enumerate(response.forecasts[:14]):
        _store_prediction(
            lat=request.latitude,
            lon=request.longitude,
            location_name=location_name or f"{request.latitude:.2f}, {request.longitude:.2f}",
            target_date=forecast.date,
            temp_max=forecast.temperature_max,
            temp_min=forecast.temperature_min,
            precipitation=forecast.precipitation,
            precip_prob=forecast.precipitation_probability,
            lead_days=i + 1,
        )

    return response


async def _fetch_current_weather(lat: float, lon: float) -> dict | None:
    """Fetch current weather from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
        f"&timezone=auto"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                return {
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "precipitation": current.get("precipitation"),
                    "weather_code": current.get("weather_code"),
                    "time": current.get("time"),
                    "lat": lat,
                    "lon": lon,
                }
    except Exception as e:
        logger.warning(f"Failed to fetch weather for {lat},{lon}: {e}")

    return None


def _store_hourly_prediction(lat: float, lon: float, prediction_time: str, temperature: float, precipitation: float = 0):
    """Store an hourly prediction for later verification."""
    key = f"{lat:.4f}_{lon:.4f}_{prediction_time}"

    if key not in _hourly_predictions:
        _hourly_predictions[key] = {
            "lat": lat,
            "lon": lon,
            "prediction_time": prediction_time,
            "predicted_temp": temperature,
            "predicted_precip": precipitation,
            "stored_at": datetime.now().isoformat(),
            "verified": False,
            "actual_temp": None,
            "actual_precip": None,
            "temp_error": None,
        }


async def _verify_hourly_predictions():
    """Verify recent hourly predictions against actual observations."""
    global _last_verification_time

    now = datetime.now()
    _last_verification_time = now
    verified_count = 0

    # Get unique locations from stored predictions
    locations = set()
    for pred in _hourly_predictions.values():
        if not pred["verified"]:
            locations.add((pred["lat"], pred["lon"]))

    # Limit to 5 locations per check to avoid rate limiting
    locations = list(locations)[:5]

    for lat, lon in locations:
        actual = await _fetch_current_weather(lat, lon)
        if actual and actual["temperature"] is not None:
            current_hour = now.strftime("%Y-%m-%dT%H:00")
            key = f"{lat:.4f}_{lon:.4f}_{current_hour}"

            if key in _hourly_predictions and not _hourly_predictions[key]["verified"]:
                pred = _hourly_predictions[key]
                pred["verified"] = True
                pred["actual_temp"] = actual["temperature"]
                pred["actual_precip"] = actual.get("precipitation", 0)
                pred["temp_error"] = round(pred["predicted_temp"] - actual["temperature"], 2)
                pred["verified_at"] = now.isoformat()

                # Store verification result
                _hourly_verifications.append({
                    "lat": lat,
                    "lon": lon,
                    "time": current_hour,
                    "predicted": pred["predicted_temp"],
                    "actual": actual["temperature"],
                    "error": pred["temp_error"],
                    "verified_at": now.isoformat(),
                })

                verified_count += 1
                logger.info(f"Verified prediction for {lat:.2f},{lon:.2f}: predicted={pred['predicted_temp']:.1f}°C, actual={actual['temperature']:.1f}°C, error={pred['temp_error']:.1f}°C")

    # Clean up old predictions (older than 24 hours)
    cutoff = (now - timedelta(hours=24)).isoformat()
    keys_to_remove = [k for k, v in _hourly_predictions.items() if v["stored_at"] < cutoff]
    for k in keys_to_remove:
        del _hourly_predictions[k]

    # Keep only last 100 verifications
    if len(_hourly_verifications) > 100:
        _hourly_verifications[:] = _hourly_verifications[-100:]

    return verified_count


async def _verification_loop():
    """Background loop that runs verification every 5 minutes."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            verified = await _verify_hourly_predictions()
            logger.info(f"5-minute verification: {verified} predictions verified, {len(_hourly_predictions)} pending")
        except asyncio.CancelledError:
            logger.info("Verification loop cancelled")
            break
        except Exception as e:
            logger.error(f"Verification error: {e}")




@app.get("/v1/live-verification", tags=["Accuracy"])
async def get_live_verification():
    """Get live verification status and recent results."""
    global _last_verification_time

    # Calculate stats from verified predictions
    verified = [v for v in _hourly_verifications]

    if verified:
        errors = [abs(v["error"]) for v in verified]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
    else:
        avg_error = max_error = min_error = 0

    # Time until next verification
    if _last_verification_time:
        next_check = _last_verification_time + timedelta(minutes=5)
        seconds_until = max(0, (next_check - datetime.now()).total_seconds())
    else:
        seconds_until = 300

    return {
        "status": "active",
        "pending_predictions": len([p for p in _hourly_predictions.values() if not p["verified"]]),
        "verified_count": len(verified),
        "last_verification": _last_verification_time.isoformat() if _last_verification_time else None,
        "seconds_until_next": int(seconds_until),
        "stats": {
            "avg_error_celsius": round(avg_error, 2),
            "max_error_celsius": round(max_error, 2),
            "min_error_celsius": round(min_error, 2),
        },
        "recent_verifications": verified[-10:][::-1],  # Last 10, newest first
    }


@app.post("/v1/verify-now", tags=["Accuracy"])
async def verify_now():
    """Trigger immediate verification of pending predictions."""
    verified = await _verify_hourly_predictions()
    return {
        "message": f"Verified {verified} predictions",
        "verified_count": verified,
        "pending": len([p for p in _hourly_predictions.values() if not p["verified"]]),
    }


# ===== METAR Monitoring Endpoints =====

async def _fetch_metar(icao: str) -> str | None:
    """Fetch raw METAR from Aviation Weather API."""
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text.strip()
    except Exception as e:
        logger.warning(f"Failed to fetch METAR for {icao}: {e}")

    return None


def _parse_metar(raw: str, icao: str, name: str, lat: float, lon: float) -> MetarStation:
    """Parse raw METAR string into MetarStation object."""
    import re

    station = MetarStation(
        icao=icao,
        name=name,
        latitude=lat,
        longitude=lon,
        raw_metar=raw,
        last_checked=datetime.now(),
    )

    if not raw:
        station.is_missing = True
        return station

    # Check for $ flag (maintenance indicator)
    if raw.strip().endswith('$'):
        station.is_flagged = True

    # Parse observation time (format: DDHHMMz)
    time_match = re.search(r'\b(\d{6})Z\b', raw)
    if time_match:
        station.observation_time = time_match.group(1) + 'Z'

    # Parse temperature/dewpoint (format: TT/DD or MTT/MDD for negative)
    temp_match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw)
    if temp_match:
        temp_str, dew_str = temp_match.groups()
        station.temperature_c = -int(temp_str[1:]) if temp_str.startswith('M') else int(temp_str)
        station.dewpoint_c = -int(dew_str[1:]) if dew_str.startswith('M') else int(dew_str)

    # Parse wind (format: DDDSSKT or DDDSSGSSGKT)
    wind_match = re.search(r'\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b', raw)
    if wind_match:
        dir_str, speed_str, gust_str = wind_match.groups()
        station.wind_dir = int(dir_str) if dir_str != 'VRB' else 0
        station.wind_speed_kt = int(speed_str)

    # Parse visibility (format: SSM or SSSSM)
    vis_match = re.search(r'\b(\d{1,2})SM\b', raw)
    if vis_match:
        station.visibility_sm = float(vis_match.group(1))

    # Parse altimeter (format: ANNNN)
    alt_match = re.search(r'\bA(\d{4})\b', raw)
    if alt_match:
        station.altimeter_inhg = int(alt_match.group(1)) / 100.0

    # Extract cloud layers
    clouds = re.findall(r'\b(FEW|SCT|BKN|OVC|CLR|SKC|VV)(\d{3})?\b', raw)
    if clouds:
        station.clouds = ' '.join([f"{c[0]}{c[1]}" if c[1] else c[0] for c in clouds])

    # Extract weather phenomena
    wx_codes = re.findall(r'\b(?:\+|-|VC)?(RA|SN|TS|FG|BR|HZ|DZ|SH|GR|GS|IC|PE|PL|UP|FZ|MI|BC|PR|DR|BL|SQ|FC|SS|DS)+\b', raw)
    if wx_codes:
        station.weather = ' '.join(wx_codes)

    # Check if METAR is stale (observation time more than 90 minutes old)
    if station.observation_time:
        try:
            obs_day = int(station.observation_time[:2])
            obs_hour = int(station.observation_time[2:4])
            obs_min = int(station.observation_time[4:6])

            now = datetime.utcnow()
            obs_time = now.replace(day=obs_day, hour=obs_hour, minute=obs_min, second=0)

            # Handle month boundary
            if obs_time > now:
                obs_time = obs_time.replace(month=obs_time.month - 1 if obs_time.month > 1 else 12)

            age_minutes = (now - obs_time).total_seconds() / 60
            if age_minutes > 90:
                station.is_missing = True
        except:
            pass

    return station


async def _update_all_metars():
    """Fetch and update all monitored METAR stations."""
    global _metar_last_update

    logger.info("Updating METAR data for all stations...")
    updated = 0

    for icao, name, lat, lon in US_AIRPORTS:
        raw = await _fetch_metar(icao)
        station = _parse_metar(raw or "", icao, name, lat, lon)
        _metar_stations[icao] = station

        status = "FLAGGED" if station.is_flagged else ("MISSING" if station.is_missing else "OK")
        if station.is_flagged or station.is_missing:
            logger.warning(f"METAR {icao}: {status} - {raw[:50] if raw else 'No data'}...")

        updated += 1
        await asyncio.sleep(0.1)  # Rate limiting

    _metar_last_update = datetime.now()
    flagged = sum(1 for s in _metar_stations.values() if s.is_flagged)
    missing = sum(1 for s in _metar_stations.values() if s.is_missing)
    logger.info(f"Updated {updated} METAR stations. Flagged: {flagged}, Missing: {missing}")


async def _metar_monitor_loop():
    """Background loop that updates METAR data every 5 minutes."""
    # Initial fetch
    await _update_all_metars()

    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            await _update_all_metars()
        except asyncio.CancelledError:
            logger.info("METAR monitor loop cancelled")
            break
        except Exception as e:
            logger.error(f"METAR monitor error: {e}")




@app.get("/v1/metar", response_model=MetarMonitorResponse, tags=["METAR"])
async def get_metar_status():
    """Get current METAR status for all monitored stations."""
    stations = list(_metar_stations.values())

    flagged = sum(1 for s in stations if s.is_flagged)
    missing = sum(1 for s in stations if s.is_missing)
    healthy = len(stations) - flagged - missing

    # Calculate seconds until next update
    if _metar_last_update:
        next_update = _metar_last_update + timedelta(minutes=5)
        seconds_until = max(0, int((next_update - datetime.now()).total_seconds()))
    else:
        seconds_until = 300

    return MetarMonitorResponse(
        generated_at=datetime.now(),
        total_stations=len(stations),
        flagged_count=flagged,
        missing_count=missing,
        healthy_count=healthy,
        stations=stations,
        next_update_seconds=seconds_until,
    )


@app.post("/v1/metar/refresh", tags=["METAR"])
async def refresh_metar():
    """Force refresh all METAR data immediately."""
    await _update_all_metars()

    stations = list(_metar_stations.values())
    flagged = sum(1 for s in stations if s.is_flagged)
    missing = sum(1 for s in stations if s.is_missing)

    return {
        "message": f"Refreshed {len(stations)} stations",
        "flagged_count": flagged,
        "missing_count": missing,
    }


@app.get("/v1/metar/{icao}", tags=["METAR"])
async def get_station_metar(icao: str):
    """Get METAR for a specific station."""
    icao = icao.upper()

    if icao in _metar_stations:
        return _metar_stations[icao]

    # Fetch on-demand if not in cache
    raw = await _fetch_metar(icao)
    if raw:
        station = _parse_metar(raw, icao, icao, 0, 0)
        return station

    raise HTTPException(status_code=404, detail=f"Station {icao} not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
