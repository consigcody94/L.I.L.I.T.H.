"""
LILITH API - Main FastAPI Application.

Provides REST API for weather forecasting:
- /v1/forecast - Single location forecast
- /v1/forecast/batch - Batch inference
- /v1/historical - Historical observations
"""

import time
import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from datetime import date, datetime, timedelta

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
)
from web.api.accuracy import aggregate_errors, prediction_errors
from inference.forecast_blend import baseline_forecast

# Global state for model
_forecaster = None
_config = None
_weather_service = None
_provider = None  # OpenMeteoProvider (keyless real data); set in lifespan

# In-memory prediction storage (would use database in production)
_predictions: dict[str, PredictionRecord] = {}
_prediction_counter = 0

# Hourly prediction tracking for 5-minute verification
_hourly_predictions: dict[str, dict] = {}  # key: lat_lon_datetime -> prediction data
_hourly_verifications: list[dict] = []  # List of verification results
_last_verification_time: Optional[datetime] = None
_verification_task = None  # Background task for 5-minute verification



# Load training stations from JSON (505 stations)
def _load_training_stations():
    """Load training station coordinates from JSON file."""
    import json
    stations_file = Path(__file__).parent.parent.parent / "data" / "training_stations.json"
    if stations_file.exists():
        with open(stations_file) as f:
            return json.load(f)
    # Fallback to a few major airports if file doesn't exist
    return [
        ("KJFK", "JFK Int'l", 40.6413, -73.7781),
        ("KLAX", "Los Angeles Int'l", 33.9416, -118.4085),
        ("KORD", "Chicago O'Hare", 41.9742, -87.9073),
    ]

TRAINING_STATIONS = _load_training_stations()


def get_forecaster():
    """Dependency to get forecaster instance."""
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _forecaster, _config, _verification_task, _weather_service, _provider
    from web.api.services.weather_service import WeatherService

    # Initialize Weather Service (OpenWeatherMap, optional - needs a key)
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    _weather_service = WeatherService(api_key=api_key)

    # Initialize the keyless real-data provider (Open-Meteo). No API key required.
    try:
        from data.providers.open_meteo import OpenMeteoProvider

        _provider = OpenMeteoProvider()
        logger.info("Open-Meteo provider ready (keyless real data)")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Open-Meteo provider unavailable: {exc}")
        _provider = None

    logger.info("Starting LILITH API...")

    # Load model checkpoint
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
def _add_uncertainty(daily: DailyForecast, lead_days: int) -> None:
    """Attach 95%-style temperature bounds that widen with lead time."""
    uncertainty = 2.0 + (lead_days / 14) * 2.0
    daily.temperature_max_lower = round(daily.temperature_max - uncertainty, 1)
    daily.temperature_max_upper = round(daily.temperature_max + uncertainty, 1)
    daily.temperature_min_lower = round(daily.temperature_min - uncertainty, 1)
    daily.temperature_min_upper = round(daily.temperature_min + uncertainty, 1)


def _daily_forecasts(forecasts: list, include_uncertainty: bool) -> list[DailyForecast]:
    """Convert forecaster/baseline day dicts into ``DailyForecast`` models."""
    out: list[DailyForecast] = []
    for f in forecasts:
        daily = DailyForecast(
            date=f["date"],
            temperature_max=f["temperature_high"],
            temperature_min=f["temperature_low"],
            precipitation=f["precipitation_mm"],
            precipitation_probability=min(1.0, max(0.0, f["precipitation_probability"] / 100.0)),
        )
        if include_uncertainty:
            _add_uncertainty(daily, f["day"])
        out.append(daily)
    return out


async def _model_forecast(request: ForecastRequest, location: Location) -> ForecastResponse:
    """Run the loaded model off the event loop and build a ForecastResponse."""
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    response = await asyncio.to_thread(
        _forecaster.forecast,
        latitude=location.latitude,
        longitude=location.longitude,
        forecast_days=request.days,
    )
    rmse = response.get("model_rmse")
    rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "N/A"
    result = ForecastResponse(
        location=location,
        generated_at=response["generated_at"],
        model_version=(
            f"SimpleLILITH v1 (RMSE: {rmse_str}°C, source: {response.get('data_source', 'model')})"
        ),
        forecast_days=response["forecast_days"],
        forecasts=_daily_forecasts(response["forecasts"], request.include_uncertainty),
    )
    _safe_store_predictions(location, response["generated_at"], response["forecasts"])
    return result


@app.post("/v1/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def create_forecast(request: ForecastRequest):
    """
    Generate a weather forecast for a single location.

    Returns up to 90 days of temperature and precipitation forecasts with optional
    uncertainty bounds. Falls back to real keyless data (Open-Meteo) when no
    trained model is loaded.
    """
    location = Location(latitude=request.latitude, longitude=request.longitude)

    if _forecaster is None:
        return await _generate_fallback_forecast(request)

    try:
        return await _model_forecast(request, location)
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
            forecast = await _generate_fallback_forecast(single_request)
        else:
            try:
                forecast = await _model_forecast(single_request, location)
            except Exception as e:
                logger.exception(f"Batch forecast error for {location}: {e}")
                forecast = await _generate_fallback_forecast(single_request)

        forecasts.append(forecast)

    processing_time = (time.time() - start_time) * 1000

    return BatchForecastResponse(
        forecasts=forecasts,
        total_locations=len(request.locations),
        processing_time_ms=processing_time,
    )


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
    # Auto-verify any unverified predictions that can be checked (off the loop).
    await asyncio.to_thread(_verify_predictions_with_actuals)

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
    verified_count = await asyncio.to_thread(_verify_predictions_with_actuals)
    return {"message": f"Verified {verified_count} predictions", "verified_count": verified_count}


# Station endpoints (backed by the real GHCN training-station list)
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _station_info(entry) -> StationInfo:
    station_id, name, lat, lon = entry
    return StationInfo(
        station_id=station_id,
        name=name,
        latitude=lat,
        longitude=lon,
        elevation=0.0,
        country="US",
    )


@app.get("/v1/stations", response_model=StationListResponse, tags=["Stations"])
async def list_stations(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    radius: float = Query(5.0, ge=0.0, le=180.0, description="Filter radius in degrees"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """
    List GHCN training stations, optionally filtered to those near a location.

    The model was trained on these 505 US stations; this exposes them for
    discovery and nearest-station lookup.
    """
    stations = list(TRAINING_STATIONS)

    if latitude is not None and longitude is not None:
        within = [
            s for s in stations if abs(s[2] - latitude) <= radius and abs(s[3] - longitude) <= radius
        ]
        within.sort(key=lambda s: _haversine_km(latitude, longitude, s[2], s[3]))
        stations = within

    total = len(stations)
    start = (page - 1) * page_size
    page_items = stations[start : start + page_size]

    return StationListResponse(
        stations=[_station_info(s) for s in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/v1/stations/{station_id}", response_model=StationInfo, tags=["Stations"])
async def get_station(station_id: str):
    """Get a single training station by GHCN ID."""
    for entry in TRAINING_STATIONS:
        if entry[0] == station_id:
            return _station_info(entry)
    raise HTTPException(status_code=404, detail=f"Station {station_id} not found")


async def _generate_fallback_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Real-data fallback forecast used when no trained model is loaded.

    Strategy (all real data, never fabricated):
      1. Open-Meteo (keyless): a real short-range forecast blended into a
         seasonal climatology to cover the full requested horizon.
      2. OpenWeatherMap (if an API key is configured): real 5-day forecast.
      3. Otherwise 503 - we never return random/mock numbers.
    """
    from datetime import datetime as _dt, timedelta as _td

    location = Location(latitude=request.latitude, longitude=request.longitude)

    # 1. Open-Meteo keyless real-data path (full horizon via climatology baseline).
    if _provider is not None:
        reference = await asyncio.to_thread(
            _provider.reference_forecast, request.latitude, request.longitude
        )
        if reference:
            start_date = _dt.now().date() + _td(days=1)
            day_dicts = baseline_forecast(
                request.latitude, request.longitude, start_date, request.days, reference
            )
            result = ForecastResponse(
                location=location,
                generated_at=_dt.now(),
                model_version="Open-Meteo + climatology (keyless, no model loaded)",
                forecast_days=len(day_dicts),
                forecasts=_daily_forecasts(day_dicts, request.include_uncertainty),
            )
            _safe_store_predictions(location, result.generated_at.isoformat(), day_dicts)
            return result

    # 2. OpenWeatherMap fallback (only if a key is configured).
    if _weather_service and _weather_service.api_key:
        owm_data = await _weather_service.get_forecast(request.latitude, request.longitude)
        if owm_data:
            daily_summaries: dict[str, dict] = {}
            for item in owm_data.get("list", []):
                date_str = item["dt_txt"].split(" ")[0]
                bucket = daily_summaries.setdefault(
                    date_str, {"temps": [], "precip": 0.0, "pop": []}
                )
                bucket["temps"].append(item["main"]["temp"])
                bucket["precip"] += item.get("rain", {}).get("3h", 0.0)
                bucket["pop"].append(item.get("pop", 0.0))

            forecasts = []
            for i, date_str in enumerate(sorted(daily_summaries)):
                if i >= request.days:
                    break
                data = daily_summaries[date_str]
                daily = DailyForecast(
                    date=date_str,
                    temperature_max=round(max(data["temps"]), 1),
                    temperature_min=round(min(data["temps"]), 1),
                    precipitation=round(data["precip"], 1),
                    precipitation_probability=round(max(data["pop"]) if data["pop"] else 0.0, 2),
                )
                if request.include_uncertainty:
                    _add_uncertainty(daily, i + 1)
                forecasts.append(daily)

            return ForecastResponse(
                location=location,
                generated_at=_dt.now(),
                model_version="OpenWeatherMap (fallback)",
                forecast_days=len(forecasts),
                forecasts=forecasts,
            )

    raise HTTPException(
        status_code=503,
        detail="No model loaded and no live weather data available.",
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
    elif abs_lat < 35:
        annual_mean = 24 - (abs_lat - 23) * 0.5
    elif abs_lat < 50:
        annual_mean = 18 - (abs_lat - 35) * 0.6
    elif abs_lat < 66:
        annual_mean = 9 - (abs_lat - 50) * 0.5
    else:
        annual_mean = -5 - (abs_lat - 66) * 0.4
    
    # Simple diurnal cycle
    forecasts = []
    
    # Generate 168 hours (7 days) or requested hours
    hours_to_generate = request.hours
    
    for i in range(hours_to_generate):
        forecast_time = now + datetime.timedelta(hours=i)
        
        # Diurnal cycle
        hour = forecast_time.hour
        # Peak temperature around 3 PM (15:00), lowest around 5 AM
        diurnal_cycle = 5 * math.cos((hour - 15) * math.pi / 12)
        
        # Seasonal trend (simplified)
        seasonal_trend = 0  # Ignore for short term
        
        temperature = annual_mean + diurnal_cycle
        
        # Add some random noise
        temperature += random.gauss(0, 1)
        
        # Precipitation chance
        precip_prob = max(0, min(100, 20 + 30 * math.sin(i / 24 * math.pi)))
        
        forecasts.append(HourlyForecast(
            datetime=forecast_time.isoformat(),
            hour=hour,
            temperature=round(temperature, 1),
            feels_like=round(temperature - 1, 1), # Wind chill / heat index?
            humidity=round(50 + 20 * math.sin((hour + 6) * math.pi / 12), 0),
            precipitation=0.0,
            precipitation_probability=round(precip_prob / 100.0, 2),
            wind_speed=round(max(0, 10 + 5 * math.sin(i/10)), 1),
            wind_direction=round((i * 10) % 360, 0),
            cloud_cover=round(max(0, min(100, 40 + 40 * math.sin(i / 12))), 0),
            pressure=1013.0,
            uv_index=round(max(0, 8 * math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0, 1),
        ))

    return HourlyForecastResponse(
        location=Location(latitude=lat, longitude=lon),
        generated_at=now,
        model_version="demo-v1 (hourly interpolated)",
        forecast_hours=hours_to_generate,
        forecasts=forecasts
    )


# Keep the prediction store bounded so a long-running server doesn't leak memory.
_MAX_STORED_PREDICTIONS = 5000
# How many lead days of each forecast to track for verification (Open-Meteo can
# supply real actuals once these dates pass).
_STORE_LEAD_DAYS = 7


def _safe_store_predictions(location: Location, generated_at: str, day_dicts: list) -> None:
    """Store the near-term forecast days for later verification (never raises)."""
    try:
        _store_predictions(location, generated_at, day_dicts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to store predictions for tracking: {exc}")


def _store_predictions(location: Location, generated_at: str, day_dicts: list) -> None:
    global _prediction_counter

    for f in day_dicts[:_STORE_LEAD_DAYS]:
        _prediction_counter += 1
        pred_id = f"PRED-{_prediction_counter:06d}"
        _predictions[pred_id] = PredictionRecord(
            id=pred_id,
            location=Location(latitude=location.latitude, longitude=location.longitude),
            predicted_at=generated_at,
            target_date=f["date"],
            predicted_temp_max=f["temperature_high"],
            predicted_temp_min=f["temperature_low"],
            predicted_precipitation=f["precipitation_mm"],
            predicted_precip_prob=min(1.0, max(0.0, f["precipitation_probability"] / 100.0)),
            lead_days=f["day"],
        )

    # Evict oldest records beyond the cap.
    if len(_predictions) > _MAX_STORED_PREDICTIONS:
        for key in list(_predictions)[: len(_predictions) - _MAX_STORED_PREDICTIONS]:
            _predictions.pop(key, None)


def _store_hourly_prediction(lat: float, lon: float, target_time: str, temp: float, precip: float):
    """Store an hourly prediction (used by the hourly endpoint)."""
    key = f"{lat:.4f}_{lon:.4f}_{target_time}"
    _hourly_predictions[key] = {
        "lat": lat,
        "lon": lon,
        "target_time": target_time,
        "temperature": temp,
        "precipitation": precip,
        "stored_at": datetime.now().isoformat(),
    }


async def _verification_loop():
    """Periodically verify stored predictions against real observations."""
    global _last_verification_time

    while True:
        try:
            count = await asyncio.to_thread(_verify_predictions_with_actuals)
            if count:
                logger.info(f"Verified {count} prediction(s) against real observations")
            _last_verification_time = datetime.now()
            await asyncio.sleep(300)  # every 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in verification loop: {e}")
            await asyncio.sleep(60)


def _verify_predictions_with_actuals() -> int:
    """
    Fill in real observed values for past predictions and compute real errors.

    Uses Open-Meteo (keyless). Predictions whose target date hasn't passed, or for
    which no observation is available, are left untouched - nothing is fabricated.
    """
    if _provider is None:
        return 0

    today = datetime.now().date()
    verified_count = 0

    for record in list(_predictions.values()):
        if record.actual_temp_max is not None:
            continue
        target = (
            date.fromisoformat(record.target_date)
            if isinstance(record.target_date, str)
            else record.target_date
        )
        if target >= today:
            continue

        obs = _provider.daily_observations(
            record.location.latitude, record.location.longitude, target
        )
        if not obs:
            continue

        record.actual_temp_max = round(obs["tmax"], 1)
        record.actual_temp_min = round(obs["tmin"], 1)
        record.actual_precipitation = round(obs["prcp"], 1)
        errs = prediction_errors(
            record.predicted_temp_max,
            record.predicted_temp_min,
            record.predicted_precipitation,
            obs["tmax"],
            obs["tmin"],
            obs["prcp"],
        )
        record.temp_max_error = errs["temp_max_error"]
        record.temp_min_error = errs["temp_min_error"]
        record.precip_error = errs["precip_error"]
        verified_count += 1

    return verified_count


def _generate_accuracy_report(lat, lon, days) -> AccuracyReportResponse:
    """Build a real accuracy report from verified predictions."""
    now = datetime.now()
    records = list(_predictions.values())

    location_filter = None
    if lat is not None and lon is not None:
        records = [
            r
            for r in records
            if abs(r.location.latitude - lat) < 0.5 and abs(r.location.longitude - lon) < 0.5
        ]
        location_filter = f"{lat:.4f},{lon:.4f}"

    verified = [r for r in records if r.actual_temp_max is not None]
    samples = [
        {
            "lead_days": r.lead_days,
            "temp_max_error": r.temp_max_error if r.temp_max_error is not None else 0.0,
            "temp_min_error": r.temp_min_error if r.temp_min_error is not None else 0.0,
            "predicted_precipitation": r.predicted_precipitation,
            "actual_precipitation": r.actual_precipitation
            if r.actual_precipitation is not None
            else 0.0,
        }
        for r in verified
    ]
    agg = aggregate_errors(samples)

    stats = AccuracyStats(
        total_predictions=len(records),
        verified_predictions=agg["verified_predictions"],
        temp_max_mae=agg["temp_max_mae"],
        temp_max_rmse=agg["temp_max_rmse"],
        temp_min_mae=agg["temp_min_mae"],
        temp_min_rmse=agg["temp_min_rmse"],
        precip_mae=agg["precip_mae"],
        precip_accuracy=agg["precip_accuracy"],
        accuracy_by_lead_day=agg["accuracy_by_lead_day"],
    )

    recent = sorted(records, key=lambda r: str(r.predicted_at), reverse=True)[:50]

    return AccuracyReportResponse(
        generated_at=now,
        period_start=(now - timedelta(days=days)).date().isoformat(),
        period_end=now.date().isoformat(),
        stats=stats,
        recent_predictions=recent,
        location_filter=location_filter,
    )
