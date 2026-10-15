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
from datetime import datetime, timezone
import hashlib

# Make torch optional for demo mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Query
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
    AccuracyReportResponse,
)
from web.api.storage.ledger import ForecastLedger
from inference.forecast_blend import baseline_forecast

# Global state for model
_forecaster = None
_config = None
_weather_service = None
_provider = None  # OpenMeteoProvider (keyless real data); set in lifespan

# Durable daily forecast archive (single-host pilot)
_ledger: Optional[ForecastLedger] = None
_checkpoint_sha256: Optional[str] = None

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
    global _ledger, _checkpoint_sha256
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
    _ledger = ForecastLedger(os.environ.get("LILITH_LEDGER_PATH", "data/storage/forecast-ledger.sqlite3"))
    _checkpoint_sha256 = None

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
            with open(checkpoint_path, "rb") as checkpoint_file:
                _checkpoint_sha256 = hashlib.sha256(checkpoint_file.read()).hexdigest()
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
            _checkpoint_sha256 = None
    else:
        logger.warning("No checkpoint provided. Serving labeled reference fallback products.")
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
    _ledger = None


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
    """Attach heuristic, uncalibrated temperature ranges that widen with lead time."""
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
            source=f.get("source"),
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
    result.checkpoint_sha256 = _checkpoint_sha256
    await _archive_response(result)
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
    except HTTPException:
        raise
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


@app.get("/v1/forecast/runs/{run_id}", response_model=ForecastResponse, tags=["Forecast"])
async def replay_forecast(run_id: str):
    """Replay the original issuance without regenerating or changing its values."""
    if _ledger is None:
        raise HTTPException(status_code=503, detail="Forecast archive unavailable")
    payload = await asyncio.to_thread(_ledger.replay, run_id)
    if payload is None or "forecast_days" not in payload:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return payload


# Historical data endpoints
@app.post("/v1/historical", response_model=HistoricalResponse, tags=["Historical"])
async def get_historical_data(request: HistoricalRequest):
    """
    Get historical observations for a station.

    Returns daily observations for the specified date range.
    """
    from data.providers.ghcn_records import historical_station
    try:
        return await asyncio.to_thread(
            historical_station, request.station_id, request.start_date, request.end_date,
            Path(os.environ.get("LILITH_GHCN_CACHE", "data/raw/ghcn_records")), request.variables,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="GHCN station not found") from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="NOAA station data unavailable") from exc


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
    if _provider is None:
        raise HTTPException(status_code=503, detail="Hourly reference provider unavailable")
    rows = await asyncio.to_thread(_provider.hourly_forecast, request.latitude, request.longitude, request.hours)
    if not rows:
        raise HTTPException(status_code=503, detail="No complete hourly reference forecast available")
    result = HourlyForecastResponse(
        location=Location(latitude=request.latitude, longitude=request.longitude),
        generated_at=datetime.now(timezone.utc),
        model_version="Open-Meteo NWP reference (not Lilith hourly predictions)",
        forecast_hours=len(rows), forecasts=[HourlyForecast(**row) for row in rows],
    )
    await _archive_response(result)
    return result


@app.get("/v1/forecast/hourly/runs/{run_id}", response_model=HourlyForecastResponse, tags=["Forecast"])
async def replay_hourly_forecast(run_id: str):
    if _ledger is None:
        raise HTTPException(status_code=503, detail="Forecast archive unavailable")
    payload = await asyncio.to_thread(_ledger.replay, run_id)
    if payload is None or "forecast_hours" not in payload:
        raise HTTPException(status_code=404, detail="Hourly run not found")
    return payload


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
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=422, detail="Supply both latitude and longitude")
    # Reading a report never starts remote provider work.
    return await asyncio.to_thread(_generate_accuracy_report, latitude, longitude, days_back)


@app.get("/v1/accuracy/predictions", response_model=list[PredictionRecord], tags=["Accuracy"])
async def get_predictions(
    limit: int = Query(50, ge=1, le=200),
    verified_only: bool = Query(False),
):
    """
    Get recent prediction records.

    Returns stored predictions with their actual observations (if available).
    """
    if _ledger is None:
        raise HTTPException(status_code=503, detail="Forecast archive unavailable")
    return await asyncio.to_thread(_ledger.records, limit=limit, verified_only=verified_only)


@app.post("/v1/accuracy/verify", tags=["Accuracy"])
async def verify_predictions():
    """
    Verify past predictions against actual observations.

    This endpoint fetches actual weather data and updates prediction
    records with observed values and error calculations.
    """
    # Mutation is operator-only through the CLI, avoiding unbounded provider calls
    # from a public endpoint. The background worker also processes bounded batches.
    raise HTTPException(status_code=405, detail="Use the operator verification CLI")


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
                model_version="Open-Meteo + heuristic seasonal baseline (no model loaded)",
                forecast_days=len(day_dicts),
                forecasts=_daily_forecasts(day_dicts, request.include_uncertainty),
            )
            await _archive_response(result)
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

            result = ForecastResponse(
                location=location,
                generated_at=_dt.now(),
                model_version="OpenWeatherMap (fallback)",
                forecast_days=len(forecasts),
                forecasts=forecasts,
            )
            await _archive_response(result)
            return result

    raise HTTPException(
        status_code=503,
        detail="No model loaded and no live weather data available.",
    )


async def _archive_response(result: ForecastResponse | HourlyForecastResponse) -> None:
    """A returned daily forecast must have a durable, replayable issuance."""
    if _ledger is None:
        raise HTTPException(status_code=503, detail="Forecast archive unavailable")
    try:
        writer = _ledger.archive_hourly if isinstance(result, HourlyForecastResponse) else _ledger.archive
        result.run_id = await asyncio.to_thread(writer, result.model_dump(mode="json"))
    except Exception as exc:
        logger.exception("Forecast archive write failed")
        raise HTTPException(status_code=503, detail="Could not archive forecast") from exc


async def _verification_loop():
    """Bounded modeled-reference comparisons, separate from station verification."""
    global _last_verification_time
    while True:
        try:
            count = await asyncio.to_thread(_verify_predictions_with_actuals)
            if count:
                logger.info(f"Compared {count} predictions with modeled reference data")
            _last_verification_time = datetime.now(timezone.utc)
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Reference comparison delayed: {exc}")
            await asyncio.sleep(60)


def _verify_predictions_with_actuals() -> int:
    if _ledger is None or _provider is None:
        return 0
    from web.api.verification import compare_pending
    return compare_pending(_ledger, _provider)


def _generate_accuracy_report(lat, lon, days) -> AccuracyReportResponse:
    if _ledger is None:
        raise HTTPException(status_code=503, detail="Forecast archive unavailable")
    return AccuracyReportResponse(**_ledger.report(lat=lat, lon=lon, days=days))
