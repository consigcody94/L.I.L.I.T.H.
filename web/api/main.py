"""
LILITH API - Main FastAPI Application.

Provides REST API for weather forecasting:
- /v1/forecast - Single location forecast
- /v1/forecast/batch - Batch inference
- /v1/stations - Station information
- /v1/historical - Historical observations
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
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
)

# Global state for model
_forecaster = None
_config = None


def get_forecaster():
    """Dependency to get forecaster instance."""
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _forecaster, _config

    logger.info("Starting LILITH API...")

    # Load configuration
    import os

    checkpoint_path = os.environ.get("LILITH_CHECKPOINT", None)
    encoder_path = os.environ.get("LILITH_ENCODER", None)
    stations_path = os.environ.get("LILITH_STATIONS", None)

    # Load model if checkpoint provided
    if checkpoint_path:
        try:
            from inference.forecast import Forecaster

            _forecaster = Forecaster.from_pretrained(
                checkpoint_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                encoder_path=encoder_path,
                stations_path=stations_path,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _forecaster = None
    else:
        logger.warning("No checkpoint provided. Running in demo mode.")
        _forecaster = None

    yield

    # Cleanup
    logger.info("Shutting down LILITH API...")
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
        gpu_available=torch.cuda.is_available(),
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
        # Return demo response
        return _generate_demo_forecast(request)

    try:
        response = _forecaster.forecast(
            latitude=request.latitude,
            longitude=request.longitude,
            forecast_days=request.days,
            include_uncertainty=request.include_uncertainty,
            ensemble_members=request.ensemble_members,
        )

        # Convert to Pydantic model
        return ForecastResponse(
            location=Location(latitude=request.latitude, longitude=request.longitude),
            generated_at=response.generated_at,
            model_version=response.model_version,
            forecast_days=response.forecast_days,
            forecasts=[
                DailyForecast(
                    date=f.date,
                    temperature_max=f.temperature_max,
                    temperature_min=f.temperature_min,
                    precipitation=f.precipitation,
                    precipitation_probability=f.precipitation_probability,
                    temperature_max_lower=f.temperature_max_lower,
                    temperature_max_upper=f.temperature_max_upper,
                    temperature_min_lower=f.temperature_min_lower,
                    temperature_min_upper=f.temperature_min_upper,
                )
                for f in response.forecasts
            ],
        )

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


def _generate_demo_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate a demo forecast when model is not loaded."""
    import datetime
    import math
    import random

    forecasts = []
    start_date = request.start_date or datetime.date.today()

    # Generate synthetic seasonal forecast
    for i in range(request.days):
        forecast_date = start_date + datetime.timedelta(days=i + 1)
        day_of_year = forecast_date.timetuple().tm_yday

        # Seasonal temperature curve
        seasonal = 15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

        # Base temperature based on latitude
        lat_effect = -0.5 * abs(request.latitude)

        base_temp = 15 + lat_effect + seasonal

        # Add some noise
        noise = random.gauss(0, 2)

        temp_max = base_temp + 5 + noise
        temp_min = base_temp - 5 + noise * 0.8

        # Precipitation (random)
        precip_prob = 0.3
        precipitation = random.expovariate(1.0) * 5 if random.random() < precip_prob else 0

        daily = DailyForecast(
            date=forecast_date.isoformat(),
            temperature_max=round(temp_max, 1),
            temperature_min=round(temp_min, 1),
            precipitation=round(precipitation, 1),
            precipitation_probability=round(precip_prob + random.gauss(0, 0.1), 2),
        )

        if request.include_uncertainty:
            # Add uncertainty bounds that widen with forecast lead time
            uncertainty_scale = 1 + (i / request.days) * 2
            daily.temperature_max_lower = round(temp_max - 2 * uncertainty_scale, 1)
            daily.temperature_max_upper = round(temp_max + 2 * uncertainty_scale, 1)
            daily.temperature_min_lower = round(temp_min - 2 * uncertainty_scale, 1)
            daily.temperature_min_upper = round(temp_min + 2 * uncertainty_scale, 1)

        forecasts.append(daily)

    return ForecastResponse(
        location=Location(latitude=request.latitude, longitude=request.longitude),
        generated_at=datetime.datetime.now(),
        model_version="demo-v1",
        forecast_days=request.days,
        forecasts=forecasts,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
