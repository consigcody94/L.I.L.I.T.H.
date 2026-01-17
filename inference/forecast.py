"""
Inference Engine for LILITH.

Provides high-level API for generating forecasts with:
- Automatic model loading and caching
- Batch inference
- Uncertainty estimation
- Post-processing and denormalization
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class ForecastRequest:
    """Request for weather forecast."""

    latitude: float
    longitude: float
    start_date: Optional[str] = None  # YYYY-MM-DD, defaults to today
    forecast_days: int = 90
    include_uncertainty: bool = True
    ensemble_members: int = 10
    variables: List[str] = None  # Defaults to ["TMAX", "TMIN", "PRCP"]

    def __post_init__(self):
        if self.variables is None:
            self.variables = ["TMAX", "TMIN", "PRCP"]


@dataclass
class DailyForecast:
    """Single day forecast."""

    date: str
    temperature_max: float
    temperature_min: float
    precipitation: float
    precipitation_probability: float

    # Uncertainty bounds (optional)
    temperature_max_lower: Optional[float] = None
    temperature_max_upper: Optional[float] = None
    temperature_min_lower: Optional[float] = None
    temperature_min_upper: Optional[float] = None


@dataclass
class ForecastResponse:
    """Complete forecast response."""

    location: Dict[str, float]
    generated_at: str
    model_version: str
    forecast_days: int
    forecasts: List[DailyForecast]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "forecast_days": self.forecast_days,
            "forecasts": [
                {
                    "date": f.date,
                    "temperature_max": f.temperature_max,
                    "temperature_min": f.temperature_min,
                    "precipitation": f.precipitation,
                    "precipitation_probability": f.precipitation_probability,
                    "temperature_max_lower": f.temperature_max_lower,
                    "temperature_max_upper": f.temperature_max_upper,
                    "temperature_min_lower": f.temperature_min_lower,
                    "temperature_min_upper": f.temperature_min_upper,
                }
                for f in self.forecasts
            ],
        }


class Forecaster:
    """
    High-level forecasting interface.

    Usage:
        forecaster = Forecaster.from_pretrained("path/to/checkpoint")
        response = forecaster.forecast(
            latitude=40.7128,
            longitude=-74.0060,
            forecast_days=90,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        encoder_path: Optional[str] = None,
        stations_path: Optional[str] = None,
    ):
        """
        Initialize forecaster.

        Args:
            model: LILITH model instance
            device: Device to run inference on
            encoder_path: Path to feature encoder (for denormalization)
            stations_path: Path to station metadata
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Load encoder for denormalization
        self.encoder = None
        if encoder_path and Path(encoder_path).exists():
            self._load_encoder(encoder_path)

        # Load station metadata for finding nearest stations
        self.stations = None
        if stations_path and Path(stations_path).exists():
            self.stations = pd.read_parquet(stations_path)
            logger.info(f"Loaded {len(self.stations)} stations")

        # Cache for recent observations (for conditioning)
        self._observation_cache: Dict[str, torch.Tensor] = {}

        logger.info(f"Forecaster initialized on {self.device}")

    def _load_encoder(self, path: str):
        """Load feature encoder for denormalization."""
        with open(path) as f:
            self.encoder_stats = json.load(f)
        logger.info(f"Loaded encoder from {path}")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        encoder_path: Optional[str] = None,
        stations_path: Optional[str] = None,
    ) -> "Forecaster":
        """
        Load forecaster from pretrained checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on
            encoder_path: Path to feature encoder
            stations_path: Path to station metadata

        Returns:
            Initialized Forecaster
        """
        from models.lilith import LILITH

        model = LILITH.from_pretrained(checkpoint_path, map_location=device)
        return cls(model, device, encoder_path, stations_path)

    def _find_nearby_stations(
        self,
        lat: float,
        lon: float,
        n_stations: int = 50,
        max_distance: float = 5.0,
    ) -> pd.DataFrame:
        """Find stations near a location."""
        if self.stations is None:
            raise ValueError("No station metadata loaded")

        # Calculate approximate distances
        dlat = self.stations["latitude"] - lat
        dlon = self.stations["longitude"] - lon
        distances = np.sqrt(dlat**2 + dlon**2)

        # Filter by distance
        mask = distances < max_distance
        nearby = self.stations[mask].copy()
        nearby["distance"] = distances[mask]

        # Sort by distance and take closest
        nearby = nearby.sort_values("distance").head(n_stations)

        return nearby

    def _get_recent_observations(
        self,
        station_ids: List[str],
        n_days: int = 30,
    ) -> torch.Tensor:
        """Get recent observations for conditioning."""
        # This would typically load from a database or cache
        # For now, return zeros (model should handle missing data)
        n_stations = len(station_ids)
        n_features = len(["TMAX", "TMIN", "PRCP"])

        return torch.zeros(1, n_stations, n_days, n_features)

    def _build_graph(
        self,
        stations: pd.DataFrame,
        radius: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph from station coordinates."""
        n_stations = len(stations)

        # Node coordinates
        node_coords = torch.tensor(
            stations[["latitude", "longitude", "elevation"]].values,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, n_stations, 3)

        # Build edges (connect stations within radius)
        lats = stations["latitude"].values
        lons = stations["longitude"].values

        edges_src = []
        edges_dst = []
        edge_weights = []

        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                dist = np.sqrt((lats[i] - lats[j])**2 + (lons[i] - lons[j])**2)
                if dist < radius:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                    edge_weights.extend([dist, dist])

        # Handle case with no edges
        if not edges_src:
            # Connect to k nearest neighbors
            for i in range(n_stations):
                dists = np.sqrt((lats - lats[i])**2 + (lons - lons[i])**2)
                neighbors = np.argsort(dists)[1:6]  # 5 nearest
                for j in neighbors:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(dists[j])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

        return node_coords, edge_index, edge_attr

    def _denormalize(
        self,
        predictions: np.ndarray,
        variable: str,
    ) -> np.ndarray:
        """Denormalize predictions to original scale."""
        if self.encoder_stats is None or variable not in self.encoder_stats:
            return predictions

        stats = self.encoder_stats[variable]

        if "prcp" in variable.lower():
            # Reverse log1p transform
            return np.expm1(predictions)
        else:
            # Reverse standard normalization
            return predictions * stats["std"] + stats["mean"]

    @torch.no_grad()
    def forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 90,
        include_uncertainty: bool = True,
        ensemble_members: int = 10,
    ) -> ForecastResponse:
        """
        Generate weather forecast for a location.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            forecast_days: Number of days to forecast
            include_uncertainty: Include uncertainty bounds
            ensemble_members: Number of ensemble members for uncertainty

        Returns:
            ForecastResponse with daily forecasts
        """
        import datetime

        # Find nearby stations
        if self.stations is not None:
            nearby_stations = self._find_nearby_stations(latitude, longitude)
            station_ids = nearby_stations["station_id"].tolist()
        else:
            # Use synthetic station at target location
            nearby_stations = pd.DataFrame({
                "station_id": ["target"],
                "latitude": [latitude],
                "longitude": [longitude],
                "elevation": [0.0],
            })
            station_ids = ["target"]

        # Get recent observations
        observations = self._get_recent_observations(station_ids)
        observations = observations.to(self.device)

        # Build graph
        node_coords, edge_index, edge_attr = self._build_graph(nearby_stations)
        node_coords = node_coords.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        # Run inference
        if include_uncertainty:
            # Generate ensemble
            ensemble = self.model.generate_ensemble(
                node_features=observations,
                node_coords=node_coords,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_members=ensemble_members,
            )

            # Get mean and bounds
            predictions = ensemble.mean(dim=0).cpu().numpy()
            lower_bound = ensemble.quantile(0.025, dim=0).cpu().numpy()
            upper_bound = ensemble.quantile(0.975, dim=0).cpu().numpy()
        else:
            outputs = self.model(
                node_features=observations,
                node_coords=node_coords,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            predictions = outputs["forecast"].cpu().numpy()
            lower_bound = None
            upper_bound = None

        # Find closest station to target location
        if len(nearby_stations) > 1:
            closest_idx = 0  # Assuming sorted by distance
        else:
            closest_idx = 0

        # Extract predictions for closest station
        preds = predictions[0, closest_idx, :forecast_days, :]  # (forecast_days, n_vars)

        # Denormalize
        tmax = self._denormalize(preds[:, 0], "TMAX")
        tmin = self._denormalize(preds[:, 1], "TMIN")
        prcp = self._denormalize(preds[:, 2], "PRCP")

        # Calculate precipitation probability (simple threshold)
        prcp_prob = np.clip(prcp / 10.0, 0, 1)  # Rough estimate

        # Build daily forecasts
        start_date = datetime.date.today()
        forecasts = []

        for i in range(forecast_days):
            forecast_date = start_date + datetime.timedelta(days=i + 1)

            daily = DailyForecast(
                date=forecast_date.isoformat(),
                temperature_max=float(tmax[i]),
                temperature_min=float(tmin[i]),
                precipitation=float(max(0, prcp[i])),
                precipitation_probability=float(prcp_prob[i]),
            )

            # Add uncertainty bounds if available
            if include_uncertainty and lower_bound is not None:
                daily.temperature_max_lower = float(self._denormalize(lower_bound[0, closest_idx, i, 0], "TMAX"))
                daily.temperature_max_upper = float(self._denormalize(upper_bound[0, closest_idx, i, 0], "TMAX"))
                daily.temperature_min_lower = float(self._denormalize(lower_bound[0, closest_idx, i, 1], "TMIN"))
                daily.temperature_min_upper = float(self._denormalize(upper_bound[0, closest_idx, i, 1], "TMIN"))

            forecasts.append(daily)

        return ForecastResponse(
            location={"latitude": latitude, "longitude": longitude},
            generated_at=datetime.datetime.now().isoformat(),
            model_version=getattr(self.model.config, "variant", "unknown"),
            forecast_days=forecast_days,
            forecasts=forecasts,
        )

    @torch.no_grad()
    def forecast_batch(
        self,
        requests: List[ForecastRequest],
    ) -> List[ForecastResponse]:
        """
        Generate forecasts for multiple locations.

        Args:
            requests: List of forecast requests

        Returns:
            List of forecast responses
        """
        # For now, process sequentially
        # TODO: Batch inference for efficiency
        responses = []
        for request in requests:
            response = self.forecast(
                latitude=request.latitude,
                longitude=request.longitude,
                forecast_days=request.forecast_days,
                include_uncertainty=request.include_uncertainty,
                ensemble_members=request.ensemble_members,
            )
            responses.append(response)

        return responses
