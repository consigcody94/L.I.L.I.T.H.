"""
Simple forecaster for LILITH trained models.

Loads a SimpleLILITH checkpoint and turns it into a calibrated daily forecast.

The model's prediction is the core signal: its day-to-day structure is preserved
and bias-corrected against a seasonally-aware climatology (see
``inference.forecast_blend``), rather than being discarded. When real data is
available (Open-Meteo, keyless) it is used to (a) seed the model with genuine
recent observations and (b) blend a real short-range forecast into the near term,
where numerical weather prediction outperforms any subseasonal model. Output is
deterministic - no random perturbations.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from loguru import logger

from inference.forecast_blend import blend_forecast, synthetic_history
from models.simple_lilith import SimpleLILITH

try:
    from data.providers.open_meteo import OpenMeteoProvider
except Exception:  # pragma: no cover - provider is optional
    OpenMeteoProvider = None  # type: ignore[assignment]


class SimpleForecaster:
    """Load a trained SimpleLILITH model and serve calibrated forecasts."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        enable_live_data: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
            device: "auto", "cuda" or "cpu".
            enable_live_data: Use Open-Meteo (keyless) for real recent observations
                and near-term reference forecasts. Falls back gracefully when off
                or unreachable.
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = self._load_checkpoint(checkpoint_path)

        # Extract config and normalization
        self.config = self.checkpoint["config"]
        self.norm = self.checkpoint["normalization"]
        self.input_days = int(self.config.get("input_days", 30))
        self.max_forecast = int(self.config.get("max_forecast", 90))

        # Convert normalization to numpy arrays
        self.X_mean = np.array(self.norm["X_mean"])
        self.X_std = np.array(self.norm["X_std"])
        self.Y_mean = np.array(self.norm["Y_mean"])
        self.Y_std = np.array(self.norm["Y_std"])

        # Create model
        self.model = SimpleLILITH(
            input_features=self.config["input_features"],
            output_features=self.config["output_features"],
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_encoder_layers=self.config["num_encoder_layers"],
            num_decoder_layers=self.config["num_decoder_layers"],
            dropout=self.config.get("dropout", 0.1),
        )

        # Load weights
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Optional keyless real-data provider
        self.provider = None
        if enable_live_data and OpenMeteoProvider is not None:
            try:
                self.provider = OpenMeteoProvider()
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Could not initialise Open-Meteo provider: {exc}")

        logger.info(f"Model loaded on {self.device}")
        logger.info(
            f"Config: d_model={self.config['d_model']}, layers={self.config['num_encoder_layers']}"
        )
        logger.info(f"Val RMSE: {self.checkpoint.get('val_rmse', 'N/A')}°C")
        logger.info(f"Live data: {'on' if self.provider else 'off'}")

    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint, preferring the safe ``weights_only=True`` path.

        ``weights_only=True`` avoids unpickling arbitrary objects. Our own
        checkpoints additionally store a few NumPy scalars (e.g. ``val_rmse``), so
        those types are allow-listed first. As a last resort - and only for a
        trusted, locally-produced checkpoint - it falls back to a full load.
        """
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            try:
                try:
                    from numpy._core import multiarray as _ma  # numpy >= 2
                except Exception:
                    from numpy.core import multiarray as _ma  # numpy < 2

                torch.serialization.add_safe_globals([_ma.scalar, np.dtype, np.ndarray])
                return torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
            except Exception as exc:
                logger.warning(
                    "Safe checkpoint load failed (%s); falling back to full load for "
                    "trusted local checkpoint %s",
                    exc,
                    checkpoint_path,
                )
                return torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using training stats."""
        return (x - self.X_mean) / (self.X_std + 1e-8)

    def _denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """Denormalize output to original scale."""
        return y * self.Y_std + self.Y_mean

    @torch.no_grad()
    def forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 14,
        history: Optional[np.ndarray] = None,
        use_live_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a calibrated weather forecast for a location.

        Args:
            latitude, longitude: Location.
            forecast_days: Number of days to forecast (clamped to the model max).
            history: Optional pre-supplied recent history [N, 3] (TMAX, TMIN, PRCP).
                When omitted, real observations are fetched (if live data is on),
                otherwise a deterministic synthetic history is used.
            use_live_data: Allow Open-Meteo lookups for this call.

        Returns:
            Dict with location, metadata and a list of per-day forecasts.
        """
        forecast_days = max(1, min(forecast_days, self.max_forecast))
        provider = self.provider if use_live_data else None
        data_source = "model"

        # 1. Recent history (model input): real observations preferred.
        if history is None and provider is not None:
            history = provider.recent_history(latitude, longitude, days=self.input_days)
            if history is not None:
                data_source = "model+observations"
        if history is None:
            history = synthetic_history(latitude, longitude, days=self.input_days)

        # 2. Run the model.
        x_norm = self._normalize_input(np.asarray(history, dtype=np.float64))
        day_of_year = datetime.now().timetuple().tm_yday / 365.0
        meta = np.array(
            [latitude / 90.0, longitude / 180.0, 0.0, day_of_year], dtype=np.float64
        )

        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
        meta_tensor = torch.from_numpy(meta).float().unsqueeze(0).to(self.device)

        pred_norm = self.model(x_tensor, meta_tensor, forecast_days).cpu().numpy()[0]
        pred = self._denormalize_output(pred_norm)  # [forecast_days, 3]

        # 3. Real near-term reference forecast for blending.
        reference = None
        if provider is not None:
            reference = provider.reference_forecast(latitude, longitude)
            if reference:
                data_source = (
                    "reference+model+observations"
                    if "observations" in data_source
                    else "reference+model"
                )

        # 4. Assemble the calibrated forecast (deterministic).
        start_date = datetime.now().date() + timedelta(days=1)
        forecasts = blend_forecast(pred, latitude, longitude, start_date, reference=reference)

        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "generated_at": datetime.now().isoformat(),
            "model_version": "SimpleLILITH v1",
            "model_rmse": self.checkpoint.get("val_rmse", None),
            "data_source": data_source,
            "forecast_days": forecast_days,
            "forecasts": forecasts,
        }

    def forecast_hourly(
        self,
        latitude: float,
        longitude: float,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Generate an hourly forecast by interpolating the daily forecast."""
        days_needed = max(2, (hours // 24) + 2)
        daily = self.forecast(latitude, longitude, forecast_days=days_needed)
        by_date = {f["date"]: f for f in daily["forecasts"]}

        hourly = []
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        for h in range(hours):
            hour_time = start_time + timedelta(hours=h)
            hour_of_day = hour_time.hour
            day_key = hour_time.date().isoformat()
            # Use the matching day if forecast, else fall back to the first day.
            day_data = by_date.get(day_key, daily["forecasts"][0])

            t_high = day_data["temperature_high"]
            t_low = day_data["temperature_low"]
            temp_mid = (t_high + t_low) / 2
            temp_range = t_high - t_low
            # Peak at 15:00, trough near 03:00.
            phase = (hour_of_day - 15) * np.pi / 12
            temp = temp_mid + (temp_range / 2) * np.cos(phase)

            hourly.append(
                {
                    "time": hour_time.isoformat(),
                    "hour": hour_of_day,
                    "temperature": round(float(temp), 1),
                    "precipitation_probability": day_data["precipitation_probability"],
                }
            )

        return {
            "location": daily["location"],
            "generated_at": datetime.now().isoformat(),
            "hours": hours,
            "hourly": hourly,
        }


# Global forecaster instance
_forecaster: Optional[SimpleForecaster] = None


def get_forecaster(checkpoint_path: Optional[str] = None) -> SimpleForecaster:
    """Get or create the singleton forecaster instance."""
    global _forecaster

    if _forecaster is None:
        if checkpoint_path is None:
            checkpoint_path = str(
                Path(__file__).parent.parent / "checkpoints" / "lilith_best.pt"
            )
        _forecaster = SimpleForecaster(str(checkpoint_path))

    return _forecaster


if __name__ == "__main__":
    forecaster = get_forecaster()
    result = forecaster.forecast(40.7128, -74.0060, forecast_days=14)

    rmse = result.get("model_rmse")
    rmse_str = f"{rmse:.2f}°C" if isinstance(rmse, (int, float)) else "N/A"
    print("\nForecast for NYC:")
    print(f"Model RMSE: {rmse_str}")
    print(f"Data source: {result['data_source']}")
    print("\nNext 14 days:")
    for f in result["forecasts"]:
        print(
            f"  {f['date']}: High {f['temperature_high']}°C, "
            f"Low {f['temperature_low']}°C, Precip {f['precipitation_mm']}mm "
            f"({f['source']})"
        )
