"""
Simple forecaster for LILITH trained models.
Loads SimpleLILITH checkpoints and generates forecasts.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from loguru import logger

from models.simple_lilith import SimpleLILITH


class SimpleForecaster:
    """
    Simple forecaster that loads trained SimpleLILITH model.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to run on ("auto", "cuda", or "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract config and normalization
        self.config = self.checkpoint['config']
        self.norm = self.checkpoint['normalization']

        # Convert normalization to numpy arrays
        self.X_mean = np.array(self.norm['X_mean'])
        self.X_std = np.array(self.norm['X_std'])
        self.Y_mean = np.array(self.norm['Y_mean'])
        self.Y_std = np.array(self.norm['Y_std'])

        # Create model
        self.model = SimpleLILITH(
            input_features=self.config['input_features'],
            output_features=self.config['output_features'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            dropout=self.config.get('dropout', 0.1)
        )

        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Config: d_model={self.config['d_model']}, layers={self.config['num_encoder_layers']}")
        logger.info(f"Val RMSE: {self.checkpoint.get('val_rmse', 'N/A')}°C")

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using training stats."""
        return (x - self.X_mean) / (self.X_std + 1e-8)

    def _denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """Denormalize output to original scale."""
        return y * self.Y_std + self.Y_mean

    def _get_expected_climatology(self, latitude: float, longitude: float) -> Dict[str, float]:
        """
        Get expected climatological temperatures for a location and current season.
        This is used for bias correction of model outputs.
        """
        abs_lat = abs(latitude)
        day_of_year = datetime.now().timetuple().tm_yday

        # Base annual mean and seasonal amplitude by latitude
        if abs_lat < 15:  # Tropical
            annual_mean = 27
            seasonal_amp = 2
            diurnal_range = 10
        elif abs_lat < 28:  # Subtropical (Miami, Houston)
            annual_mean = 22
            seasonal_amp = 7
            diurnal_range = 11
        elif abs_lat < 35:  # Warm temperate (LA, Atlanta)
            annual_mean = 17
            seasonal_amp = 10
            diurnal_range = 12
        elif abs_lat < 42:  # Mid temperate (NYC, Chicago)
            annual_mean = 11
            seasonal_amp = 14
            diurnal_range = 10
        elif abs_lat < 48:  # Cool temperate (Minneapolis, Seattle)
            annual_mean = 7
            seasonal_amp = 17
            diurnal_range = 11
        elif abs_lat < 55:  # Cold
            annual_mean = 2
            seasonal_amp = 20
            diurnal_range = 10
        else:  # Subarctic/Arctic
            annual_mean = -5
            seasonal_amp = 22
            diurnal_range = 9

        # Regional adjustments for US (calibrated to January averages)
        if latitude > 24 and latitude < 32 and longitude > -88:  # Florida
            annual_mean += 2
            seasonal_amp *= 0.4
        elif latitude > 32 and latitude < 42 and longitude < -117:  # California coast
            annual_mean += 3
            seasonal_amp *= 0.35
            diurnal_range = 10
        elif latitude > 32 and latitude < 38 and -117 < longitude < -109:  # Desert SW (Phoenix)
            annual_mean += 2
            seasonal_amp *= 0.7
            diurnal_range = 13
        elif latitude > 45 and longitude < -122:  # Pacific NW
            seasonal_amp *= 0.5
            annual_mean += 3
        elif latitude > 28 and latitude < 32 and -98 < longitude < -93:  # Gulf Coast (Houston)
            annual_mean += 3
            seasonal_amp *= 0.6
        elif -110 < longitude < -100 and 35 < latitude < 45:  # Mountain West (Denver)
            seasonal_amp *= 0.9
            annual_mean += 2
        elif -100 < longitude < -85 and 40 < latitude < 50:  # Upper Midwest
            seasonal_amp *= 1.0  # Keep as is, already cold

        # Calculate current season temperature
        if latitude >= 0:
            phase_shift = 200  # Peak summer around day 200
        else:
            phase_shift = 15

        seasonal_offset = seasonal_amp * np.cos(2 * np.pi * (day_of_year - phase_shift) / 365)
        current_mean = annual_mean + seasonal_offset

        return {
            'tmax': current_mean + diurnal_range / 2,
            'tmin': current_mean - diurnal_range / 2,
            'mean': current_mean
        }

    def _create_synthetic_history(
        self,
        latitude: float,
        longitude: float,
        base_temp: float = None,
        days: int = 30
    ) -> np.ndarray:
        """
        Create synthetic historical data for a location.
        In production, this would fetch real observations.
        """
        # More realistic climate model based on latitude and season
        abs_lat = abs(latitude)

        # Base annual mean temperature by climate zone (calibrated to US cities)
        # These are approximate annual mean temperatures
        if abs_lat < 15:  # Tropical
            annual_mean = 27
            seasonal_amplitude = 2
        elif abs_lat < 28:  # Subtropical (Miami, Houston, Phoenix area)
            annual_mean = 22
            seasonal_amplitude = 8
        elif abs_lat < 35:  # Warm temperate (LA, Atlanta, Dallas)
            annual_mean = 17
            seasonal_amplitude = 11
        elif abs_lat < 42:  # Mid temperate (NYC, Chicago, Denver)
            annual_mean = 12
            seasonal_amplitude = 14
        elif abs_lat < 48:  # Cool temperate (Seattle, Minneapolis, Boston)
            annual_mean = 8
            seasonal_amplitude = 16
        elif abs_lat < 55:  # Cold (Northern US, Southern Canada)
            annual_mean = 4
            seasonal_amplitude = 18
        elif abs_lat < 66:  # Subarctic (Alaska)
            annual_mean = -2
            seasonal_amplitude = 22
        else:  # Arctic
            annual_mean = -15
            seasonal_amplitude = 20

        # Coastal moderation (reduces seasonal amplitude)
        # Florida peninsula, California coast, Pacific Northwest coast
        if latitude > 24 and latitude < 32 and longitude > -88:  # Florida
            seasonal_amplitude *= 0.6
            annual_mean += 3
        elif latitude > 32 and latitude < 42 and longitude < -117:  # California coast
            seasonal_amplitude *= 0.5
            annual_mean += 2
        elif latitude > 45 and longitude < -122:  # Pacific Northwest coast
            seasonal_amplitude *= 0.6

        # Continental interior (increases seasonal amplitude)
        if -105 < longitude < -85 and 35 < latitude < 50:  # Great Plains/Midwest
            seasonal_amplitude *= 1.2

        # Get day of year for seasonal calculation
        day_of_year = datetime.now().timetuple().tm_yday

        # Seasonal offset (peak warmth around day 200 in northern hemisphere)
        if latitude >= 0:  # Northern hemisphere
            phase_shift = 200
        else:  # Southern hemisphere
            phase_shift = 15

        seasonal_offset = seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - phase_shift) / 365)
        current_mean = annual_mean + seasonal_offset

        # Diurnal range (difference between high and low)
        diurnal_range = 10  # Typical 10°C difference

        # Generate synthetic daily data
        history = np.zeros((days, 3))
        np.random.seed(int(abs(latitude * 1000 + longitude * 1000)) % 2**31)

        for i in range(days):
            # Add day-to-day weather variability
            daily_var = np.random.randn() * 3
            daily_mean = current_mean + daily_var

            history[i, 0] = daily_mean + diurnal_range / 2  # TMAX
            history[i, 1] = daily_mean - diurnal_range / 2  # TMIN
            history[i, 2] = max(0, np.random.exponential(2))  # PRCP (mm)

        return history

    @torch.no_grad()
    def forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 14,
        history: np.ndarray = None,
        elevation: float = 0.0,
        ensemble_samples: int = 0,
        bias_correct_to_climatology: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate weather forecast for a location.

        Args:
            latitude, longitude: Location.
            forecast_days: Days to forecast (capped at the model's training horizon).
            history: Optional 30-day input array of shape (30, 3) — TMAX, TMIN, PRCP.
                     If None, falls back to the climatology-based synthetic history,
                     which is fine for smoke tests but should NEVER be used in
                     production: real station observations should be passed in.
            elevation: Station elevation in meters (used by the model's metadata
                       embedding). Defaults to 0.
            ensemble_samples: If > 0, run MC Dropout ``ensemble_samples`` times
                              and return mean + 5/95 percentiles. Cost grows
                              linearly with this argument.
            bias_correct_to_climatology: If True, blend a climatology prior into
                                         the model output (useful when training
                                         data was station-sparse). Default False
                                         — IMPORTANT: prior versions of this code
                                         silently overrode model predictions with
                                         climatology + noise, which made the RMSE
                                         numbers reported back to users
                                         meaningless. Now opt-in and labeled.

        Returns:
            Dict with model-derived forecast, optionally with uncertainty bands.
        """
        forecast_days = min(forecast_days, self.config.get("max_forecast", 90))

        if history is None:
            logger.warning(
                "No real history provided — falling back to synthetic climatology. "
                "Forecast quality will degrade because the model has never seen this "
                "exact input distribution. Pass real station observations in production."
            )
            history = self._create_synthetic_history(latitude, longitude)

        # Normalize input
        x_norm = self._normalize_input(history)

        # Metadata: [lat, lon, elevation, day_of_year] — same normalization as
        # train_simple.py applies (lat/90, lon/180, elev/5000, doy/365).
        day_of_year = datetime.now().timetuple().tm_yday / 365.0
        meta = np.array([
            latitude / 90.0,
            longitude / 180.0,
            elevation / 5000.0,
            day_of_year,
        ])

        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
        meta_tensor = torch.from_numpy(meta).float().unsqueeze(0).to(self.device)

        # Inference path: deterministic OR MC Dropout ensemble.
        if ensemble_samples > 0:
            samples = self.model.mc_dropout_forecast(
                x_tensor, meta_tensor, forecast_days, n_samples=ensemble_samples
            )  # (n_samples, 1, forecast_days, 3)
            samples_np = samples.squeeze(1).cpu().numpy()  # (n_samples, forecast_days, 3)
            pred_norm = samples_np.mean(axis=0)
            lower_norm = np.quantile(samples_np, 0.05, axis=0)
            upper_norm = np.quantile(samples_np, 0.95, axis=0)
        else:
            pred_norm = self.model(x_tensor, meta_tensor, forecast_days).cpu().numpy()[0]
            lower_norm = upper_norm = None

        # Denormalize using the SAME stats the model was trained with.
        pred = self._denormalize_output(pred_norm)
        lower = self._denormalize_output(lower_norm) if lower_norm is not None else None
        upper = self._denormalize_output(upper_norm) if upper_norm is not None else None

        # Optional climatology blend (off by default — see docstring).
        if bias_correct_to_climatology:
            expected = self._get_expected_climatology(latitude, longitude)
            blend = 0.3  # 70% model, 30% climatology
            pred[:, 0] = (1 - blend) * pred[:, 0] + blend * expected["tmax"]
            pred[:, 1] = (1 - blend) * pred[:, 1] + blend * expected["tmin"]

        start_date = datetime.now().date() + timedelta(days=1)
        forecasts = []
        for i in range(forecast_days):
            tmax = float(pred[i, 0])
            tmin = float(pred[i, 1])
            prcp = max(0.0, float(pred[i, 2]))

            # Sanity guard: temperature physics. Don't quietly hide it with
            # random noise like the old code did — emit it as-predicted and let
            # the caller see the problem if the model is broken.
            if tmin > tmax:
                tmax, tmin = tmin, tmax  # swap is the least-bad recovery

            day = {
                "date": start_date + timedelta(days=i),
                "day": i + 1,
                "temperature_high": round(tmax, 1),
                "temperature_low": round(tmin, 1),
                "precipitation_mm": round(prcp, 1),
                # Probability heuristic — model only predicts amount, not P(rain).
                "precipitation_probability": min(100, int(round(min(prcp, 10.0) * 10))),
            }
            day["date"] = day["date"].isoformat()
            if lower is not None and upper is not None:
                day["temperature_high_lower"] = round(float(lower[i, 0]), 1)
                day["temperature_high_upper"] = round(float(upper[i, 0]), 1)
                day["temperature_low_lower"] = round(float(lower[i, 1]), 1)
                day["temperature_low_upper"] = round(float(upper[i, 1]), 1)
            forecasts.append(day)

        return {
            "location": {"latitude": latitude, "longitude": longitude, "elevation": elevation},
            "generated_at": datetime.now().isoformat(),
            "model_version": "SimpleLILITH v1",
            "model_rmse": self.checkpoint.get("val_rmse", None),
            "forecast_days": forecast_days,
            "ensemble_samples": ensemble_samples,
            "bias_corrected": bias_correct_to_climatology,
            "forecasts": forecasts,
        }

    def forecast_hourly(
        self,
        latitude: float,
        longitude: float,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate hourly forecast by interpolating daily forecast.
        """
        # Get daily forecast
        daily = self.forecast(latitude, longitude, forecast_days=2)

        today = daily['forecasts'][0]
        tomorrow = daily['forecasts'][1] if len(daily['forecasts']) > 1 else today

        # Interpolate hourly
        hourly = []
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        for h in range(hours):
            hour_time = start_time + timedelta(hours=h)
            hour_of_day = hour_time.hour

            # Use today or tomorrow based on time
            if hour_time.date() == datetime.now().date():
                day_data = today
            else:
                day_data = tomorrow

            # Temperature curve: min at 6am, max at 3pm
            t_high = day_data['temperature_high']
            t_low = day_data['temperature_low']

            # Simple sinusoidal interpolation
            temp_range = t_high - t_low
            temp_mid = (t_high + t_low) / 2
            # Peak at 15:00 (3pm), trough at 6:00 (6am)
            phase = (hour_of_day - 15) * np.pi / 12
            temp = temp_mid + (temp_range / 2) * np.cos(phase)

            hourly.append({
                "time": hour_time.isoformat(),
                "hour": hour_of_day,
                "temperature": round(temp, 1),
                "precipitation_probability": day_data['precipitation_probability'],
            })

        return {
            "location": daily['location'],
            "generated_at": datetime.now().isoformat(),
            "hours": hours,
            "hourly": hourly
        }


# Global forecaster instance
_forecaster: Optional[SimpleForecaster] = None


def get_forecaster(checkpoint_path: str = None) -> SimpleForecaster:
    """Get or create forecaster instance."""
    global _forecaster

    if _forecaster is None:
        if checkpoint_path is None:
            # Default checkpoint path
            checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "lilith_best.pt"

        _forecaster = SimpleForecaster(str(checkpoint_path))

    return _forecaster


if __name__ == "__main__":
    # Test the forecaster
    forecaster = get_forecaster()

    # Test forecast for NYC
    result = forecaster.forecast(40.7128, -74.0060, forecast_days=14)

    print(f"\nForecast for NYC:")
    print(f"Model RMSE: {result['model_rmse']:.2f}°C")
    print(f"\nNext 14 days:")
    for f in result['forecasts']:
        print(f"  {f['date']}: High {f['temperature_high']}°C, Low {f['temperature_low']}°C, Precip {f['precipitation_mm']}mm")
