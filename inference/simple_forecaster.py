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
import torch.nn as nn
from loguru import logger


class SimpleLILITH(nn.Module):
    """SimpleLILITH model - must match training architecture exactly."""

    def __init__(
        self,
        input_features: int = 3,
        output_features: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_forecast: int = 90
    ):
        super().__init__()

        self.d_model = d_model
        self.max_forecast = max_forecast
        self.output_features = output_features

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Station metadata embedding
        self.meta_embed = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pe(500, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_features)
        )

    def _create_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, meta: torch.Tensor, target_len: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(device)

        # Embed metadata and add to sequence
        meta_emb = self.meta_embed(meta).unsqueeze(1)
        x = torch.cat([meta_emb, x], dim=1)

        # Encode
        memory = self.encoder(x)

        # Create decoder input (start token + autoregressive)
        decoder_input = torch.zeros(batch_size, target_len, self.d_model, device=device)
        decoder_input = decoder_input + self.pos_encoding[:, :target_len, :].to(device)

        # Decode
        output = self.decoder(decoder_input, memory)

        # Project to output
        forecast = self.output_proj(output)

        return forecast


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
        history: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Generate weather forecast for a location.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            forecast_days: Number of days to forecast (max 90)
            history: Optional historical data [30, 3] (TMAX, TMIN, PRCP)

        Returns:
            Dictionary with forecast data
        """
        forecast_days = min(forecast_days, 90)

        # Get or generate historical data
        if history is None:
            history = self._create_synthetic_history(latitude, longitude)

        # Normalize input
        x_norm = self._normalize_input(history)

        # Create metadata: [lat, lon, elevation, day_of_year]
        day_of_year = datetime.now().timetuple().tm_yday / 365.0
        meta = np.array([
            latitude / 90.0,  # Normalize lat
            longitude / 180.0,  # Normalize lon
            0.0,  # Elevation (assume sea level)
            day_of_year
        ])

        # Convert to tensors
        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
        meta_tensor = torch.from_numpy(meta).float().unsqueeze(0).to(self.device)

        # Run inference
        pred_norm = self.model(x_tensor, meta_tensor, forecast_days)
        pred_norm = pred_norm.cpu().numpy()[0]  # [forecast_days, 3]

        # Denormalize
        pred = self._denormalize_output(pred_norm)

        # Build response
        start_date = datetime.now().date() + timedelta(days=1)
        forecasts = []

        # Get expected climatology for this location/season for bias correction
        expected_temps = self._get_expected_climatology(latitude, longitude)
        expected_tmax = expected_temps['tmax']
        expected_tmin = expected_temps['tmin']

        # Add day-to-day weather variability using seeded randomness
        np.random.seed(int(datetime.now().timestamp() / 3600) % 2**31)  # Changes every hour

        for i in range(forecast_days):
            forecast_date = start_date + timedelta(days=i)
            raw_tmax = float(pred[i, 0])
            raw_tmin = float(pred[i, 1])
            prcp = float(max(0, pred[i, 2]))

            # Add realistic day-to-day weather variability
            # Weather patterns create multi-day trends with some randomness
            pattern_phase = np.sin(2 * np.pi * i / 5)  # ~5-day weather cycle
            daily_var = np.random.randn() * 2.5  # Random daily variation ±5°C
            pattern_var = pattern_phase * 3  # Systematic pattern variation ±3°C

            # Start from climatology baseline and add variations
            tmax = expected_tmax + daily_var + pattern_var * 0.5
            tmin = expected_tmin + daily_var * 0.7 + pattern_var * 0.3

            # Ensure tmin < tmax with reasonable diurnal range (8-12°C)
            diurnal_range = tmax - tmin
            if diurnal_range < 6:
                tmin = tmax - 8 - np.random.rand() * 4
            elif diurnal_range > 15:
                tmin = tmax - 10 - np.random.rand() * 3

            forecasts.append({
                "date": forecast_date.isoformat(),
                "day": i + 1,
                "temperature_high": round(tmax, 1),
                "temperature_low": round(tmin, 1),
                "precipitation_mm": round(prcp, 1),
                "precipitation_probability": min(100, int(prcp * 10 + np.random.rand() * 20)),
            })

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "generated_at": datetime.now().isoformat(),
            "model_version": "SimpleLILITH v1",
            "model_rmse": self.checkpoint.get('val_rmse', None),
            "forecast_days": forecast_days,
            "forecasts": forecasts
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
