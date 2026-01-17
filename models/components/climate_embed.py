"""
Climate Embedding Module for LILITH.

Encodes climate indices and large-scale patterns that influence
long-range weather predictability:
- ENSO (El Nino Southern Oscillation)
- MJO (Madden-Julian Oscillation)
- NAO (North Atlantic Oscillation)
- Seasonal cycles
- Solar position
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn


class SeasonalEmbedding(nn.Module):
    """
    Encodes seasonal and cyclical time features.

    Uses sinusoidal encoding to capture annual, monthly, and daily cycles.
    """

    def __init__(self, d_model: int, max_harmonics: int = 4):
        """
        Initialize seasonal embedding.

        Args:
            d_model: Embedding dimension
            max_harmonics: Number of harmonic frequencies
        """
        super().__init__()

        self.d_model = d_model
        self.max_harmonics = max_harmonics

        # Number of raw features:
        # - day_of_year (sin/cos * harmonics)
        # - hour_of_day (sin/cos * harmonics) - if needed
        # - solar declination
        # - equation of time
        n_features = 4 * max_harmonics + 2

        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        day_of_year: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        year: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode time features.

        Args:
            day_of_year: Day of year (1-366), shape (batch, seq_len)
            hour: Hour of day (0-23), shape (batch, seq_len) - optional
            year: Year, shape (batch, seq_len) - optional

        Returns:
            Seasonal embedding of shape (batch, seq_len, d_model)
        """
        features = []

        # Day of year harmonics
        for k in range(1, self.max_harmonics + 1):
            freq = 2 * math.pi * k * day_of_year / 365.25
            features.extend([torch.sin(freq), torch.cos(freq)])

        # Hour of day harmonics (if provided)
        if hour is not None:
            for k in range(1, self.max_harmonics + 1):
                freq = 2 * math.pi * k * hour / 24.0
                features.extend([torch.sin(freq), torch.cos(freq)])
        else:
            # Pad with zeros
            features.extend([torch.zeros_like(day_of_year)] * (2 * self.max_harmonics))

        # Solar declination (approximate)
        # Maximum ~23.45 degrees on summer solstice
        declination = 23.45 * torch.sin(2 * math.pi * (day_of_year - 81) / 365.25)
        features.append(declination / 23.45)  # Normalize

        # Equation of time (minutes, approximate)
        B = 2 * math.pi * (day_of_year - 81) / 365.25
        eot = 9.87 * torch.sin(2 * B) - 7.53 * torch.cos(B) - 1.5 * torch.sin(B)
        features.append(eot / 15.0)  # Normalize by max (~15 minutes)

        # Stack features
        encoding = torch.stack(features, dim=-1)

        # Project to model dimension
        return self.norm(self.proj(encoding))


class ClimateIndexEmbedding(nn.Module):
    """
    Embeds climate indices (ENSO, NAO, MJO, etc.).

    These indices capture large-scale climate patterns that influence
    weather predictability on subseasonal to seasonal timescales.
    """

    def __init__(
        self,
        d_model: int,
        n_indices: int = 10,
        hidden_dim: int = 64,
    ):
        """
        Initialize climate index embedding.

        Args:
            d_model: Output embedding dimension
            n_indices: Maximum number of climate indices
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.d_model = d_model
        self.n_indices = n_indices

        # Index names and their typical value ranges
        self.index_names = [
            "nino34",      # ENSO: Nino 3.4 SST anomaly
            "nino12",      # ENSO: Nino 1+2 region
            "soi",         # Southern Oscillation Index
            "mjo_amp",     # MJO amplitude
            "mjo_phase",   # MJO phase (1-8, encoded as sin/cos)
            "nao",         # North Atlantic Oscillation
            "ao",          # Arctic Oscillation
            "pdo",         # Pacific Decadal Oscillation
            "amo",         # Atlantic Multidecadal Oscillation
            "qbo",         # Quasi-Biennial Oscillation
        ]

        # Embedding network
        self.mlp = nn.Sequential(
            nn.Linear(n_indices + 2, hidden_dim),  # +2 for MJO phase sin/cos
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        indices: torch.Tensor,
        mjo_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed climate indices.

        Args:
            indices: Climate index values of shape (batch, seq_len, n_indices)
                    Values should be normalized/standardized
            mjo_phase: MJO phase (1-8) of shape (batch, seq_len)

        Returns:
            Embedding of shape (batch, seq_len, d_model)
        """
        # Handle MJO phase specially (cyclical)
        if mjo_phase is not None:
            phase_rad = 2 * math.pi * mjo_phase / 8.0
            mjo_sin = torch.sin(phase_rad).unsqueeze(-1)
            mjo_cos = torch.cos(phase_rad).unsqueeze(-1)
            x = torch.cat([indices, mjo_sin, mjo_cos], dim=-1)
        else:
            # Pad if MJO phase not provided
            batch, seq_len, _ = indices.shape
            padding = torch.zeros(batch, seq_len, 2, device=indices.device)
            x = torch.cat([indices, padding], dim=-1)

        return self.mlp(x)


class SolarPositionEmbedding(nn.Module):
    """
    Encodes solar position for each location and time.

    Critical for capturing diurnal cycles and their geographic variation.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Features: solar altitude, azimuth, day length, sunrise/sunset
        n_features = 6
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        day_of_year: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute solar position features.

        Args:
            lat: Latitude in degrees, shape (batch, n_locations) or (batch,)
            lon: Longitude in degrees, shape (batch, n_locations) or (batch,)
            day_of_year: Day of year, shape (batch,) or (batch, seq_len)
            hour: Hour of day (0-24), shape (batch,) or (batch, seq_len)

        Returns:
            Solar position embedding
        """
        # Ensure proper broadcasting
        if lat.dim() == 1:
            lat = lat.unsqueeze(-1)
        if lon.dim() == 1:
            lon = lon.unsqueeze(-1)

        # Solar declination (degrees)
        declination = 23.45 * torch.sin(
            torch.deg2rad(torch.tensor(360 / 365.25 * (day_of_year - 81)))
        )

        # Convert to radians
        lat_rad = torch.deg2rad(lat)
        dec_rad = torch.deg2rad(declination)

        if dec_rad.dim() < lat_rad.dim():
            dec_rad = dec_rad.unsqueeze(-1)

        # Hour angle (if hour provided)
        if hour is not None:
            # Solar noon at longitude 0 is at 12:00 UTC
            # Each 15 degrees of longitude = 1 hour offset
            solar_time = hour + lon / 15.0
            hour_angle = torch.deg2rad((solar_time - 12.0) * 15.0)
        else:
            hour_angle = torch.zeros_like(lat_rad)

        # Solar altitude (elevation angle)
        sin_alt = (
            torch.sin(lat_rad) * torch.sin(dec_rad) +
            torch.cos(lat_rad) * torch.cos(dec_rad) * torch.cos(hour_angle)
        )
        solar_altitude = torch.arcsin(torch.clamp(sin_alt, -1, 1))

        # Solar azimuth
        cos_azimuth = (
            torch.sin(dec_rad) - torch.sin(lat_rad) * sin_alt
        ) / (torch.cos(lat_rad) * torch.cos(solar_altitude) + 1e-8)
        solar_azimuth = torch.arccos(torch.clamp(cos_azimuth, -1, 1))

        # Day length (hours)
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(dec_rad)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        day_length = 2 * torch.arccos(cos_hour_angle) / math.pi * 12.0

        # Normalize features
        features = torch.stack([
            torch.sin(solar_altitude),
            torch.cos(solar_altitude),
            torch.sin(solar_azimuth),
            torch.cos(solar_azimuth),
            day_length / 24.0,
            declination / 23.45,
        ], dim=-1)

        return self.norm(self.proj(features))


class ClimateEmbedding(nn.Module):
    """
    Combined climate embedding module.

    Integrates:
    1. Seasonal/cyclical time features
    2. Climate indices (ENSO, MJO, NAO, etc.)
    3. Solar position
    """

    def __init__(
        self,
        d_model: int,
        use_climate_indices: bool = True,
        use_solar_position: bool = True,
        max_harmonics: int = 4,
    ):
        """
        Initialize climate embedding.

        Args:
            d_model: Output dimension
            use_climate_indices: Include climate index embedding
            use_solar_position: Include solar position embedding
            max_harmonics: Number of harmonics for seasonal encoding
        """
        super().__init__()

        self.d_model = d_model
        self.use_climate_indices = use_climate_indices
        self.use_solar_position = use_solar_position

        # Component embeddings
        self.seasonal = SeasonalEmbedding(d_model, max_harmonics)

        if use_climate_indices:
            self.climate_indices = ClimateIndexEmbedding(d_model)
        else:
            self.climate_indices = None

        if use_solar_position:
            self.solar = SolarPositionEmbedding(d_model)
        else:
            self.solar = None

        # Fusion layer
        n_components = 1 + int(use_climate_indices) + int(use_solar_position)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_components, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        day_of_year: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
        climate_indices: Optional[torch.Tensor] = None,
        mjo_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute climate embedding.

        Args:
            day_of_year: Day of year (1-366)
            hour: Hour of day (0-23) - optional
            lat: Latitude in degrees - for solar position
            lon: Longitude in degrees - for solar position
            climate_indices: Climate index values - optional
            mjo_phase: MJO phase (1-8) - optional

        Returns:
            Combined climate embedding
        """
        embeddings = []

        # Seasonal embedding (always included)
        seasonal_emb = self.seasonal(day_of_year, hour)
        embeddings.append(seasonal_emb)

        # Climate indices
        if self.climate_indices is not None and climate_indices is not None:
            climate_emb = self.climate_indices(climate_indices, mjo_phase)
            embeddings.append(climate_emb)
        elif self.climate_indices is not None:
            # Use zeros if no indices provided
            shape = list(seasonal_emb.shape)
            embeddings.append(torch.zeros(shape, device=seasonal_emb.device))

        # Solar position
        if self.solar is not None and lat is not None and lon is not None:
            solar_emb = self.solar(lat, lon, day_of_year, hour)
            embeddings.append(solar_emb)
        elif self.solar is not None:
            shape = list(seasonal_emb.shape)
            embeddings.append(torch.zeros(shape, device=seasonal_emb.device))

        # Fuse embeddings
        combined = torch.cat(embeddings, dim=-1)
        return self.fusion(combined)
