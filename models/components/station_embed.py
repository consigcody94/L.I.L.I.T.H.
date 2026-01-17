"""
Station Embedding Module for LILITH.

Learns dense representations of weather stations based on:
- Geographic coordinates (lat/lon/elevation)
- Historical observation patterns
- Station characteristics
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding3D(nn.Module):
    """
    3D positional encoding for geographic coordinates.

    Uses spherical harmonics-inspired encoding for lat/lon
    and linear encoding for elevation.
    """

    def __init__(self, d_model: int, max_freq: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

        # Frequencies for sinusoidal encoding
        freqs = torch.exp(
            torch.arange(0, max_freq) * (-math.log(10000.0) / max_freq)
        )
        self.register_buffer("freqs", freqs)

        # Projection to model dimension
        # 2 coords * 2 (sin/cos) * max_freq + elevation features
        input_dim = 4 * max_freq + 4
        self.proj = nn.Linear(input_dim, d_model)

    def forward(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        elev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode geographic coordinates.

        Args:
            lat: Latitude in degrees (-90, 90), shape (batch, n_stations)
            lon: Longitude in degrees (-180, 180), shape (batch, n_stations)
            elev: Elevation in meters, shape (batch, n_stations)

        Returns:
            Positional encoding of shape (batch, n_stations, d_model)
        """
        # Normalize coordinates
        lat_norm = lat / 90.0  # [-1, 1]
        lon_norm = lon / 180.0  # [-1, 1]

        # Convert to radians for spherical encoding
        lat_rad = lat_norm * (math.pi / 2)
        lon_rad = lon_norm * math.pi

        # Sinusoidal encoding for latitude
        lat_enc = torch.cat([
            torch.sin(lat_rad.unsqueeze(-1) * self.freqs),
            torch.cos(lat_rad.unsqueeze(-1) * self.freqs),
        ], dim=-1)

        # Sinusoidal encoding for longitude
        lon_enc = torch.cat([
            torch.sin(lon_rad.unsqueeze(-1) * self.freqs),
            torch.cos(lon_rad.unsqueeze(-1) * self.freqs),
        ], dim=-1)

        # Elevation encoding (normalized and log-scaled)
        elev_norm = torch.clamp(elev / 8848.0, -1, 1)  # Normalize by Everest height
        elev_log = torch.sign(elev) * torch.log1p(torch.abs(elev) / 100.0) / 5.0
        elev_enc = torch.stack([
            elev_norm,
            elev_log,
            torch.sin(elev_norm * math.pi),
            torch.cos(elev_norm * math.pi),
        ], dim=-1)

        # Concatenate all encodings
        encoding = torch.cat([lat_enc, lon_enc, elev_enc], dim=-1)

        return self.proj(encoding)


class StationEmbedding(nn.Module):
    """
    Embeds weather station observations into a dense vector space.

    Combines:
    1. Feature embedding (weather variables)
    2. Positional embedding (geographic location)
    3. Temporal embedding (time features)

    Architecture:
        Input features → LayerNorm → MLP → + Position Encoding → Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_position: bool = True,
    ):
        """
        Initialize station embedding module.

        Args:
            input_dim: Number of input weather features
            hidden_dim: Hidden dimension of MLP
            output_dim: Output embedding dimension
            n_layers: Number of MLP layers
            dropout: Dropout probability
            use_position: Whether to add positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_position = use_position

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Feature embedding MLP
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Remove last dropout
        self.feature_mlp = nn.Sequential(*layers[:-1])

        # Positional encoding
        if use_position:
            self.pos_encoding = PositionalEncoding3D(output_dim)

        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed station observations.

        Args:
            features: Weather features of shape (batch, n_stations, seq_len, n_features)
                     or (batch, n_stations, n_features) for single timestep
            coords: Station coordinates (lat, lon, elev) of shape (batch, n_stations, 3)
            mask: Valid observation mask of shape (batch, n_stations, seq_len)

        Returns:
            Embeddings of shape (batch, n_stations, seq_len, output_dim)
            or (batch, n_stations, output_dim) for single timestep
        """
        # Handle different input shapes
        single_timestep = features.dim() == 3
        if single_timestep:
            features = features.unsqueeze(2)  # Add seq_len dimension

        batch_size, n_stations, seq_len, n_features = features.shape

        # Reshape for MLP processing
        x = features.reshape(-1, n_features)

        # Apply mask if provided (zero out invalid observations)
        if mask is not None:
            mask_flat = mask.reshape(-1, 1).float()
            x = x * mask_flat

        # Normalize input
        x = self.input_norm(x)

        # Feature embedding
        x = self.feature_mlp(x)

        # Reshape back
        x = x.reshape(batch_size, n_stations, seq_len, self.output_dim)

        # Add positional encoding
        if self.use_position and coords is not None:
            lat = coords[:, :, 0]
            lon = coords[:, :, 1]
            elev = coords[:, :, 2]
            pos_enc = self.pos_encoding(lat, lon, elev)  # (batch, n_stations, output_dim)
            x = x + pos_enc.unsqueeze(2)  # Broadcast over seq_len

        # Output normalization
        x = self.output_norm(x)

        if single_timestep:
            x = x.squeeze(2)  # Remove seq_len dimension

        return x


class TemporalPositionEncoding(nn.Module):
    """
    Temporal position encoding using cyclical features.

    Encodes day-of-year, month, and other temporal patterns.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Projection from temporal features to model dimension
        # Features: day_sin, day_cos, month_sin, month_cos, year_normalized
        self.proj = nn.Linear(5, d_model)

    def forward(
        self,
        day_of_year: torch.Tensor,
        year: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode temporal position.

        Args:
            day_of_year: Day of year (1-366), shape (batch, seq_len)
            year: Year, shape (batch, seq_len)

        Returns:
            Temporal encoding of shape (batch, seq_len, d_model)
        """
        # Day of year (cyclical)
        day_rad = 2 * math.pi * day_of_year / 365.0
        day_sin = torch.sin(day_rad)
        day_cos = torch.cos(day_rad)

        # Month (cyclical) - approximate from day
        month_rad = 2 * math.pi * day_of_year / 30.0
        month_sin = torch.sin(month_rad)
        month_cos = torch.cos(month_rad)

        # Year normalized (for climate trends)
        if year is not None:
            year_norm = (year - 2000) / 50.0  # Center around 2000, scale by 50 years
        else:
            year_norm = torch.zeros_like(day_sin)

        # Combine features
        features = torch.stack([
            day_sin, day_cos, month_sin, month_cos, year_norm
        ], dim=-1)

        return self.proj(features)
