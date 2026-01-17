"""
LILITH - Long-range Intelligent Learning for Integrated Trend Hindcasting

Main model implementation combining:
- Station embedding for irregular observation data
- Graph Attention Network for spatial relationships
- Temporal Transformer for sequence modeling
- Spherical Fourier Neural Operator for global dynamics
- Climate embeddings for long-range predictability
- Ensemble head for uncertainty quantification
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.station_embed import StationEmbedding, TemporalPositionEncoding
from models.components.gat_encoder import GATEncoder
from models.components.temporal_transformer import TemporalTransformer
from models.components.sfno import SphericalFourierNeuralOperator
from models.components.climate_embed import ClimateEmbedding
from models.components.ensemble_head import EnsembleHead


@dataclass
class LILITHConfig:
    """Configuration for LILITH model."""

    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    ffn_dim: Optional[int] = None  # Defaults to 4 * hidden_dim

    # Input/Output
    input_features: int = 7  # TMAX, TMIN, PRCP, etc.
    output_features: int = 3  # TMAX, TMIN, PRCP
    sequence_length: int = 30  # Input days
    forecast_length: int = 90  # Output days

    # Component depths
    gat_layers: int = 3
    temporal_layers: int = 6
    sfno_layers: int = 4

    # Grid configuration
    use_grid: bool = True
    nlat: int = 64
    nlon: int = 128

    # Features
    use_climate_embed: bool = True
    use_solar_position: bool = True
    use_flash_attention: bool = True
    use_rope: bool = True

    # Ensemble
    ensemble_method: str = "gaussian"  # "gaussian", "quantile", "mc_dropout", "diffusion"
    ensemble_members: int = 10

    # Regularization
    dropout: float = 0.1

    # Memory optimization
    gradient_checkpointing: bool = False

    # Variant presets
    variant: str = "base"  # "tiny", "base", "large", "xl"

    def __post_init__(self):
        """Apply variant presets."""
        if self.variant == "tiny":
            self.hidden_dim = 128
            self.num_heads = 4
            self.gat_layers = 2
            self.temporal_layers = 4
            self.sfno_layers = 2
        elif self.variant == "large":
            self.hidden_dim = 384
            self.num_heads = 12
            self.gat_layers = 4
            self.temporal_layers = 8
            self.sfno_layers = 6
        elif self.variant == "xl":
            self.hidden_dim = 512
            self.num_heads = 16
            self.gat_layers = 4
            self.temporal_layers = 12
            self.sfno_layers = 8

        if self.ffn_dim is None:
            self.ffn_dim = self.hidden_dim * 4


class StationToGrid(nn.Module):
    """
    Converts sparse station observations to a regular grid.

    Uses learned interpolation weights based on station locations.
    """

    def __init__(
        self,
        hidden_dim: int,
        nlat: int = 64,
        nlon: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.nlat = nlat
        self.nlon = nlon

        # Grid coordinates
        lat_grid = torch.linspace(-90, 90, nlat)
        lon_grid = torch.linspace(-180, 180, nlon)
        lat_mesh, lon_mesh = torch.meshgrid(lat_grid, lon_grid, indexing="ij")

        self.register_buffer("lat_grid", lat_mesh.flatten())
        self.register_buffer("lon_grid", lon_mesh.flatten())

        # Learned distance weighting
        self.distance_mlp = nn.Sequential(
            nn.Linear(3, 32),  # dist, dlat, dlon
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        station_features: torch.Tensor,
        station_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate station features to grid.

        Args:
            station_features: (batch, n_stations, hidden_dim)
            station_coords: (batch, n_stations, 3) - lat, lon, elev
            mask: (batch, n_stations) - valid stations

        Returns:
            Grid features: (batch, hidden_dim, nlat, nlon)
        """
        batch_size, n_stations, _ = station_features.shape
        n_grid = self.nlat * self.nlon

        # Get station locations
        station_lat = station_coords[:, :, 0]  # (batch, n_stations)
        station_lon = station_coords[:, :, 1]

        # Compute distances from each station to each grid point
        # Using approximate great-circle distance
        dlat = station_lat.unsqueeze(2) - self.lat_grid.unsqueeze(0).unsqueeze(0)
        dlon = station_lon.unsqueeze(2) - self.lon_grid.unsqueeze(0).unsqueeze(0)

        # Approximate distance in degrees
        dist = torch.sqrt(dlat**2 + dlon**2 + 1e-6)

        # Stack distance features
        dist_features = torch.stack([dist, dlat, dlon], dim=-1)

        # Compute attention weights
        weights = self.distance_mlp(dist_features).squeeze(-1)  # (batch, n_stations, n_grid)

        # Apply mask if provided
        if mask is not None:
            weights = weights * mask.unsqueeze(-1)

        # Normalize weights (softmax over stations)
        weights = F.softmax(weights, dim=1)

        # Weighted sum of station features
        grid_features = torch.einsum("bsg,bsd->bgd", weights, station_features)

        # Reshape to grid
        grid_features = grid_features.view(batch_size, self.nlat, self.nlon, self.hidden_dim)
        grid_features = grid_features.permute(0, 3, 1, 2)  # (batch, hidden, nlat, nlon)

        return grid_features


class GridToStation(nn.Module):
    """
    Samples grid features at station locations.

    Uses bilinear interpolation with learned refinement.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        grid_features: torch.Tensor,
        station_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample grid at station locations.

        Args:
            grid_features: (batch, hidden_dim, nlat, nlon)
            station_coords: (batch, n_stations, 3) - lat, lon, elev

        Returns:
            Station features: (batch, n_stations, hidden_dim)
        """
        batch_size, _, nlat, nlon = grid_features.shape
        n_stations = station_coords.size(1)

        # Normalize coordinates to [-1, 1] for grid_sample
        lat = station_coords[:, :, 0]
        lon = station_coords[:, :, 1]

        # Normalize lat from [-90, 90] to [-1, 1]
        lat_norm = lat / 90.0
        # Normalize lon from [-180, 180] to [-1, 1]
        lon_norm = lon / 180.0

        # Create sampling grid (grid_sample expects (x, y) format)
        sample_grid = torch.stack([lon_norm, lat_norm], dim=-1)  # (batch, n_stations, 2)
        sample_grid = sample_grid.unsqueeze(1)  # (batch, 1, n_stations, 2)

        # Sample grid
        sampled = F.grid_sample(
            grid_features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (batch, hidden_dim, 1, n_stations)

        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (batch, n_stations, hidden_dim)

        # Refine
        return self.refine(sampled)


class ForecastDecoder(nn.Module):
    """
    Decodes latent representation to multi-day forecasts.

    Uses autoregressive rollout with multi-timescale strategy.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        forecast_length: int = 90,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.forecast_length = forecast_length

        # Step predictor (one step at a time)
        self.step_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Temporal position encoding for forecast steps
        self.pos_encoding = nn.Parameter(
            torch.randn(1, forecast_length, hidden_dim) * 0.02
        )

    def forward(
        self,
        latent: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate multi-step forecast.

        Args:
            latent: Encoded representation (batch, hidden_dim) or (batch, n_stations, hidden_dim)
            n_steps: Number of forecast steps (default: forecast_length)

        Returns:
            Forecasts of shape (batch, n_steps, output_dim) or (batch, n_stations, n_steps, output_dim)
        """
        n_steps = n_steps or self.forecast_length

        # Handle both station-wise and global latent
        if latent.dim() == 2:
            batch_size = latent.size(0)
            latent = latent.unsqueeze(1)  # (batch, 1, hidden_dim)
            squeeze_output = True
        else:
            batch_size, n_stations, _ = latent.shape
            squeeze_output = False

        # Add positional encoding for forecast steps
        pos_enc = self.pos_encoding[:, :n_steps, :]  # (1, n_steps, hidden_dim)

        # Initialize output
        outputs = []

        # Autoregressive rollout
        state = latent
        for t in range(n_steps):
            # Add positional info
            step_input = state + pos_enc[:, t:t+1, :]

            # Predict next state
            state = state + self.step_predictor(step_input)

            # Project to output
            output = self.output_proj(state)
            outputs.append(output)

        # Stack outputs
        forecasts = torch.cat(outputs, dim=-2)  # (batch, n_stations, n_steps, output_dim)

        if squeeze_output:
            forecasts = forecasts.squeeze(1)

        return forecasts


class LILITH(nn.Module):
    """
    LILITH: Long-range Intelligent Learning for Integrated Trend Hindcasting

    A neural weather prediction model that combines:
    1. Station embedding for sparse observation data
    2. Graph neural network for spatial relationships
    3. Temporal transformer for sequence modeling
    4. Spherical Fourier operator for global dynamics
    5. Climate embeddings for long-range predictability
    6. Ensemble generation for uncertainty

    Architecture:
        Stations → Embed → GAT → Grid → SFNO → Temporal → Decode → Forecast
    """

    def __init__(self, config: Optional[LILITHConfig] = None):
        """
        Initialize LILITH model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config or LILITHConfig()
        cfg = self.config

        # Station embedding
        self.station_embed = StationEmbedding(
            input_dim=cfg.input_features,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_position=True,
        )

        # Graph encoder for spatial relationships
        self.gat_encoder = GATEncoder(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.gat_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            edge_dim=1,  # Edge distances
        )

        # Station to grid conversion
        if cfg.use_grid:
            self.station_to_grid = StationToGrid(cfg.hidden_dim, cfg.nlat, cfg.nlon)
            self.grid_to_station = GridToStation(cfg.hidden_dim)

            # Spherical Fourier operator for global dynamics
            self.sfno = SphericalFourierNeuralOperator(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.hidden_dim,
                num_layers=cfg.sfno_layers,
                nlat=cfg.nlat,
                nlon=cfg.nlon,
                dropout=cfg.dropout,
            )
        else:
            self.station_to_grid = None
            self.grid_to_station = None
            self.sfno = None

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.temporal_layers,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            use_flash=cfg.use_flash_attention,
            use_rope=cfg.use_rope,
        )

        # Climate embedding
        if cfg.use_climate_embed:
            self.climate_embed = ClimateEmbedding(
                d_model=cfg.hidden_dim,
                use_climate_indices=True,
                use_solar_position=cfg.use_solar_position,
            )
        else:
            self.climate_embed = None

        # Forecast decoder
        self.decoder = ForecastDecoder(
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_features,
            forecast_length=cfg.forecast_length,
            dropout=cfg.dropout,
        )

        # Ensemble head for uncertainty
        self.ensemble_head = EnsembleHead(
            input_dim=cfg.hidden_dim,
            output_dim=cfg.output_features,
            hidden_dim=cfg.hidden_dim,
            method=cfg.ensemble_method,
        )

        # Apply gradient checkpointing if configured
        if cfg.gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.temporal_transformer.enable_gradient_checkpointing()
        if self.sfno is not None:
            self.sfno.enable_gradient_checkpointing()

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        day_of_year: Optional[torch.Tensor] = None,
        climate_indices: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-station forecasting.

        Args:
            node_features: Station observations (batch, n_stations, seq_len, n_features)
            node_coords: Station coordinates (batch, n_stations, 3) - lat, lon, elev
            edge_index: Graph connectivity (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            mask: Valid observation mask (batch, n_stations, seq_len)
            day_of_year: Day of year for climate embedding (batch, seq_len)
            climate_indices: Climate indices (batch, seq_len, n_indices)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dict with keys:
            - "forecast": (batch, n_stations, forecast_len, output_features)
            - "uncertainty_lower": optional lower bound
            - "uncertainty_upper": optional upper bound
        """
        batch_size, n_stations, seq_len, n_features = node_features.shape

        # 1. Station embedding
        # Reshape for embedding: (batch * n_stations, seq_len, n_features)
        x = node_features.view(batch_size * n_stations, seq_len, n_features)
        coords_flat = node_coords.view(batch_size * n_stations, 3)

        embedded = self.station_embed(
            x.view(batch_size * n_stations * seq_len, n_features),
            None,  # coords handled separately
        )
        embedded = embedded.view(batch_size, n_stations, seq_len, -1)

        # 2. Temporal processing per station
        # Process each station's sequence
        embedded_flat = embedded.view(batch_size * n_stations, seq_len, -1)
        temporal_out = self.temporal_transformer(embedded_flat)
        temporal_out = temporal_out.view(batch_size, n_stations, seq_len, -1)

        # Take last timestep as state
        station_state = temporal_out[:, :, -1, :]  # (batch, n_stations, hidden_dim)

        # 3. Graph processing (spatial relationships)
        # Process all nodes together
        station_state_flat = station_state.view(-1, self.config.hidden_dim)

        # Adjust edge_index for batch
        # Assuming edge_index is already batched or we process graphs separately
        graph_out = self.gat_encoder(
            station_state_flat,
            edge_index,
            edge_attr,
        )
        graph_out = graph_out.view(batch_size, n_stations, -1)

        # 4. Grid processing (global dynamics)
        if self.sfno is not None:
            # Convert stations to grid
            grid = self.station_to_grid(graph_out, node_coords)

            # Apply SFNO for global processing
            grid = self.sfno(grid)

            # Sample back to stations
            global_context = self.grid_to_station(grid, node_coords)

            # Combine local and global
            combined = graph_out + global_context
        else:
            combined = graph_out

        # 5. Add climate embedding
        if self.climate_embed is not None and day_of_year is not None:
            climate_emb = self.climate_embed(
                day_of_year[:, -1],  # Last day
                lat=node_coords[:, :, 0].mean(dim=1),  # Mean lat
                lon=node_coords[:, :, 1].mean(dim=1),  # Mean lon
                climate_indices=climate_indices[:, -1] if climate_indices is not None else None,
            )
            combined = combined + climate_emb.unsqueeze(1)

        # 6. Decode to forecast
        forecast = self.decoder(combined)  # (batch, n_stations, forecast_len, output_features)

        result = {"forecast": forecast}

        # 7. Uncertainty estimation
        if return_uncertainty:
            # Use ensemble head for uncertainty
            combined_flat = combined.view(-1, self.config.hidden_dim)
            mean, lower, upper = self.ensemble_head.predict_with_uncertainty(combined_flat)

            # Reshape
            mean = mean.view(batch_size, n_stations, -1)
            lower = lower.view(batch_size, n_stations, -1)
            upper = upper.view(batch_size, n_stations, -1)

            result["uncertainty_lower"] = lower
            result["uncertainty_upper"] = upper

        return result

    def generate_ensemble(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        n_members: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate ensemble forecast.

        Args:
            node_features: Input observations
            node_coords: Station coordinates
            edge_index: Graph connectivity
            edge_attr: Edge features
            n_members: Number of ensemble members
            **kwargs: Additional arguments passed to forward

        Returns:
            Ensemble forecasts of shape (n_members, batch, n_stations, forecast_len, output_features)
        """
        # Get deterministic forecast first
        with torch.no_grad():
            result = self.forward(
                node_features, node_coords, edge_index, edge_attr, **kwargs
            )

        # Generate ensemble using ensemble head
        # This is a simplified version; full implementation would use
        # the ensemble head throughout the forward pass

        ensemble = [result["forecast"]]

        # For now, add noise-based perturbations
        # A more sophisticated approach would use the diffusion ensemble head
        base_forecast = result["forecast"]
        for _ in range(n_members - 1):
            # Add calibrated noise
            noise = torch.randn_like(base_forecast) * 0.1
            ensemble.append(base_forecast + noise)

        return torch.stack(ensemble, dim=0)

    @torch.no_grad()
    def predict(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction with uncertainty.

        Args:
            node_features: Input observations
            node_coords: Station coordinates
            edge_index: Graph connectivity
            edge_attr: Edge features
            **kwargs: Additional arguments

        Returns:
            Dict with forecast and uncertainty bounds
        """
        self.eval()
        return self.forward(
            node_features,
            node_coords,
            edge_index,
            edge_attr,
            return_uncertainty=True,
            **kwargs,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, "station_embed"):
            n_params -= sum(p.numel() for p in self.station_embed.parameters())
        return n_params

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "LILITH":
        """Load a pretrained model."""
        checkpoint = torch.load(path, map_location=map_location)
        config = LILITHConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save model and config."""
        torch.save({
            "config": self.config.__dict__,
            "model_state_dict": self.state_dict(),
        }, path)
