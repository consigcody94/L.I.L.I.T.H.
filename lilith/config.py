"""
Configuration management for LILITH.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/storage/parquet"
    tensor_dir: str = "data/storage/zarr"

    # GHCN settings
    ghcn_daily_url: str = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
    ghcn_hourly_url: str = "https://www.ncei.noaa.gov/data/global-hourly/access/"

    # Processing
    min_years: int = 30  # Minimum years of data for a station
    max_gap_days: int = 7  # Maximum gap to interpolate
    grid_resolution: float = 0.25  # Degrees

    # Variables
    variables: list[str] = field(
        default_factory=lambda: ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    )


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    name: str = "lilith-base"
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1

    # Station encoder
    station_embed_dim: int = 64
    max_stations: int = 10000
    geo_hash_resolution: int = 6

    # Graph attention
    gat_heads: int = 4
    gat_layers: int = 2

    # SFNO processor
    sfno_modes: int = 32
    sfno_layers: int = 4

    # Temporal transformer
    max_seq_len: int = 365
    temporal_heads: int = 8

    # Climate embeddings
    use_climate_embed: bool = True
    climate_embed_dim: int = 64

    # Output
    forecast_days: int = 90
    output_variables: list[str] = field(
        default_factory=lambda: ["temp_mean", "temp_max", "temp_min", "precip", "wind"]
    )


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # cosine, linear, constant
    min_lr_ratio: float = 0.1

    # Curriculum learning
    curriculum: bool = True
    curriculum_stages: list[int] = field(default_factory=lambda: [7, 14, 42, 90])

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    keep_last_n: int = 5

    # Logging
    log_every: int = 100
    eval_every: int = 500
    use_wandb: bool = True
    wandb_project: str = "lilith"


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Model
    checkpoint_path: str = "checkpoints/best.pt"
    quantization: Optional[str] = None  # None, int8, int4

    # Serving
    batch_size: int = 16
    max_concurrent: int = 10
    timeout_seconds: float = 30.0

    # Caching
    use_cache: bool = True
    cache_ttl_seconds: int = 21600  # 6 hours
    redis_url: str = "redis://localhost:6379"


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # General
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Use OmegaConf for merging with defaults
        schema = OmegaConf.structured(cls)
        loaded = OmegaConf.create(raw)
        merged = OmegaConf.merge(schema, loaded)

        return OmegaConf.to_object(merged)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        cfg = OmegaConf.structured(self)
        with open(path, "w") as f:
            OmegaConf.save(cfg, f)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure directories exist
        Path(self.data.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.tensor_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
