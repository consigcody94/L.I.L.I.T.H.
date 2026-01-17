"""
Data Processing Pipeline

Orchestrates the full data processing workflow:
1. Load raw GHCN data
2. Apply quality control
3. Normalize and encode features
4. Grid data (station → regular grid)
5. Save to efficient formats (Parquet/Zarr)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from data.download.ghcn_daily import GHCNDailyDownloader
from data.processing.quality_control import QualityController


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    # Input/Output
    raw_dir: str = "data/raw/ghcn_daily"
    output_dir: str = "data/storage/parquet"
    tensor_dir: str = "data/storage/zarr"

    # Processing
    min_years: int = 30
    min_observations_per_year: int = 300
    target_variables: list[str] = None

    # Normalization
    normalize: bool = True
    clip_outliers: bool = True
    outlier_std: float = 5.0

    # Gridding
    grid_resolution: float = 0.25  # degrees
    interpolation_method: str = "idw"  # 'idw', 'kriging', 'nearest'
    max_interpolation_distance: float = 2.0  # degrees

    def __post_init__(self):
        if self.target_variables is None:
            self.target_variables = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]


class FeatureEncoder:
    """
    Encodes and normalizes weather features for ML training.

    Handles:
    - Cyclical encoding for time features (day of year, hour)
    - Log transformation for precipitation
    - Standard normalization for temperatures
    - Sin/cos encoding for wind direction
    """

    def __init__(self):
        self.stats: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Compute normalization statistics from data."""
        for col in df.select_dtypes(include=[np.number]).columns:
            self.stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding and normalization."""
        result = df.copy()

        # Add time features
        if isinstance(df.index, pd.DatetimeIndex):
            # Day of year (cyclical)
            day_of_year = df.index.dayofyear
            result["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
            result["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

            # Month (cyclical)
            month = df.index.month
            result["month_sin"] = np.sin(2 * np.pi * month / 12)
            result["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Normalize numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self.stats:
                stats = self.stats[col]

                # Special handling for precipitation (log transform)
                if "prcp" in col.lower() or "precip" in col.lower():
                    # Log1p transform for precipitation
                    result[col] = np.log1p(df[col].clip(lower=0))
                else:
                    # Standard normalization
                    if stats["std"] > 0:
                        result[col] = (df[col] - stats["mean"]) / stats["std"]
                    else:
                        result[col] = 0.0

        # Wind direction encoding (if present)
        for col in ["wind_direction", "WDIR"]:
            if col in df.columns:
                rad = np.deg2rad(df[col])
                result[f"{col}_sin"] = np.sin(rad)
                result[f"{col}_cos"] = np.cos(rad)
                result = result.drop(columns=[col])

        return result

    def inverse_transform(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Reverse normalization for predictions."""
        result = df.copy()
        columns = columns or list(self.stats.keys())

        for col in columns:
            if col not in self.stats or col not in df.columns:
                continue

            stats = self.stats[col]

            if "prcp" in col.lower() or "precip" in col.lower():
                # Reverse log1p
                result[col] = np.expm1(df[col])
            else:
                # Reverse standard normalization
                result[col] = df[col] * stats["std"] + stats["mean"]

        return result

    def save(self, path: str) -> None:
        """Save encoder statistics to file."""
        import json

        with open(path, "w") as f:
            json.dump(self.stats, f)

    @classmethod
    def load(cls, path: str) -> "FeatureEncoder":
        """Load encoder from file."""
        import json

        encoder = cls()
        with open(path) as f:
            encoder.stats = json.load(f)
        return encoder


class SpatialGridder:
    """
    Converts irregular station data to regular lat/lon grid.

    Uses inverse distance weighting (IDW) or other interpolation methods
    to create gridded fields from station observations.
    """

    def __init__(
        self,
        resolution: float = 0.25,
        method: str = "idw",
        max_distance: float = 2.0,
        power: float = 2.0,
    ):
        self.resolution = resolution
        self.method = method
        self.max_distance = max_distance
        self.power = power

        # Create grid
        self.lat_grid = np.arange(-90, 90 + resolution, resolution)
        self.lon_grid = np.arange(-180, 180, resolution)

    def grid_stations(
        self,
        stations: pd.DataFrame,
        variable: str,
    ) -> np.ndarray:
        """
        Grid station observations to regular grid.

        Args:
            stations: DataFrame with columns ['latitude', 'longitude', variable]
            variable: Column name to grid

        Returns:
            2D array of shape (n_lat, n_lon)
        """
        # Initialize output grid
        grid = np.full((len(self.lat_grid), len(self.lon_grid)), np.nan)

        # Get valid stations
        valid = stations[["latitude", "longitude", variable]].dropna()
        if len(valid) == 0:
            return grid

        station_lats = valid["latitude"].values
        station_lons = valid["longitude"].values
        station_vals = valid[variable].values

        # IDW interpolation
        for i, lat in enumerate(self.lat_grid):
            for j, lon in enumerate(self.lon_grid):
                # Calculate distances to all stations
                dlat = station_lats - lat
                dlon = station_lons - lon

                # Approximate distance in degrees
                distances = np.sqrt(dlat**2 + dlon**2)

                # Find stations within max distance
                mask = distances < self.max_distance
                if not mask.any():
                    continue

                nearby_distances = distances[mask]
                nearby_values = station_vals[mask]

                # Handle exact matches (distance = 0)
                if (nearby_distances == 0).any():
                    grid[i, j] = nearby_values[nearby_distances == 0][0]
                else:
                    # IDW weights
                    weights = 1.0 / (nearby_distances ** self.power)
                    grid[i, j] = np.average(nearby_values, weights=weights)

        return grid


class DataPipeline:
    """
    Main data processing pipeline.

    Coordinates downloading, quality control, encoding, and output.

    Example usage:
        pipeline = DataPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.downloader = GHCNDailyDownloader(output_dir=self.config.raw_dir)
        self.qc = QualityController()
        self.encoder = FeatureEncoder()
        self.gridder = SpatialGridder(resolution=self.config.grid_resolution)

        # Ensure output directories exist
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.tensor_dir).mkdir(parents=True, exist_ok=True)

    def run(
        self,
        stations: Optional[list] = None,
        max_stations: Optional[int] = None,
        download: bool = True,
    ) -> None:
        """
        Run the full pipeline.

        Args:
            stations: List of stations to process (or download new)
            max_stations: Maximum number of stations to process
            download: Whether to download data if not present
        """
        logger.info("Starting data pipeline")

        # 1. Get stations
        if stations is None:
            if download:
                self.downloader.download_stations()
                self.downloader.download_inventory()

            stations = self.downloader.get_stations(
                min_years=self.config.min_years,
                elements=self.config.target_variables,
            )

        if max_stations:
            stations = stations[:max_stations]

        logger.info(f"Processing {len(stations)} stations")

        # 2. Process each station
        all_data = []
        station_metadata = []

        for station in tqdm(stations, desc="Processing stations"):
            try:
                # Download if needed
                if download:
                    self.downloader.download_station_data(station.id)

                # Load and process
                df = self.downloader.station_to_dataframe(station.id)
                if df.empty:
                    continue

                # Quality control
                df_clean, flags = self.qc.process(df, station_id=station.id)

                # Fill small gaps
                df_clean, fill_flags = self.qc.fill_gaps(df_clean)

                # Filter to target variables
                target_cols = [c for c in self.config.target_variables if c in df_clean.columns]
                if not target_cols:
                    continue

                df_clean = df_clean[target_cols]

                # Add station metadata
                df_clean["station_id"] = station.id
                df_clean["latitude"] = station.latitude
                df_clean["longitude"] = station.longitude
                df_clean["elevation"] = station.elevation

                all_data.append(df_clean)
                station_metadata.append({
                    "station_id": station.id,
                    "name": station.name,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "elevation": station.elevation,
                    "country": station.id[:2],
                    "start_date": df_clean.index.min().isoformat(),
                    "end_date": df_clean.index.max().isoformat(),
                    "n_observations": len(df_clean),
                })

            except Exception as e:
                logger.warning(f"Error processing {station.id}: {e}")
                continue

        if not all_data:
            logger.error("No data processed successfully")
            return

        # 3. Combine all data
        logger.info("Combining station data")
        combined = pd.concat(all_data)

        # 4. Fit encoder on full dataset
        logger.info("Fitting feature encoder")
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["latitude", "longitude", "elevation"]]
        self.encoder.fit(combined[numeric_cols])

        # 5. Save encoder
        encoder_path = Path(self.config.output_dir) / "encoder.json"
        self.encoder.save(str(encoder_path))
        logger.info(f"Saved encoder to {encoder_path}")

        # 6. Save station metadata
        metadata_df = pd.DataFrame(station_metadata)
        metadata_path = Path(self.config.output_dir) / "stations.parquet"
        metadata_df.to_parquet(metadata_path)
        logger.info(f"Saved {len(metadata_df)} stations to {metadata_path}")

        # 7. Save processed data (partitioned by year)
        logger.info("Saving processed data")
        combined["year"] = combined.index.year

        for year, year_data in combined.groupby("year"):
            year_path = Path(self.config.output_dir) / f"observations_{year}.parquet"
            year_data.to_parquet(year_path)

        logger.success(f"Pipeline complete. Processed {len(station_metadata)} stations, {len(combined)} observations")

    def create_training_tensors(
        self,
        start_year: int = 1950,
        end_year: int = 2023,
        sequence_length: int = 365,
    ) -> None:
        """
        Create training tensors from processed data.

        Outputs Zarr arrays suitable for PyTorch DataLoaders.
        """
        import zarr

        logger.info(f"Creating training tensors for {start_year}-{end_year}")

        output_path = Path(self.config.tensor_dir)

        # Load encoder
        encoder_path = Path(self.config.output_dir) / "encoder.json"
        if encoder_path.exists():
            self.encoder = FeatureEncoder.load(str(encoder_path))

        # Load station metadata
        stations = pd.read_parquet(Path(self.config.output_dir) / "stations.parquet")

        # Initialize Zarr store
        store = zarr.DirectoryStore(str(output_path / "training"))
        root = zarr.group(store)

        # Process year by year
        all_features = []
        all_targets = []
        all_station_ids = []
        all_timestamps = []

        for year in tqdm(range(start_year, end_year + 1), desc="Years"):
            year_path = Path(self.config.output_dir) / f"observations_{year}.parquet"
            if not year_path.exists():
                continue

            df = pd.read_parquet(year_path)

            # Encode features
            encoded = self.encoder.transform(df[self.config.target_variables])

            # Store
            all_features.append(encoded.values)
            all_station_ids.extend(df["station_id"].tolist())
            all_timestamps.extend(df.index.tolist())

        # Concatenate and save
        if all_features:
            features = np.concatenate(all_features, axis=0)
            root.create_dataset("features", data=features, chunks=(10000, features.shape[1]))
            root.attrs["n_samples"] = len(features)
            root.attrs["feature_names"] = list(self.encoder.stats.keys())

            logger.success(f"Created training tensors: {features.shape}")


def main():
    """CLI entry point for the data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LILITH data pipeline")
    parser.add_argument("--raw-dir", default="data/raw/ghcn_daily", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/storage/parquet", help="Output directory")
    parser.add_argument("--max-stations", type=int, default=None, help="Max stations to process")
    parser.add_argument("--min-years", type=int, default=30, help="Min years of data required")
    parser.add_argument("--no-download", action="store_true", help="Don't download new data")
    parser.add_argument("--create-tensors", action="store_true", help="Create training tensors")

    args = parser.parse_args()

    config = PipelineConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        min_years=args.min_years,
    )

    pipeline = DataPipeline(config)
    pipeline.run(max_stations=args.max_stations, download=not args.no_download)

    if args.create_tensors:
        pipeline.create_training_tensors()


if __name__ == "__main__":
    main()
