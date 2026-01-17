"""
Station-based PyTorch Dataset for LILITH.

Provides efficient data loading for station observations with support for:
- Sequence-based loading for temporal models
- Multi-station batching for graph-based models
- Lazy loading for large datasets
- Train/val/test splitting
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


@dataclass
class StationSample:
    """A single training sample from a station."""

    station_id: str
    latitude: float
    longitude: float
    elevation: float

    # Input sequence
    input_features: torch.Tensor  # Shape: (seq_len, n_features)
    input_mask: torch.Tensor  # Shape: (seq_len,) - True for valid values

    # Target sequence (for forecasting)
    target_features: torch.Tensor  # Shape: (forecast_len, n_targets)
    target_mask: torch.Tensor  # Shape: (forecast_len,)

    # Timestamps
    input_timestamps: np.ndarray
    target_timestamps: np.ndarray


class StationDataset(Dataset):
    """
    PyTorch Dataset for station-based weather data.

    Loads sequences of observations from individual stations for
    training temporal forecasting models.

    Example usage:
        dataset = StationDataset(
            data_dir="data/storage/parquet",
            sequence_length=365,
            forecast_length=90,
            target_variables=["TMAX", "TMIN", "PRCP"],
        )
        sample = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 365,
        forecast_length: int = 90,
        target_variables: Optional[list[str]] = None,
        input_variables: Optional[list[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        station_ids: Optional[list[str]] = None,
        min_valid_ratio: float = 0.8,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing processed Parquet files
            sequence_length: Number of days in input sequence
            forecast_length: Number of days to forecast
            target_variables: Variables to predict (default: TMAX, TMIN, PRCP)
            input_variables: Variables to use as input (default: all available)
            start_year: Start year for data (inclusive)
            end_year: End year for data (inclusive)
            station_ids: Specific stations to include (default: all)
            min_valid_ratio: Minimum ratio of valid values in a sequence
            normalize: Whether data is already normalized
            cache_in_memory: Load all data into memory (faster, more RAM)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.total_length = sequence_length + forecast_length
        self.min_valid_ratio = min_valid_ratio
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        # Default variables
        self.target_variables = target_variables or ["TMAX", "TMIN", "PRCP"]
        self.input_variables = input_variables

        # Load station metadata
        self.stations = self._load_stations()

        # Filter stations if specified
        if station_ids:
            self.stations = self.stations[self.stations["station_id"].isin(station_ids)]

        # Build index of valid samples
        self.samples = self._build_sample_index(start_year, end_year)

        # Cache for data
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"StationDataset initialized: {len(self.stations)} stations, "
            f"{len(self.samples)} samples"
        )

    def _load_stations(self) -> pd.DataFrame:
        """Load station metadata."""
        stations_path = self.data_dir / "stations.parquet"
        if not stations_path.exists():
            raise FileNotFoundError(f"Station metadata not found: {stations_path}")

        return pd.read_parquet(stations_path)

    def _build_sample_index(
        self,
        start_year: Optional[int],
        end_year: Optional[int],
    ) -> list[Tuple[str, pd.Timestamp]]:
        """
        Build an index of valid training samples.

        Returns list of (station_id, start_date) tuples.
        """
        samples = []

        # Find available year files
        year_files = sorted(self.data_dir.glob("observations_*.parquet"))

        for year_file in year_files:
            year = int(year_file.stem.split("_")[1])

            # Filter by year range
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue

            # Load year data
            df = pd.read_parquet(year_file)

            # Group by station
            for station_id, station_data in df.groupby("station_id"):
                # Check if station has enough data
                if len(station_data) < self.total_length:
                    continue

                # Find valid sequence start points
                # (where we have enough consecutive data)
                dates = station_data.index.sort_values()

                for i in range(len(dates) - self.total_length + 1):
                    start_date = dates[i]
                    end_date = dates[i + self.total_length - 1]

                    # Check for gaps (should be consecutive days)
                    expected_days = self.total_length
                    actual_days = (end_date - start_date).days + 1

                    if actual_days == expected_days:
                        # Check valid ratio
                        sample_data = station_data.loc[start_date:end_date]
                        target_cols = [c for c in self.target_variables if c in sample_data.columns]
                        valid_ratio = sample_data[target_cols].notna().mean().mean()

                        if valid_ratio >= self.min_valid_ratio:
                            samples.append((station_id, start_date))

        return samples

    def _load_station_data(self, station_id: str, year: int) -> pd.DataFrame:
        """Load data for a specific station and year."""
        cache_key = f"{station_id}_{year}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        year_file = self.data_dir / f"observations_{year}.parquet"
        if not year_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(year_file)
        station_data = df[df["station_id"] == station_id].sort_index()

        if self.cache_in_memory:
            self._cache[cache_key] = station_data

        return station_data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns dict with keys:
        - input_features: (seq_len, n_features)
        - input_mask: (seq_len,)
        - target_features: (forecast_len, n_targets)
        - target_mask: (forecast_len,)
        - station_coords: (3,) - [lat, lon, elev]
        - timestamps: (total_len,)
        """
        station_id, start_date = self.samples[idx]
        year = start_date.year

        # Load data (may span two years)
        data = self._load_station_data(station_id, year)
        if year + 1 <= 2023:  # Check for year boundary
            next_year_data = self._load_station_data(station_id, year + 1)
            if not next_year_data.empty:
                data = pd.concat([data, next_year_data])

        # Extract sequence
        end_date = start_date + pd.Timedelta(days=self.total_length - 1)
        sequence = data.loc[start_date:end_date]

        if len(sequence) < self.total_length:
            # Pad if necessary
            sequence = sequence.reindex(
                pd.date_range(start_date, periods=self.total_length, freq="D")
            )

        # Get station metadata
        station_meta = self.stations[self.stations["station_id"] == station_id].iloc[0]

        # Prepare features
        feature_cols = self.input_variables or [
            c for c in sequence.columns
            if c not in ["station_id", "latitude", "longitude", "elevation", "year"]
        ]

        # Input sequence
        input_seq = sequence.iloc[:self.sequence_length]
        input_features = input_seq[feature_cols].values.astype(np.float32)
        input_mask = ~np.isnan(input_features).any(axis=1)

        # Target sequence
        target_seq = sequence.iloc[self.sequence_length:]
        target_cols = [c for c in self.target_variables if c in sequence.columns]
        target_features = target_seq[target_cols].values.astype(np.float32)
        target_mask = ~np.isnan(target_features).any(axis=1)

        # Fill NaN with 0 for tensor conversion (mask indicates valid values)
        input_features = np.nan_to_num(input_features, nan=0.0)
        target_features = np.nan_to_num(target_features, nan=0.0)

        # Station coordinates
        station_coords = np.array([
            station_meta["latitude"],
            station_meta["longitude"],
            station_meta["elevation"],
        ], dtype=np.float32)

        return {
            "input_features": torch.from_numpy(input_features),
            "input_mask": torch.from_numpy(input_mask),
            "target_features": torch.from_numpy(target_features),
            "target_mask": torch.from_numpy(target_mask),
            "station_coords": torch.from_numpy(station_coords),
            "station_id": station_id,
        }


class StationDataModule:
    """
    Data module for managing train/val/test splits.

    Provides DataLoaders with proper batching and shuffling.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        sequence_length: int = 365,
        forecast_length: int = 90,
        **dataset_kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.dataset_kwargs = dataset_kwargs

        self._train_dataset: Optional[StationDataset] = None
        self._val_dataset: Optional[StationDataset] = None
        self._test_dataset: Optional[StationDataset] = None

    def setup(self) -> None:
        """Set up train/val/test datasets."""
        # Load all stations
        stations = pd.read_parquet(self.data_dir / "stations.parquet")
        all_station_ids = stations["station_id"].tolist()

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_station_ids)

        n_train = int(len(all_station_ids) * self.train_ratio)
        n_val = int(len(all_station_ids) * self.val_ratio)

        train_ids = all_station_ids[:n_train]
        val_ids = all_station_ids[n_train:n_train + n_val]
        test_ids = all_station_ids[n_train + n_val:]

        # Create datasets
        common_kwargs = {
            "data_dir": self.data_dir,
            "sequence_length": self.sequence_length,
            "forecast_length": self.forecast_length,
            **self.dataset_kwargs,
        }

        self._train_dataset = StationDataset(station_ids=train_ids, **common_kwargs)
        self._val_dataset = StationDataset(station_ids=val_ids, **common_kwargs)
        self._test_dataset = StationDataset(station_ids=test_ids, **common_kwargs)

        logger.info(
            f"Data split: {len(self._train_dataset)} train, "
            f"{len(self._val_dataset)} val, {len(self._test_dataset)} test"
        )

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._train_dataset is None:
            self.setup()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self._val_dataset is None:
            self.setup()
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        if self._test_dataset is None:
            self.setup()
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
