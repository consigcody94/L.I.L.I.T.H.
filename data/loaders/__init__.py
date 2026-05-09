"""PyTorch DataLoaders for LILITH."""

from data.loaders.station_dataset import StationDataset, StationDataModule
from data.loaders.forecast_dataset import ForecastDataset, collate_variable_graphs

__all__ = [
    "StationDataset",
    "StationDataModule",
    "ForecastDataset",
    "collate_variable_graphs",  # used by scripts/train_model.py and training/trainer.py
]
