"""
Forecast Dataset for LILITH.

Provides data loading optimized for multi-station forecasting
with graph-based models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


class ForecastDataset(Dataset):
    """
    Dataset for graph-based multi-station forecasting.

    Instead of loading single stations, this dataset loads data for
    multiple stations simultaneously, suitable for GNN-based models.

    Each sample contains:
    - Observations from N stations for the input period
    - Targets for N stations for the forecast period
    - Station coordinates and connectivity graph
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 30,
        forecast_length: int = 14,
        max_stations: int = 500,
        spatial_radius: float = 5.0,  # degrees
        target_variables: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the forecast dataset.

        Args:
            data_dir: Directory with processed Parquet files
            sequence_length: Days of input history
            forecast_length: Days to forecast
            max_stations: Maximum stations per sample
            spatial_radius: Radius in degrees for station sampling
            target_variables: Variables to forecast
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.total_length = sequence_length + forecast_length
        self.max_stations = max_stations
        self.spatial_radius = spatial_radius
        self.target_variables = target_variables or ["TMAX", "TMIN", "PRCP"]
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Load station metadata
        self.stations = pd.read_parquet(self.data_dir / "stations.parquet")

        # Parse date range
        self.start_date = pd.Timestamp(start_date) if start_date else pd.Timestamp("2000-01-01")
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp("2023-12-31")

        # Build date index
        self.dates = pd.date_range(
            self.start_date,
            self.end_date - pd.Timedelta(days=self.total_length),
            freq="D",
        )

        # Build spatial clusters for efficient sampling
        self._build_spatial_clusters()

        # Cache for loaded data
        self._data_cache: Dict[int, pd.DataFrame] = {}

        logger.info(
            f"ForecastDataset: {len(self.dates)} dates, "
            f"{len(self.stations)} stations, {len(self.clusters)} clusters"
        )

    def _build_spatial_clusters(self) -> None:
        """
        Build spatial clusters of stations for efficient sampling.

        Groups stations into overlapping clusters based on spatial proximity.
        """
        self.clusters = []

        # Grid-based clustering
        lat_bins = np.arange(-90, 90, self.spatial_radius * 2)
        lon_bins = np.arange(-180, 180, self.spatial_radius * 2)

        for lat in lat_bins:
            for lon in lon_bins:
                # Find stations in this grid cell (with overlap)
                mask = (
                    (self.stations["latitude"] >= lat - self.spatial_radius) &
                    (self.stations["latitude"] < lat + self.spatial_radius * 3) &
                    (self.stations["longitude"] >= lon - self.spatial_radius) &
                    (self.stations["longitude"] < lon + self.spatial_radius * 3)
                )
                cluster_stations = self.stations[mask]["station_id"].tolist()

                if len(cluster_stations) >= 10:  # Minimum cluster size
                    self.clusters.append({
                        "center_lat": lat + self.spatial_radius,
                        "center_lon": lon + self.spatial_radius,
                        "station_ids": cluster_stations,
                    })

    def _load_data_for_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Load data for a specific date range, with caching."""
        year = date.year
        end_year = (date + pd.Timedelta(days=self.total_length)).year

        # Load required years
        dfs = []
        for y in range(year, end_year + 1):
            if y in self._data_cache:
                dfs.append(self._data_cache[y])
            else:
                year_file = self.data_dir / f"observations_{y}.parquet"
                if year_file.exists():
                    df = pd.read_parquet(year_file)
                    self._data_cache[y] = df
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs)

    def _build_station_graph(
        self,
        station_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build adjacency information for stations.

        Returns edge_index and edge_attr for PyTorch Geometric.

        Args:
            station_coords: (N, 3) array of [lat, lon, elev]

        Returns:
            edge_index: (2, E) source and target node indices
            edge_attr: (E, 1) edge distances
        """
        n_stations = len(station_coords)
        edges_src = []
        edges_dst = []
        edge_weights = []

        # Connect stations within spatial radius
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                # Calculate distance
                dlat = station_coords[i, 0] - station_coords[j, 0]
                dlon = station_coords[i, 1] - station_coords[j, 1]
                dist = np.sqrt(dlat**2 + dlon**2)

                if dist < self.spatial_radius:
                    # Bidirectional edges
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                    edge_weights.extend([dist, dist])

        if not edges_src:
            # Fallback: connect to k nearest neighbors
            from scipy.spatial import KDTree

            tree = KDTree(station_coords[:, :2])
            for i in range(n_stations):
                _, neighbors = tree.query(station_coords[i, :2], k=min(5, n_stations))
                for j in neighbors:
                    if i != j:
                        dist = np.linalg.norm(station_coords[i, :2] - station_coords[j, :2])
                        edges_src.append(i)
                        edges_dst.append(j)
                        edge_weights.append(dist)

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_attr = np.array(edge_weights, dtype=np.float32).reshape(-1, 1)

        return edge_index, edge_attr

    def __len__(self) -> int:
        return len(self.dates) * len(self.clusters)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multi-station sample.

        Returns:
            Dict with keys:
            - node_features: (N, seq_len, F) station observations
            - node_coords: (N, 3) lat/lon/elev
            - edge_index: (2, E) graph connectivity
            - edge_attr: (E, 1) edge weights
            - target_features: (N, forecast_len, T) targets
            - mask: (N, seq_len + forecast_len) valid mask
        """
        # Decode index
        date_idx = idx // len(self.clusters)
        cluster_idx = idx % len(self.clusters)

        date = self.dates[date_idx]
        cluster = self.clusters[cluster_idx]

        # Sample stations from cluster
        station_ids = cluster["station_ids"]
        if len(station_ids) > self.max_stations:
            station_ids = self.rng.choice(station_ids, self.max_stations, replace=False).tolist()

        n_stations = len(station_ids)

        # Load data
        data = self._load_data_for_date(date)
        if data.empty:
            return self._empty_sample(n_stations)

        # Filter to selected stations and date range
        end_date = date + pd.Timedelta(days=self.total_length - 1)
        mask = (
            data["station_id"].isin(station_ids) &
            (data.index >= date) &
            (data.index <= end_date)
        )
        data = data[mask]

        # Prepare feature arrays
        feature_cols = [c for c in self.target_variables if c in data.columns]
        n_features = len(feature_cols)

        node_features = np.zeros((n_stations, self.sequence_length, n_features), dtype=np.float32)
        target_features = np.zeros((n_stations, self.forecast_length, n_features), dtype=np.float32)
        node_coords = np.zeros((n_stations, 3), dtype=np.float32)
        valid_mask = np.zeros((n_stations, self.total_length), dtype=bool)

        # Fill in data for each station
        for i, station_id in enumerate(station_ids):
            station_data = data[data["station_id"] == station_id].sort_index()

            # Get station coords
            station_meta = self.stations[self.stations["station_id"] == station_id]
            if not station_meta.empty:
                node_coords[i] = [
                    station_meta.iloc[0]["latitude"],
                    station_meta.iloc[0]["longitude"],
                    station_meta.iloc[0].get("elevation", 0),
                ]

            # Fill input sequence
            for j, d in enumerate(pd.date_range(date, periods=self.sequence_length, freq="D")):
                if d in station_data.index:
                    row = station_data.loc[d]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    for k, col in enumerate(feature_cols):
                        val = row.get(col, np.nan)
                        if not pd.isna(val):
                            node_features[i, j, k] = val
                            valid_mask[i, j] = True

            # Fill target sequence
            target_start = date + pd.Timedelta(days=self.sequence_length)
            for j, d in enumerate(pd.date_range(target_start, periods=self.forecast_length, freq="D")):
                if d in station_data.index:
                    row = station_data.loc[d]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    for k, col in enumerate(feature_cols):
                        val = row.get(col, np.nan)
                        if not pd.isna(val):
                            target_features[i, j, k] = val
                            valid_mask[i, self.sequence_length + j] = True

        # Build graph
        edge_index, edge_attr = self._build_station_graph(node_coords)

        # Replace NaN with 0 (mask indicates valid values)
        node_features = np.nan_to_num(node_features, nan=0.0)
        target_features = np.nan_to_num(target_features, nan=0.0)

        return {
            "node_features": torch.from_numpy(node_features),
            "node_coords": torch.from_numpy(node_coords),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "target_features": torch.from_numpy(target_features),
            "mask": torch.from_numpy(valid_mask),
            "n_stations": n_stations,
            "date": str(date.date()),
        }

    def _empty_sample(self, n_stations: int) -> Dict[str, torch.Tensor]:
        """Return an empty sample for error cases."""
        return {
            "node_features": torch.zeros(n_stations, self.sequence_length, len(self.target_variables)),
            "node_coords": torch.zeros(n_stations, 3),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_attr": torch.zeros(0, 1),
            "target_features": torch.zeros(n_stations, self.forecast_length, len(self.target_variables)),
            "mask": torch.zeros(n_stations, self.total_length, dtype=torch.bool),
            "n_stations": n_stations,
            "date": "",
        }


def collate_variable_graphs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size graphs.

    Combines multiple samples into a single batched graph.
    """
    # Stack fixed-size tensors
    node_features = torch.cat([b["node_features"] for b in batch], dim=0)
    node_coords = torch.cat([b["node_coords"] for b in batch], dim=0)
    target_features = torch.cat([b["target_features"] for b in batch], dim=0)
    masks = torch.cat([b["mask"] for b in batch], dim=0)

    # Combine edge indices with offsets
    edge_indices = []
    edge_attrs = []
    offset = 0

    for b in batch:
        edge_index = b["edge_index"]
        if edge_index.size(1) > 0:
            edge_indices.append(edge_index + offset)
            edge_attrs.append(b["edge_attr"])
        offset += b["n_stations"]

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)

    # Batch indices for graph batching
    batch_idx = torch.cat([
        torch.full((b["n_stations"],), i, dtype=torch.long)
        for i, b in enumerate(batch)
    ])

    return {
        "node_features": node_features,
        "node_coords": node_coords,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "target_features": target_features,
        "mask": masks,
        "batch": batch_idx,
    }
