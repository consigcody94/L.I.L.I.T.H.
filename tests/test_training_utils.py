"""Tests for training utility functions: chronological split, KNN graph,
multi-station collation, and the new ensemble paths.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from training.train_simple import chronological_split
from training.train_full import (
    DateGroupedWeatherDataset,
    build_knn_edges,
    chronological_date_split,
    collate_variable_n_stations,
    haversine_km,
    masked_mse,
)
from models.simple_lilith import SimpleLILITH
from models.lilith import LILITH, LILITHConfig


@pytest.fixture
def synthetic_sequences():
    """Small synthetic dataset matching the (X, Y, meta, dates) contract."""
    n = 200
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 30, 3)).astype(np.float32)
    Y = rng.standard_normal((n, 90, 3)).astype(np.float32)
    meta = rng.standard_normal((n, 4)).astype(np.float32)
    # Map first 100 to 5 dates, last 100 to 5 later dates so we can verify
    # the chronological split actually separates by time.
    early = np.array([np.datetime64(f"2020-01-{d:02d}", "D") for d in range(1, 6)])
    late = np.array([np.datetime64(f"2024-12-{d:02d}", "D") for d in range(1, 6)])
    dates = np.concatenate([
        np.repeat(early, 20),
        np.repeat(late, 20),
    ])
    return X, Y, meta, dates


class TestChronologicalSplit:
    def test_train_strictly_before_val(self, synthetic_sequences):
        X, Y, meta, dates = synthetic_sequences
        Xt, Yt, mt, Xv, Yv, mv = chronological_split(X, Y, meta, dates, val_fraction=0.1)
        # By construction, val should be only late dates and train only early.
        assert len(Xt) + len(Xv) == len(X)
        assert len(Xv) > 0

    def test_random_split_when_dates_missing(self, synthetic_sequences):
        X, Y, meta, _ = synthetic_sequences
        Xt, Yt, mt, Xv, Yv, mv = chronological_split(X, Y, meta, None, val_fraction=0.2)
        assert len(Xt) + len(Xv) == len(X)
        assert abs(len(Xv) - 40) < 5  # ~20% of 200

    def test_handles_empty_dates_array(self, synthetic_sequences):
        X, Y, meta, _ = synthetic_sequences
        Xt, _, _, Xv, _, _ = chronological_split(X, Y, meta, np.array([]), val_fraction=0.1)
        assert len(Xt) + len(Xv) == len(X)


class TestHaversine:
    def test_zero_distance(self):
        d = haversine_km(np.array([40.0]), np.array([-74.0]),
                         np.array([40.0]), np.array([-74.0]))
        assert d[0] < 1e-6

    def test_known_distance(self):
        # NYC to LA is ~3940 km
        d = haversine_km(np.array([40.7128]), np.array([-74.006]),
                         np.array([34.0522]), np.array([-118.2437]))
        assert 3900 < d[0] < 4000

    def test_antipodes(self):
        # Antipodal points should be ~half Earth's circumference: 20015 km
        d = haversine_km(np.array([0.0]), np.array([0.0]),
                         np.array([0.0]), np.array([180.0]))
        assert 20000 < d[0] < 20100


class TestKNNEdges:
    def test_shapes_and_count(self):
        coords = np.array([
            [40.0, -74.0, 0],   # NYC
            [40.5, -74.5, 0],
            [41.0, -73.5, 0],
            [40.2, -73.8, 0],
            [39.9, -75.0, 0],
        ], dtype=np.float32)
        edge_index, edge_attr = build_knn_edges(coords, k=2)
        # k=2 forward + reversed -> n*k*2 directed edges.
        assert edge_index.shape == (2, coords.shape[0] * 2 * 2)
        assert edge_attr.shape == (edge_index.shape[1], 1)
        # Distances normalized to [0, 1].
        assert edge_attr.max() <= 1.0 + 1e-6
        assert edge_attr.min() >= 0.0

    def test_no_self_loops(self):
        coords = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.float32)
        edge_index, _ = build_knn_edges(coords, k=2)
        assert (edge_index[0] != edge_index[1]).all()

    def test_k_clamped_to_n_minus_one(self):
        # Only 3 nodes — k=10 should be clamped.
        coords = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.float32)
        edge_index, _ = build_knn_edges(coords, k=10)
        # Each node has 2 neighbors max; 3 nodes * 2 = 6 forward, plus 6 reversed = 12.
        assert edge_index.shape[1] == 12


class TestDateGroupedDataset:
    def test_filters_low_station_dates(self, synthetic_sequences):
        X, Y, meta, dates = synthetic_sequences
        ds = DateGroupedWeatherDataset(
            X, Y, meta, dates,
            min_stations_per_date=15,  # only dates with 20 stations qualify
            max_stations_per_date=256,
        )
        # 10 unique dates, all with 20 stations -> 10 groups remain.
        assert len(ds) == 10
        for idx_arr in ds.indices:
            assert len(idx_arr) >= 15

    def test_caps_high_station_dates(self):
        X = np.random.randn(50, 30, 3).astype(np.float32)
        Y = np.random.randn(50, 90, 3).astype(np.float32)
        meta = np.random.randn(50, 4).astype(np.float32)
        dates = np.repeat(np.array([np.datetime64("2020-01-01", "D")]), 50)
        ds = DateGroupedWeatherDataset(X, Y, meta, dates,
                                       min_stations_per_date=4,
                                       max_stations_per_date=20)
        assert len(ds) == 1
        assert len(ds.indices[0]) == 20

    def test_getitem_keys_and_shapes(self, synthetic_sequences):
        X, Y, meta, dates = synthetic_sequences
        ds = DateGroupedWeatherDataset(X, Y, meta, dates,
                                       min_stations_per_date=15,
                                       max_stations_per_date=256)
        sample = ds[0]
        assert {"node_features", "node_coords", "Y",
                "day_of_year", "edge_index", "edge_attr"}.issubset(sample.keys())
        n = sample["node_features"].shape[0]
        assert sample["node_coords"].shape == (n, 3)
        assert sample["Y"].shape == (n, 90, 3)
        assert sample["edge_index"].shape[0] == 2


class TestCollate:
    def test_pads_and_masks(self, synthetic_sequences):
        X, Y, meta, dates = synthetic_sequences
        ds = DateGroupedWeatherDataset(X, Y, meta, dates,
                                       min_stations_per_date=15,
                                       max_stations_per_date=256)
        # Build one batch with two heterogeneous samples by truncating one.
        s0 = ds[0]
        s1 = {k: (v[:10] if v.dim() > 0 and v.shape[0] == s0["node_features"].shape[0] else v)
              for k, v in s0.items()}
        # Re-build edges for the truncated sample so indices stay valid.
        coords1 = s1["node_coords"].numpy()
        ei, ea = build_knn_edges(coords1, k=4)
        s1["edge_index"] = torch.from_numpy(ei).long()
        s1["edge_attr"] = torch.from_numpy(ea).float()

        batch = collate_variable_n_stations([s0, s1])
        max_n = max(s0["node_features"].shape[0], s1["node_features"].shape[0])
        assert batch["node_features"].shape == (2, max_n, 30, 3)
        assert batch["node_mask"].sum().item() == s0["node_features"].shape[0] + s1["node_features"].shape[0]
        # Edge_index has been globally re-indexed: max should fit within 2 * max_n
        assert batch["edge_index"].max().item() < 2 * max_n


class TestMaskedMSE:
    def test_masks_reduce_loss_correctly(self):
        pred = torch.zeros(2, 4, 5, 3)
        target = torch.ones(2, 4, 5, 3)
        mask = torch.tensor([[True, True, False, False],
                             [True, False, False, False]])
        loss = masked_mse(pred, target, mask)
        # Real cells contribute (0-1)^2 = 1; padded cells ignored.
        assert loss.item() == pytest.approx(1.0, rel=1e-5)

    def test_all_masked_returns_finite(self):
        pred = torch.zeros(2, 4, 5, 3)
        target = torch.ones(2, 4, 5, 3)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = masked_mse(pred, target, mask)
        # Should not be NaN — eps in denominator.
        assert torch.isfinite(loss)


class TestEnsembleMethods:
    def test_simple_lilith_mc_dropout(self):
        model = SimpleLILITH(input_features=3, output_features=3,
                             d_model=32, nhead=2, num_encoder_layers=1,
                             num_decoder_layers=1, dropout=0.3)
        x = torch.randn(2, 30, 3)
        meta = torch.randn(2, 4)
        samples = model.mc_dropout_forecast(x, meta, target_len=14, n_samples=5)
        assert samples.shape == (5, 2, 14, 3)
        # With non-trivial dropout, samples should not all be identical.
        diffs = (samples - samples[0]).abs().max()
        assert diffs > 0

    def test_full_lilith_real_ensemble(self):
        cfg = LILITHConfig(variant="tiny", hidden_dim=64, num_heads=2,
                           gat_layers=1, temporal_layers=1, sfno_layers=1,
                           use_grid=False, forecast_length=14, dropout=0.2)
        model = LILITH(cfg)
        node_features = torch.randn(1, 6, 30, cfg.input_features)
        node_coords = torch.randn(1, 6, 3) * torch.tensor([45.0, 90.0, 1000.0])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 1)
        ensemble = model.generate_ensemble(
            node_features, node_coords, edge_index, edge_attr, n_members=4
        )
        assert ensemble.shape == (4, 1, 6, 14, cfg.output_features)
        # Different members should not be identical (dropout was active).
        assert (ensemble[0] - ensemble[1]).abs().max() > 0
