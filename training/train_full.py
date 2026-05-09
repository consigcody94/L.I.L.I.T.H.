"""
Train the full LILITH model (Station-Graph Temporal Transformer).

The original repo only ever trained SimpleLILITH; the GAT + SFNO + ensemble
stack in models/lilith.py was dead code. This trainer wires it up properly:

    1. Group existing per-station sequences by forecast date so each batch
       contains many stations seen at the same moment in time.
    2. Build a K-NN graph from station coordinates (great-circle distance).
    3. Forward through the full pipeline (StationEmbed -> GAT -> Grid -> SFNO
       -> TemporalTransformer -> Decoder -> EnsembleHead).
    4. Train with masked MSE; switch to CRPS / Gaussian-NLL once the
       ensemble head's uncertainty is being supervised explicitly.

NOTE on the architecture: the GAT + SFNO + station-graph stack is a
research-novel combination — I am not aware of a published paper that
benchmarks this exact arrangement at the surface-station level. GraphCast
(arXiv:2212.12794) uses a graph but on the sphere directly, not GAT;
FourCastNet / SFNO (arXiv:2306.03838) operate on regular grids, not
stations. Treat this trainer as an experimental research path, not a known
SOTA architecture. If your goal is "best forecast skill, fastest" the
safer first move is fine-tuning a foundation TS model (TimesFM / Chronos)
or distilling pretrained GraphCast weights — both are stronger established
baselines than building a custom architecture from scratch.

Hardware target: RTX 5070 (12 GB, Blackwell). LILITH-Tiny (~50 M params)
fits comfortably in BF16; LILITH-Base needs gradient checkpointing.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lilith import LILITH, LILITHConfig
from models.losses import CRPSLoss, GaussianNLLLoss, WeightedMSELoss


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between two arrays of lat/lon (degrees).

    Vectorized — pass in NxM via broadcasting to get pairwise distances.
    """
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_knn_edges(coords: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """K-NN edges on the sphere by great-circle distance.

    Args:
        coords: (n_stations, 3) — lat, lon, elev. Only lat/lon used.
        k: neighbors per node, edges are bidirectional.

    Returns:
        edge_index: (2, n_edges) int64
        edge_dist:  (n_edges, 1) float32 — distance in km, useful as edge_attr
    """
    n = coords.shape[0]
    lat = coords[:, 0]
    lon = coords[:, 1]
    # Pairwise distances
    dists = haversine_km(
        lat[:, None], lon[:, None], lat[None, :], lon[None, :]
    )
    # Self-loops set to inf so they don't get picked as neighbors
    np.fill_diagonal(dists, np.inf)

    # k nearest indices per row
    k_eff = min(k, n - 1)
    nbr_idx = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]

    src = np.repeat(np.arange(n), k_eff)
    dst = nbr_idx.flatten()
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # Make undirected (also include reverse)
    edge_index_rev = edge_index[[1, 0]]
    edge_index = np.concatenate([edge_index, edge_index_rev], axis=1)

    edge_dist = dists[edge_index[0], edge_index[1]].astype(np.float32).reshape(-1, 1)
    # Normalize by max distance for stable training
    if edge_dist.max() > 0:
        edge_dist = edge_dist / edge_dist.max()

    return edge_index, edge_dist


class DateGroupedWeatherDataset(Dataset):
    """One sample = one forecast date with all stations observed on that date.

    The full LILITH model needs multi-station batches because the GAT and SFNO
    stages operate on station relationships. Groups the per-station sequences
    by forecast start date and yields tuples ready for the graph forward pass.

    Each sample returns:
        node_features: (n_stations_today, seq_len, n_features)
        node_coords:   (n_stations_today, 3)
        Y:             (n_stations_today, target_days, n_features)
        day_of_year:   scalar (normalized 0..1)
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        meta: np.ndarray,
        dates: np.ndarray,
        min_stations_per_date: int = 4,
        max_stations_per_date: int = 256,
        seed: int = 0,
    ):
        # Group indices by date
        unique_dates, inv = np.unique(dates, return_inverse=True)
        groups: Dict[int, List[int]] = {}
        for sample_idx, group_idx in enumerate(inv):
            groups.setdefault(int(group_idx), []).append(sample_idx)

        # Filter to dates with enough stations to be worth a batch
        self.dates = []
        self.indices: List[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for group_idx, idx_list in groups.items():
            if len(idx_list) < min_stations_per_date:
                continue
            if len(idx_list) > max_stations_per_date:
                idx_list = list(rng.choice(idx_list, max_stations_per_date, replace=False))
            self.dates.append(unique_dates[group_idx])
            self.indices.append(np.asarray(idx_list, dtype=np.int64))

        self.X = X
        self.Y = Y
        self.meta = meta
        logger.info(
            f"Date-grouped dataset: {len(self.dates):,} dates, "
            f"avg stations/date={np.mean([len(i) for i in self.indices]):.1f}"
        )

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.indices[idx]
        Xs = self.X[sample_idx]      # (n_st, seq_len, F)
        Ys = self.Y[sample_idx]
        ms = self.meta[sample_idx]   # (n_st, 4)  lat, lon, elev, doy

        coords = ms[:, :3].astype(np.float32)
        edge_index, edge_attr = build_knn_edges(coords, k=8)

        return {
            "node_features": torch.from_numpy(Xs).float(),
            "node_coords": torch.from_numpy(coords).float(),
            "Y": torch.from_numpy(Ys).float(),
            "day_of_year": torch.tensor(ms[0, 3], dtype=torch.float32),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr).float(),
        }


def collate_variable_n_stations(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad to the largest n_stations in the batch and emit a mask.

    Different dates have different station counts. We pad with zeros and
    track validity in node_mask so the loss only counts real stations.
    """
    max_n = max(b["node_features"].shape[0] for b in batch)
    seq_len = batch[0]["node_features"].shape[1]
    feat_dim = batch[0]["node_features"].shape[-1]
    out_dim = batch[0]["Y"].shape[-1]
    target_len = batch[0]["Y"].shape[1]
    B = len(batch)

    node_features = torch.zeros(B, max_n, seq_len, feat_dim)
    node_coords = torch.zeros(B, max_n, 3)
    Y = torch.zeros(B, max_n, target_len, out_dim)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)
    day_of_year = torch.zeros(B)

    # Edge index needs to be re-indexed to a global node id per sample.
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    for i, item in enumerate(batch):
        n = item["node_features"].shape[0]
        node_features[i, :n] = item["node_features"]
        node_coords[i, :n] = item["node_coords"]
        Y[i, :n] = item["Y"]
        node_mask[i, :n] = True
        day_of_year[i] = item["day_of_year"]
        ei = item["edge_index"] + i * max_n  # offset to flatten across batch
        edge_indices.append(ei)
        edge_attrs.append(item["edge_attr"])

    return {
        "node_features": node_features,
        "node_coords": node_coords,
        "Y": Y,
        "node_mask": node_mask,
        "day_of_year": day_of_year,
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
    }


def chronological_date_split(
    dataset: DateGroupedWeatherDataset,
    val_fraction: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """Train/val by date — val is the most-recent slice."""
    order = np.argsort(dataset.dates)
    n_train = int(len(order) * (1 - val_fraction))
    return order[:n_train].tolist(), order[n_train:].tolist()


def select_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE that ignores padded stations.

    pred/target: (B, n_stations, target_len, out_dim)
    mask:        (B, n_stations) — True for real stations.
    """
    sq = (pred - target).pow(2)
    m = mask[:, :, None, None].float()
    return (sq * m).sum() / (m.sum() * sq.shape[-1] * sq.shape[-2] + 1e-8)


def train_one_epoch(
    model: LILITH,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    amp_dtype: torch.dtype,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch", leave=True)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(
                node_features=batch["node_features"],
                node_coords=batch["node_coords"],
                edge_index=batch["edge_index"],
                edge_attr=batch["edge_attr"],
                day_of_year=None,
            )
            loss = masked_mse(out["forecast"], batch["Y"], batch["node_mask"])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{total/n:.4f}"})

    return total / max(n, 1)


@torch.no_grad()
def validate(
    model: LILITH,
    loader: DataLoader,
    amp_dtype: torch.dtype,
    device: torch.device,
    Y_std_temp_mean: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    sq_err = torch.zeros(1, device=device)
    n_elems = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(
                node_features=batch["node_features"],
                node_coords=batch["node_coords"],
                edge_index=batch["edge_index"],
                edge_attr=batch["edge_attr"],
            )
            loss = masked_mse(out["forecast"], batch["Y"], batch["node_mask"])
        total_loss += loss.item()
        n += 1

        # Temperature RMSE on first 2 channels, masked
        diff = (out["forecast"][..., :2].float() - batch["Y"][..., :2].float())
        m = batch["node_mask"][:, :, None, None].float()
        sq_err += (diff.pow(2) * m).sum()
        n_elems += int(m.sum().item()) * diff.shape[-2] * diff.shape[-1]

    rmse = (sq_err / max(n_elems, 1)).sqrt().item() * Y_std_temp_mean
    return total_loss / max(n, 1), rmse


def main():
    parser = argparse.ArgumentParser(description="Train the full LILITH (graph + SFNO) model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of dates per batch — multi-station batches are large")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--variant", type=str, default="tiny",
                        choices=["tiny", "base", "large", "xl"],
                        help="LILITH variant. 5070 12 GB fits 'tiny' easily, 'base' with gradient ckpt.")
    parser.add_argument("--use-grid", action="store_true",
                        help="Enable Station->Grid->SFNO->Station path (heavier)")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ensemble-method", type=str, default="mc_dropout",
                        choices=["gaussian", "quantile", "mc_dropout", "diffusion"])
    parser.add_argument("--target-days", type=int, default=90)
    parser.add_argument("--max-stations-per-date", type=int, default=128)
    parser.add_argument("--min-stations-per-date", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / "data" / "processed" / "training"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if not (training_dir / "X.npy").exists():
        raise FileNotFoundError(
            f"Training data not found at {training_dir}. Run "
            "`python -m training.train_simple --process-data` first to generate it."
        )

    logger.info("Loading training data...")
    X = np.load(training_dir / "X.npy")
    Y = np.load(training_dir / "Y.npy")
    meta = np.load(training_dir / "meta.npy")
    dates_path = training_dir / "dates.npy"
    if not dates_path.exists():
        raise FileNotFoundError(
            "dates.npy missing — required for the full LILITH trainer. "
            "Re-run `python -m training.train_simple --process-data` to regenerate."
        )
    dates = np.load(dates_path)

    stats = np.load(training_dir / "stats.npz")
    X = (X - stats["X_mean"]) / (stats["X_std"] + 1e-6)
    Y = (Y - stats["Y_mean"]) / (stats["Y_std"] + 1e-6)
    Y_std_temp_mean = float(np.mean(stats["Y_std"][:2]))

    # Don't normalize lat/lon/elev here; the StationEmbedding's PositionalEncoding3D
    # expects raw degrees / meters and applies its own sin/cos transforms.
    meta = meta.copy().astype(np.float32)

    dataset = DateGroupedWeatherDataset(
        X, Y, meta, dates,
        min_stations_per_date=args.min_stations_per_date,
        max_stations_per_date=args.max_stations_per_date,
        seed=args.seed,
    )
    train_idx, val_idx = chronological_date_split(dataset)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    if args.num_workers is None:
        num_workers = 2 if os.name == "nt" else 4
    else:
        num_workers = args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_variable_n_stations,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_variable_n_stations,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB | sm_{props.major}{props.minor}")

    cfg = LILITHConfig(
        variant=args.variant,
        input_features=X.shape[-1],
        output_features=Y.shape[-1],
        sequence_length=X.shape[1],
        forecast_length=args.target_days,
        use_grid=args.use_grid,
        gradient_checkpointing=args.gradient_checkpointing,
        ensemble_method=args.ensemble_method,
    )
    model = LILITH(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.variant} ({n_params:,} params)")

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_dtype = select_amp_dtype(device) if not args.no_amp else torch.float32
    scaler = (
        torch.amp.GradScaler(device.type)
        if (amp_dtype == torch.float16 and not args.no_amp)
        else None
    )
    logger.info(f"AMP dtype: {amp_dtype}, scaler: {'on' if scaler else 'off'}")

    start_epoch = 0
    best_val = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd, strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best val loss {best_val:.4f}")

    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
        f"Train dates: {len(train_dataset):,} | Val dates: {len(val_dataset):,}"
    )

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, amp_dtype, device, epoch, args.epochs)
        val_loss, val_rmse = validate(model, val_loader, amp_dtype, device, Y_std_temp_mean)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Temp RMSE: {val_rmse:.2f}°C"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": (
                    model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "config": cfg.__dict__,
                "normalization": {k: stats[k].tolist() for k in ("X_mean", "X_std", "Y_mean", "Y_std")},
            }
            torch.save(ckpt, checkpoints_dir / "lilith_full_best.pt")
            logger.success(f"Saved best — RMSE {val_rmse:.2f}°C")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(ckpt, checkpoints_dir / f"lilith_full_{timestamp}.pt")
    logger.success(f"Done. Best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
