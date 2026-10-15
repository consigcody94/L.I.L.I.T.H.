"""Reproducible research training with train-only scaling and untouched test sets.

Run prepare_global first. This intentionally does not use the serving blend:
raw model skill must be measured before any NWP or climatology blending.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.simple_lilith import SimpleLILITH


from training.verification_metrics import error_metrics


def train(data_dir: Path, output: Path, epochs: int = 12, seed: int = 20261014,
          max_seconds: int = 1500) -> dict:
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat = device.type == "cuda" and torch.cuda.is_bf16_supported()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dataset = {name: dict(np.load(data_dir / f"{name}.npz", allow_pickle=False))
               for name in ("train", "validation", "test", "station_holdout")}
    train_data = dataset["train"]
    x_mean, x_std = train_data["X"].mean(axis=(0, 1)), train_data["X"].std(axis=(0, 1)) + 1e-6
    y_mean, y_std = train_data["Y"].mean(axis=(0, 1)), train_data["Y"].std(axis=(0, 1)) + 1e-6
    loaders = {}
    for name, arrays in dataset.items():
        tensors = TensorDataset(torch.from_numpy((arrays["X"] - x_mean) / x_std),
                                torch.from_numpy((arrays["Y"] - y_mean) / y_std),
                                torch.from_numpy(arrays["meta"]))
        loaders[name] = DataLoader(tensors, batch_size=256, shuffle=name == "train",
                                   num_workers=0, pin_memory=device.type == "cuda")
    config = dict(input_features=3, output_features=3, d_model=96, nhead=4,
                  num_encoder_layers=2, num_decoder_layers=2, dropout=0.1,
                  max_forecast=manifest["target_days"])
    model = SimpleLILITH(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bfloat)
    history, best, selected_epoch = [], float("inf"), None

    def predict(loader):
        model.eval()
        result = []
        with torch.no_grad():
            for x, y, meta in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16 if use_bfloat else torch.float16,
                                    enabled=device.type == "cuda"):
                    prediction = model(x.to(device), meta.to(device), y.shape[1])
                result.append(prediction.float().cpu().numpy())
        return np.concatenate(result) * y_std + y_mean

    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, meta in loaders["train"]:
            x, y, meta = x.to(device), y.to(device), meta.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16 if use_bfloat else torch.float16,
                                enabled=device.type == "cuda"):
                prediction = model(x, meta, manifest["target_days"])
                loss = torch.square(prediction - y).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(x)
            count += len(x)
        val_pred = predict(loaders["validation"])
        val_error = (val_pred - dataset["validation"]["Y"]) / y_std
        validation_loss = float(np.square(val_error).mean())
        history.append({"epoch": epoch + 1, "train_normalized_mse": total_loss / count,
                        "validation_normalized_mse": validation_loss,
                        "elapsed_seconds": round(time.monotonic() - started, 1)})
        print(json.dumps(history[-1]), flush=True)
        if validation_loss < best:
            best, selected_epoch = validation_loss, epoch + 1
            checkpoint = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": {**config, "input_days": 30},
                "normalization": {"X_mean": x_mean.tolist(), "X_std": x_std.tolist(),
                                  "Y_mean": y_mean.tolist(), "Y_std": y_std.tolist()},
                "epoch": selected_epoch, "seed": seed, "validation_loss": best,
                "status": "research_not_promoted", "training_manifest": manifest,
                "trained_horizon_days": manifest["target_days"],
            }
            torch.save(checkpoint, output / "candidate.pt")
        if time.monotonic() - started > max_seconds:
            print("Training time budget reached after completed epoch", flush=True)
            break
    model.load_state_dict(torch.load(output / "candidate.pt", map_location=device,
                                     weights_only=True)["model_state_dict"])
    evaluation = {}
    # Test sets are first evaluated after validation selects a checkpoint.
    for name in ("validation", "test", "station_holdout"):
        arrays = dataset[name]
        prediction = predict(loaders[name])
        # Same clipping must be applied to validation and all evaluations.
        # Preserve raw values here to expose physical violations rather than hide them.
        persistence = np.repeat(arrays["X"][:, -1:, :], manifest["target_days"], axis=1)
        actual = arrays["Y"]
        score = {"model": error_metrics(prediction, actual),
                 "persistence": error_metrics(persistence, actual),
                 "climatology": error_metrics(arrays["climatology"], actual)}
        model_mse = np.square(prediction - actual).mean(axis=(0, 1))
        climate_mse = np.square(arrays["climatology"] - actual).mean(axis=(0, 1))
        score["mse_skill_vs_climatology"] = (1 - model_mse / climate_mse).tolist()
        score["physical_violation_fraction"] = {
            "tmax_below_tmin": float((prediction[:, :, 0] < prediction[:, :, 1]).mean()),
            "negative_precipitation": float((prediction[:, :, 2] < 0).mean())}
        score["stations"] = {}
        for sid in sorted(set(arrays["station_id"])):
            mask = arrays["station_id"] == sid
            score["stations"][sid] = {
                "model": error_metrics(prediction[mask], actual[mask]),
                "climatology": error_metrics(arrays["climatology"][mask], actual[mask])}
        evaluation[name] = score
        np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction,
                            actual=actual, persistence=persistence, climatology=arrays["climatology"],
                            station_id=arrays["station_id"], target_start=arrays["target_start"])
    digest = hashlib.sha256((output / "candidate.pt").read_bytes()).hexdigest()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "research_not_promoted",
        "model": "SimpleLILITH", "config": config, "selected_epoch": selected_epoch,
        "seed": seed, "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": digest,
        "dataset_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
        "training_history": history, "evaluation": evaluation,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "promotion": {"approved": False, "reason": "Requires regional/seasonal skill, uncertainty calibration, independent review and operational replay"},
        "limitations": manifest["limitations"] + [
            "One random seed; no independent deep ensemble or probability calibration",
            "No humidity, wind, radar, cyclone or convective-event training targets",
            "Validation normalization objective equally weights standardized variables",
            "Historical daily truth can arrive after the simulated issue time",
            "90-day daily outputs, when requested, are research outputs, not validated event forecasts"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"selected_epoch": selected_epoch, "checkpoint_sha256": digest,
                      "test_model_rmse": evaluation["test"]["model"]["rmse"],
                      "test_climatology_rmse": evaluation["test"]["climatology"]["rmse"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-seconds", type=int, default=1500)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.seed, args.max_seconds)
