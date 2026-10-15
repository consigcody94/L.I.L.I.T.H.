"""Fit a small, inspectable anomaly regression against train-only climatology.

This is a trained statistical baseline, not a replacement for the Transformer or
a severe-event model. Select regularization on validation, evaluate tests once.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from training.verification_metrics import error_metrics


def features(arrays, variable):
    history = arrays["X"][:, :, variable]
    climate_first = arrays["climatology"][:, 0, variable]
    return np.stack([history[:, -1] - climate_first,
                     history[:, -3:].mean(axis=1) - climate_first,
                     history[:, -7:].mean(axis=1) - climate_first,
                     np.ones(len(history))], axis=1).astype(np.float64)


def forecast(arrays, coefficients):
    result = arrays["climatology"].astype(np.float64).copy()
    for variable in range(3):
        result[:, :, variable] += features(arrays, variable) @ coefficients[variable]
    # This product specifies nonnegative precipitation and ordered temperatures.
    # Score the exact constrained outputs that would be returned.
    result[:, :, 2] = np.maximum(0, result[:, :, 2])
    high, low = result[:, :, 0].copy(), result[:, :, 1].copy()
    result[:, :, 0], result[:, :, 1] = np.maximum(high, low), np.minimum(high, low)
    return result


def train(data_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    data = {name: dict(np.load(data_dir / f"{name}.npz", allow_pickle=False))
            for name in ("train", "validation", "test", "station_holdout")}
    candidates = []
    y_std = data["train"]["Y"].std(axis=(0, 1)) + 1e-6
    for alpha in (1.0, 100.0, 10000.0):
        coefficients = []
        for variable in range(3):
            x = features(data["train"], variable)
            anomaly = data["train"]["Y"][:, :, variable] - data["train"]["climatology"][:, :, variable]
            penalty = np.eye(x.shape[1]) * alpha
            coefficients.append(np.linalg.solve(x.T @ x + penalty, x.T @ anomaly))
        coefficients = np.asarray(coefficients)
        val_pred = forecast(data["validation"], coefficients)
        loss = float(np.square((val_pred - data["validation"]["Y"]) / y_std).mean())
        candidates.append((loss, alpha, coefficients))
    best_loss, alpha, coefficients = min(candidates, key=lambda c: c[0])
    np.savez_compressed(output / "coefficients.npz", coefficients=coefficients, alpha=alpha)
    evaluation = {}
    for name in ("validation", "test", "station_holdout"):
        arrays = data[name]
        prediction = forecast(arrays, coefficients)
        truth, climate = arrays["Y"], arrays["climatology"]
        persistence = np.repeat(arrays["X"][:, -1:, :], truth.shape[1], axis=1)
        model_mse = np.square(prediction - truth).mean(axis=(0, 1))
        climate_mse = np.square(climate - truth).mean(axis=(0, 1))
        score = {"model": error_metrics(prediction, truth),
                 "climatology": error_metrics(climate, truth),
                 "persistence": error_metrics(persistence, truth),
                 "mse_skill_vs_climatology": (1 - model_mse / climate_mse).tolist(), "stations": {}}
        for station in sorted(set(arrays["station_id"])):
            mask = arrays["station_id"] == station
            score["stations"][station] = {"model": error_metrics(prediction[mask], truth[mask]),
                                          "climatology": error_metrics(climate[mask], truth[mask])}
        evaluation[name] = score
        np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction, actual=truth,
                            climatology=climate, persistence=persistence,
                            station_id=arrays["station_id"], target_start=arrays["target_start"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": "empirical_climatology_anomaly_ridge",
        "status": "research_not_promoted", "selected_alpha": alpha, "validation_normalized_mse": best_loss,
        "selection_trials": [{"alpha": a, "validation_normalized_mse": loss} for loss, a, _ in candidates],
        "target_days": manifest["target_days"], "evaluation": evaluation,
        "checkpoint_sha256": hashlib.sha256((output / "coefficients.npz").read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
        "promotion": {"approved": False, "reason": "Research baseline; requires operational replay, regional review and uncertainty model"},
        "limitations": manifest["limitations"] + [
            "Uses empirical station normals; a global serving normals catalog is not implemented",
            "No calibrated probabilities or extreme-event classifier",
            "Benchmark reused after inspecting Transformer test results; fresh prospective audit needed"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"selected_alpha": alpha, "test_rmse": evaluation["test"]["model"]["rmse"],
                      "climatology_rmse": evaluation["test"]["climatology"]["rmse"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.data, args.output)
