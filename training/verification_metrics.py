"""Physical-unit metrics shared by GPU and small statistical experiments."""

import numpy as np


def error_metrics(prediction, actual) -> dict:
    predicted, observed = np.asarray(prediction, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 3:
        raise ValueError("Expected matching [sample, lead, variable] arrays")
    error = predicted - observed
    if not np.isfinite(error).all() or not len(error):
        raise ValueError("Expected nonempty, finite arrays")
    return {"samples": len(error), "mae": np.abs(error).mean(axis=(0, 1)).tolist(),
            "rmse": np.sqrt(np.square(error).mean(axis=(0, 1))).tolist(),
            "bias": error.mean(axis=(0, 1)).tolist(),
            "mae_by_lead": np.abs(error).mean(axis=0).tolist(),
            "rmse_by_lead": np.sqrt(np.square(error).mean(axis=0)).tolist()}
