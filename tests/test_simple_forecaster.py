"""Regression tests for SimpleForecaster.

The original simple_forecaster.forecast() ran the model, computed predictions,
then THREW THEM AWAY and returned (climatology + random noise) instead. This
file pins down the contract: forecasts must reflect the loaded checkpoint's
weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.simple_forecaster import SimpleForecaster
from models.simple_lilith import SimpleLILITH


@pytest.fixture
def constant_checkpoint(tmp_path):
    """Build a checkpoint whose weights produce a known, distinctive output.

    We bias the output projection so the model always returns large constant
    values — easy to detect in the forecast and impossible to mistake for
    climatology + noise.
    """
    model = SimpleLILITH(
        input_features=3,
        output_features=3,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        max_forecast=14,
    )
    # Override output bias so model emits a constant ~99 in normalized space.
    with torch.no_grad():
        for module in model.output_proj.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 3:
                module.weight.zero_()
                module.bias.fill_(99.0)

    norm = {
        "X_mean": [10.0, 0.0, 2.0],
        "X_std": [12.0, 10.0, 5.0],
        "Y_mean": [10.0, 0.0, 2.0],
        "Y_std": [12.0, 10.0, 5.0],
    }
    config = {
        "input_features": 3,
        "output_features": 3,
        "d_model": 32,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dropout": 0.0,
        "max_forecast": 14,
    }
    ckpt_path = tmp_path / "constant.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.0,
            "val_rmse": 0.0,
            "config": config,
            "normalization": norm,
        },
        ckpt_path,
    )
    return ckpt_path


def test_forecast_uses_model_output_not_climatology(constant_checkpoint):
    """If the model's output projection is biased to 99, the denormalized
    forecast should be roughly 99*Y_std + Y_mean = 99*12+10 ~= 1198 for tmax.

    The previous (buggy) code ignored model output and returned climatology
    around 5–25 °C. So if temperature_high is in any reasonable Earth range,
    the model output is being thrown away and we want to fail loudly.
    """
    fc = SimpleForecaster(str(constant_checkpoint), device="cpu")
    history = np.zeros((30, 3), dtype=np.float32)
    out = fc.forecast(latitude=40.0, longitude=-74.0, forecast_days=7, history=history)

    assert len(out["forecasts"]) == 7
    first = out["forecasts"][0]

    # Sanity: this is a wildly out-of-distribution number that proves the
    # model output is reaching the response. (~1198 °C — would only happen
    # if we're respecting the constant-99 model.)
    assert first["temperature_high"] > 500, (
        f"Got {first['temperature_high']} — looks like climatology blend, "
        "not model output. The 'use model predictions' fix has regressed."
    )


def test_forecast_no_synthetic_history_warning_path(constant_checkpoint):
    """Forecaster should fall back to synthetic history with a warning when
    no history is passed — and still produce a result, because callers should
    be able to do quick sanity calls without provisioning real data."""
    fc = SimpleForecaster(str(constant_checkpoint), device="cpu")
    out = fc.forecast(latitude=40.0, longitude=-74.0, forecast_days=3)
    assert len(out["forecasts"]) == 3
    assert out["bias_corrected"] is False


def test_forecast_ensemble_returns_uncertainty_bands(constant_checkpoint):
    """ensemble_samples > 0 should add uncertainty fields to each day."""
    fc = SimpleForecaster(str(constant_checkpoint), device="cpu")
    history = np.zeros((30, 3), dtype=np.float32)
    out = fc.forecast(
        latitude=40.0, longitude=-74.0, forecast_days=3,
        history=history, ensemble_samples=5,
    )
    assert out["ensemble_samples"] == 5
    first = out["forecasts"][0]
    assert "temperature_high_lower" in first
    assert "temperature_high_upper" in first
    assert first["temperature_high_lower"] <= first["temperature_high"]
    assert first["temperature_high_upper"] >= first["temperature_high"]


def test_bias_correction_is_opt_in(constant_checkpoint):
    """The default must be no-blend — climatology can only enter if explicitly
    enabled. Otherwise the original silent-override bug could come back as a
    one-line refactor."""
    fc = SimpleForecaster(str(constant_checkpoint), device="cpu")
    history = np.zeros((30, 3), dtype=np.float32)

    raw = fc.forecast(latitude=40.0, longitude=-74.0, forecast_days=2, history=history)
    blended = fc.forecast(
        latitude=40.0, longitude=-74.0, forecast_days=2,
        history=history, bias_correct_to_climatology=True,
    )
    # With constant-99 model, climatology blend should pull the value DOWN
    # toward Earth-realistic temps.
    assert raw["forecasts"][0]["temperature_high"] > blended["forecasts"][0]["temperature_high"]
    assert raw["bias_corrected"] is False
    assert blended["bias_corrected"] is True
