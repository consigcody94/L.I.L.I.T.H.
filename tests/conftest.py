"""Shared fixtures for the LILITH test suite.

Provides a session-scoped tiny SimpleLILITH checkpoint so the API tests
exercise the real inference pipeline end-to-end instead of falling through
to the 503 'no model loaded' branch. Without this fixture the API tests
were effectively only testing FastAPI's input validation — the model code
itself was never hit by the API integration tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.simple_lilith import SimpleLILITH


def _build_tiny_checkpoint(path: Path) -> None:
    """Construct a minimal but real SimpleLILITH checkpoint.

    Has to match the schema SimpleForecaster.load expects: ``config``,
    ``normalization``, and ``model_state_dict``. The model is intentionally
    untrained — we're only verifying the inference path runs, not skill.
    """
    model = SimpleLILITH(
        input_features=3,
        output_features=3,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
        max_forecast=90,
    )
    # Climatologically-plausible normalization stats (tenths-of-°C TMAX/TMIN/PRCP).
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
        "dropout": 0.1,
        "max_forecast": 90,
    }
    ckpt = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "val_loss": 1.0,
        "val_rmse": 4.0,  # plausible for an untrained tiny model
        "config": config,
        "normalization": norm,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


@pytest.fixture(scope="session", autouse=True)
def tiny_test_checkpoint(tmp_path_factory):
    """Create the tiny checkpoint and expose it via the env var SimpleForecaster reads."""
    ckpt_path = tmp_path_factory.mktemp("checkpoints") / "lilith_test.pt"
    _build_tiny_checkpoint(ckpt_path)

    # Persist the env var for the entire pytest session — the API lifespan
    # reads LILITH_CHECKPOINT at startup.
    prev = os.environ.get("LILITH_CHECKPOINT")
    os.environ["LILITH_CHECKPOINT"] = str(ckpt_path)
    try:
        yield ckpt_path
    finally:
        if prev is None:
            os.environ.pop("LILITH_CHECKPOINT", None)
        else:
            os.environ["LILITH_CHECKPOINT"] = prev


@pytest.fixture
def client(tiny_test_checkpoint):  # noqa: ARG001 — fixture establishes env var
    """FastAPI TestClient that actually runs the lifespan startup/shutdown.

    The default TestClient(app) does not trigger lifespan events; using it
    as a context manager does. Without this, the API's _forecaster stays
    None and every forecast request returns 503. This fixture overrides
    the per-test client fixture defined in test_api.py.
    """
    from fastapi.testclient import TestClient
    from web.api.main import app

    with TestClient(app) as c:
        yield c
