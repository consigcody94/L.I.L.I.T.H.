"""Inference utilities for LILITH."""

from inference.forecast import Forecaster
from inference.quantize import quantize_model

__all__ = ["Forecaster", "quantize_model"]
