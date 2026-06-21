"""Inference utilities for LILITH.

Heavy entry points (``Forecaster``, ``quantize_model``) pull in pandas/torch, so
they are imported lazily. This keeps lightweight modules - e.g. ``forecast_blend``
(NumPy only) and the keyless data providers - importable without the full ML stack.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type-checkers / IDEs only; not executed at runtime
    from inference.forecast import Forecaster
    from inference.forecast_blend import blend_forecast
    from inference.quantize import quantize_model
    from inference.simple_forecaster import SimpleForecaster

__all__ = ["Forecaster", "quantize_model", "SimpleForecaster", "blend_forecast"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "Forecaster":
        from inference.forecast import Forecaster

        return Forecaster
    if name == "quantize_model":
        from inference.quantize import quantize_model

        return quantize_model
    if name == "SimpleForecaster":
        from inference.simple_forecaster import SimpleForecaster

        return SimpleForecaster
    if name == "blend_forecast":
        from inference.forecast_blend import blend_forecast

        return blend_forecast
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
