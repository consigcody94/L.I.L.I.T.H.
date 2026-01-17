"""
Model Quantization for LILITH.

Provides utilities for quantizing models for efficient inference:
- Dynamic INT8 quantization
- Static INT8 quantization
- INT4 AWQ quantization
- ONNX export for TensorRT
"""

from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn as nn
from loguru import logger


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization.

    Quantizes weights to INT8 and computes activations in FP32.
    Good balance of speed and accuracy.

    Args:
        model: Model to quantize

    Returns:
        Quantized model
    """
    model_fp32 = model.cpu()

    # Quantize linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8,
    )

    logger.info("Applied dynamic INT8 quantization")
    return quantized_model


def quantize_static_int8(
    model: nn.Module,
    calibration_fn: Callable,
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    Apply static INT8 quantization with calibration.

    Uses calibration data to determine optimal quantization parameters.
    Better accuracy than dynamic quantization.

    Args:
        model: Model to quantize
        calibration_fn: Function that runs calibration batches through model
        num_calibration_batches: Number of batches for calibration

    Returns:
        Quantized model
    """
    model_fp32 = model.cpu()
    model_fp32.eval()

    # Prepare model for quantization
    model_fp32.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model_fp32, inplace=True)

    # Run calibration
    logger.info(f"Running calibration with {num_calibration_batches} batches...")
    calibration_fn(model_fp32, num_calibration_batches)

    # Convert to quantized model
    torch.quantization.convert(model_fp32, inplace=True)

    logger.info("Applied static INT8 quantization")
    return model_fp32


def quantize_int4_awq(
    model: nn.Module,
    calibration_data: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Apply INT4 AWQ (Activation-aware Weight Quantization).

    Aggressive quantization for memory-constrained deployment.
    Requires AWQ library.

    Args:
        model: Model to quantize
        calibration_data: Optional calibration data

    Returns:
        Quantized model
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        logger.warning("AWQ not available. Install with: pip install autoawq")
        return model

    # AWQ quantization would go here
    # This is a placeholder - actual implementation depends on AWQ API
    logger.warning("INT4 AWQ quantization not yet implemented")
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: dict,
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
):
    """
    Export model to ONNX format.

    Can be further optimized with TensorRT.

    Args:
        model: Model to export
        output_path: Path to save ONNX model
        input_shapes: Dictionary of input names to shapes
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size
    """
    model = model.cpu()
    model.eval()

    # Create dummy inputs
    dummy_inputs = {}
    for name, shape in input_shapes.items():
        dummy_inputs[name] = torch.randn(*shape)

    # Export
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        output_path,
        input_names=list(input_shapes.keys()),
        output_names=["forecast"],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Exported ONNX model to {output_path}")


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Apply general inference optimizations.

    Includes:
    - Fusing batch norm with conv
    - Removing dropout
    - Setting eval mode

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    model.eval()

    # Fuse modules where possible
    try:
        model = torch.jit.optimize_for_inference(torch.jit.script(model))
        logger.info("Applied JIT optimization")
    except Exception as e:
        logger.warning(f"JIT optimization failed: {e}")

    return model


def quantize_model(
    model: nn.Module,
    method: str = "dynamic_int8",
    calibration_fn: Optional[Callable] = None,
    **kwargs,
) -> nn.Module:
    """
    Quantize model using specified method.

    Args:
        model: Model to quantize
        method: Quantization method ("dynamic_int8", "static_int8", "int4_awq")
        calibration_fn: Calibration function (for static methods)
        **kwargs: Additional arguments for specific methods

    Returns:
        Quantized model
    """
    if method == "dynamic_int8":
        return quantize_dynamic_int8(model)
    elif method == "static_int8":
        if calibration_fn is None:
            raise ValueError("calibration_fn required for static INT8 quantization")
        return quantize_static_int8(model, calibration_fn, **kwargs)
    elif method == "int4_awq":
        return quantize_int4_awq(model, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def estimate_memory_usage(model: nn.Module, precision: str = "fp32") -> dict:
    """
    Estimate model memory usage.

    Args:
        model: Model to analyze
        precision: Precision ("fp32", "fp16", "int8", "int4")

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Bytes per parameter
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    param_bytes = total_params * bytes_per_param.get(precision, 4)

    # Estimate activation memory (rough estimate: 2x params for batch=1)
    activation_bytes = param_bytes * 2

    return {
        "total_parameters": total_params,
        "precision": precision,
        "parameter_memory_mb": param_bytes / (1024 * 1024),
        "estimated_activation_memory_mb": activation_bytes / (1024 * 1024),
        "estimated_total_memory_mb": (param_bytes + activation_bytes) / (1024 * 1024),
    }
