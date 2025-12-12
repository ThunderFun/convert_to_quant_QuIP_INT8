"""
convert_to_quant - Quantization toolkit for neural network weights.

Supports FP8, INT8 block-wise, NF4, and FP4 quantization formats
with learned rounding optimization for minimal accuracy loss.
"""

__version__ = "0.1.0"

from .convert_to_quant import (
    LearnedRoundingConverter,
    convert_to_fp8_scaled,
    main,
)

__all__ = [
    "__version__",
    "LearnedRoundingConverter",
    "convert_to_fp8_scaled",
    "main",
]
