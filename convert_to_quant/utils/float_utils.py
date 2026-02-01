"""Float utilities for INT8 quantization.

Provides axis-wise and tensor-wise quantization utilities.
"""
import torch
from typing import Optional


def quantize_int8_axiswise(
    x: torch.Tensor, 
    dim: int = -1,
    outlier_percentile: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Axis-wise (per-row/per-token) INT8 quantization.

    Args:
        x: Input tensor
        dim: Dimension to quantize along (default -1 for per-row)
        outlier_percentile: If provided (0.5-1.0), use percentile instead of max
                           to compute scale (e.g., 0.999 for 99.9th percentile).
                           This ignores extreme outliers for better quantization.

    Returns:
        Tuple of (quantized_int8, scale)
    """
    if outlier_percentile is not None and 0.5 < outlier_percentile < 1.0:
        # Use percentile to ignore extreme outliers
        # quantile() doesn't support BFloat16, so we must convert to float
        abs_max = x.abs().float().quantile(outlier_percentile, dim=dim, keepdim=True).to(x.dtype)
    else:
        # Standard max-based scaling
        abs_max = x.abs().amax(dim=dim, keepdim=True)
    
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_int8_tensorwise(
    x: torch.Tensor,
    outlier_percentile: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tensor-wise INT8 quantization.

    Args:
        x: Input tensor
        outlier_percentile: If provided (0.5-1.0), use percentile instead of max
                           to compute scale (e.g., 0.999 for 99.9th percentile).
                           This ignores extreme outliers for better quantization.

    Returns:
        Tuple of (quantized_int8, scale)
    """
    if outlier_percentile is not None and 0.5 < outlier_percentile < 1.0:
        # Use percentile to ignore extreme outliers
        # quantile() doesn't support BFloat16, so we must convert to float
        abs_max = x.abs().float().quantile(outlier_percentile).to(x.dtype)
    else:
        # Standard max-based scaling
        abs_max = x.abs().max()
    
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale
