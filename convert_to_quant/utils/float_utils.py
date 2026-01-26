"""
Float utilities for INT8 quantization.

Provides axis-wise and tensor-wise quantization utilities.
"""
import torch

def quantize_int8_axiswise(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Axis-wise (per-row/per-token) INT8 quantization.

    Args:
        x: Input tensor
        dim: Dimension to quantize along (default -1 for per-row)

    Returns:
        Tuple of (quantized_int8, scale)
    """
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_int8_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tensor-wise INT8 quantization.

    Args:
        x: Input tensor

    Returns:
        Tuple of (quantized_int8, scale)
    """
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale
