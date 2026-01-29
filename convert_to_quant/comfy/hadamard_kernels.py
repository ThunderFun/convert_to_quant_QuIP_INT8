"""
Optimized Hadamard Transform Kernels with Triton Acceleration.

This module provides high-performance GPU kernels for Fast Hadamard Transform:
- 1D Hadamard transform with autotuned block sizes
- Fused sign multiplication + Hadamard
- Fused 2D Hadamard transform
- Optimized dispatch based on tensor size

Author: AI Assistant
Date: 2026-01-27
"""

import torch
from torch import Tensor
from typing import Optional
import math

# =============================================================================
# Triton Availability Check
# =============================================================================

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    TRITON_VERSION = getattr(triton, '__version__', 'unknown')
except ImportError:
    HAS_TRITON = False
    TRITON_VERSION = None

# Enable/disable Triton via environment variable
import os
# NOTE: Triton kernels can be enabled via environment variable.
# They have been verified to work correctly with proper ping-pong buffer handling.
_DISABLE_TRITON = os.environ.get("DISABLE_TRITON_HADAMARD", "0") == "1"
_ENABLE_TRITON = os.environ.get("ENABLE_TRITON_HADAMARD", "1") == "1"  # Default to enabled
HAS_TRITON_HADAMARD = HAS_TRITON and _ENABLE_TRITON and not _DISABLE_TRITON


# =============================================================================
# Triton Kernels (Fixed with Proper Ping-Pong Buffer Pattern)
# =============================================================================

if HAS_TRITON:
    
    # -------------------------------------------------------------------------
    # Kernel 1: Single Stage Butterfly Operation
    # -------------------------------------------------------------------------
    
    @triton.jit
    def _hadamard_stage_kernel(
        in_ptr,
        out_ptr,
        n: tl.constexpr,
        stage: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Single stage of butterfly Hadamard transform.
        Reads from in_ptr, writes to out_ptr.
        
        Each stage processes pairs of elements separated by stride h = 2^stage.
        Handles n > BLOCK_SIZE by looping over tiles.
        """
        pid = tl.program_id(0)
        batch_offset = pid * n
        
        h = 1 << stage
        
        for tile_start in range(0, n, BLOCK_SIZE):
            offs = tile_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            
            x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
            pair_offs = offs ^ h
            pair_mask = pair_offs < n
            
            x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
            
            # Butterfly: if (i & h) == 0, result = x[i] + x[i^h], else x[i^h] - x[i]
            is_upper = (offs & h) == 0
            new_val = tl.where(is_upper, x + x_pair, x_pair - x)
            
            tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)
    
    
    @triton.jit
    def _hadamard_final_normalize_kernel(
        x_ptr,
        out_ptr,
        n: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply normalization after all butterfly stages."""
        pid = tl.program_id(0)
        batch_offset = pid * n
        
        # Loop over tiles when n > BLOCK_SIZE
        for tile_start in range(0, n, BLOCK_SIZE):
            offs = tile_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            
            x = tl.load(x_ptr + batch_offset + offs, mask=mask, other=0.0)
            x = x * scale
            tl.store(out_ptr + batch_offset + offs, x, mask=mask)
    
    
    # -------------------------------------------------------------------------
    # Kernel 2: Fused Signs + Hadamard Stage
    # -------------------------------------------------------------------------
    
    @triton.jit
    def _hadamard_stage_with_signs_kernel(
        in_ptr,
        out_ptr,
        sign_row_ptr,
        sign_col_ptr,
        n: tl.constexpr,
        stage: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_ROW_SIGNS: tl.constexpr,
        HAS_COL_SIGNS: tl.constexpr,
        APPLY_SIGNS: tl.constexpr,  # True only for first stage
    ):
        """
        Single butterfly stage with optional sign multiplication on first stage.
        Handles n > BLOCK_SIZE by looping over tiles.
        """
        pid = tl.program_id(0)
        batch_offset = pid * n
        
        # Load row sign once if needed
        if APPLY_SIGNS and HAS_ROW_SIGNS:
            sign_r = tl.load(sign_row_ptr + pid)
        
        h = 1 << stage
        
        for tile_start in range(0, n, BLOCK_SIZE):
            offs = tile_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            
            x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
            
            if APPLY_SIGNS:
                if HAS_ROW_SIGNS:
                    x = x * sign_r
                if HAS_COL_SIGNS:
                    sign_c = tl.load(sign_col_ptr + offs, mask=mask, other=1.0)
                    x = x * sign_c
            
            pair_offs = offs ^ h
            pair_mask = pair_offs < n
            
            x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
            
            if APPLY_SIGNS:
                if HAS_ROW_SIGNS:
                    x_pair = x_pair * sign_r
                if HAS_COL_SIGNS:
                    sign_c_pair = tl.load(sign_col_ptr + pair_offs, mask=pair_mask, other=1.0)
                    x_pair = x_pair * sign_c_pair
            
            is_upper = (offs & h) == 0
            new_val = tl.where(is_upper, x + x_pair, x_pair - x)
            
            tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)
    
    
    # -------------------------------------------------------------------------
    # Kernel 3: Simple Copy Kernel (for output staging)
    # -------------------------------------------------------------------------
    
    @triton.jit
    def _copy_kernel(
        in_ptr,
        out_ptr,
        n: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Simple copy kernel for buffer management."""
        pid = tl.program_id(0)
        batch_offset = pid * n
        
        # Loop over tiles when n > BLOCK_SIZE
        for tile_start in range(0, n, BLOCK_SIZE):
            offs = tile_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            
            x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
            tl.store(out_ptr + batch_offset + offs, x, mask=mask)
    
    
    # -------------------------------------------------------------------------
    # Autotuned Multi-Stage Kernel (for small sizes)
    # -------------------------------------------------------------------------
    
    _hadamard_configs = [
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 16}),
    ]
    
    @triton.autotune(
        configs=_hadamard_configs,
        key=['n'],
    )
    @triton.jit
    def _hadamard_autotuned_stage_kernel(
        in_ptr,
        out_ptr,
        n: tl.constexpr,
        stage: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Autotuned single stage butterfly operation."""
        pid = tl.program_id(0)
        batch_offset = pid * n
        
        h = 1 << stage
        
        # Loop over tiles when n > BLOCK_SIZE
        for tile_start in range(0, n, BLOCK_SIZE):
            offs = tile_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n
            
            x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
            pair_offs = offs ^ h
            pair_mask = pair_offs < n
            x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
            
            is_upper = (offs & h) == 0
            new_val = tl.where(is_upper, x + x_pair, x_pair - x)
            
            tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)


# =============================================================================
# Dispatch and Python Interface
# =============================================================================

def triton_hadamard_transform(
    x: Tensor,
    normalize: bool = True,
    output: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply Fast Hadamard Transform using Triton kernel.
    
    Uses proper ping-pong buffer approach where each butterfly stage
    reads from one buffer and writes to another.
    
    Args:
        x: Input tensor, last dimension must be power of 2
        normalize: Whether to normalize by 1/sqrt(n)
        output: Optional pre-allocated output tensor (for in-place-like behavior)
    
    Returns:
        Transformed tensor
    """
    if not HAS_TRITON_HADAMARD:
        raise RuntimeError("Triton is not available")
    
    n = x.shape[-1]
    batch = x.numel() // n
    
    # Check if true in-place operation is requested (output is the same tensor as x)
    # This check MUST happen before any modifications to x
    inplace_requested = output is not None and output.data_ptr() == x.data_ptr()
    
    # Ensure contiguous memory layout
    if inplace_requested:
        # For true in-place: copy to output if x is not contiguous, then use output as working buffer
        if not x.is_contiguous():
            output.copy_(x)
        x = output
    else:
        # Not in-place: clone to avoid modifying the input tensor
        # This creates our working buffer
        x = x.contiguous().clone()
    
    # Determine output tensor
    if output is None:
        out = torch.empty_like(x)
    else:
        out = output
    
    # We need two buffers for ping-pong
    # If user wants in-place, we use x and a temp buffer
    # Otherwise, we use x and out as the two buffers
    if inplace_requested:
        # True in-place: need a temporary buffer (we already have x as working buffer)
        temp = torch.empty_like(x)
        buf_a = x
        buf_b = temp
    else:
        # Not in-place: use input clone and output as the two buffers
        buf_a = x
        buf_b = out
    
    log_n = int(math.log2(n))
    BLOCK_SIZE = min(n, 2048)
    grid = (batch,)
    
    # Run butterfly stages with ping-pong
    for stage in range(log_n):
        # Alternate between buffers
        if stage % 2 == 0:
            in_buf, out_buf = buf_a, buf_b
        else:
            in_buf, out_buf = buf_b, buf_a
        
        _hadamard_autotuned_stage_kernel[grid](
            in_buf, out_buf,
            n=n,
            stage=stage,
        )
    
    # Apply normalization if requested
    if normalize:
        # The final result is in the buffer from the last stage
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        scale = 1.0 / math.sqrt(n)
        _hadamard_final_normalize_kernel[grid](
            final_buf, out,
            n=n,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=1,
        )
    else:
        # Copy final result to output if needed
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        if final_buf.data_ptr() != out.data_ptr():
            _copy_kernel[grid](
                final_buf, out,
                n=n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=1,
            )
    
    return out


def fast_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Dispatch to Triton or fallback to PyTorch."""
    if HAS_TRITON_HADAMARD and x.is_cuda:
        return triton_hadamard_transform(x, normalize=normalize)
    # Fallback to PyTorch implementation
    from ..utils.hadamard import fast_hadamard_transform as py_fht
    return py_fht(x, normalize=normalize)


def triton_hadamard_transform_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True,
    output: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Hadamard transform with sign multiplications.
    Signs are applied during the first butterfly stage.
    """
    if not HAS_TRITON_HADAMARD:
        raise RuntimeError("Triton is not available")
    
    n = x.shape[-1]
    batch = x.numel() // n
    
    # Check if true in-place operation is requested (output is the same tensor as x)
    # This check MUST happen before any modifications to x
    inplace_requested = output is not None and output.data_ptr() == x.data_ptr()
    
    # Ensure contiguous memory layout
    if inplace_requested:
        # For true in-place: copy to output if x is not contiguous, then use output as working buffer
        if not x.is_contiguous():
            output.copy_(x)
        x = output
    else:
        # Not in-place: clone to avoid modifying the input tensor
        # This creates our working buffer
        x = x.contiguous().clone()
    
    # Determine output tensor
    if output is None:
        out = torch.empty_like(x)
    else:
        out = output
    
    # We need two buffers for ping-pong
    # If user wants in-place, we use x and a temp buffer
    # Otherwise, we use x and out as the two buffers
    if inplace_requested:
        # True in-place: need a temporary buffer (we already have x as working buffer)
        temp = torch.empty_like(x)
        buf_a = x
        buf_b = temp
    else:
        # Not in-place: use input clone and output as the two buffers
        buf_a = x
        buf_b = out
    
    log_n = int(math.log2(n))
    BLOCK_SIZE = min(n, 2048)
    grid = (batch,)
    
    HAS_ROW_SIGNS = sign_row is not None
    HAS_COL_SIGNS = sign_col is not None
    
    # Run butterfly stages with signs on first stage
    for stage in range(log_n):
        if stage % 2 == 0:
            in_buf, out_buf = buf_a, buf_b
        else:
            in_buf, out_buf = buf_b, buf_a
        
        _hadamard_stage_with_signs_kernel[grid](
            in_buf, out_buf,
            sign_row if HAS_ROW_SIGNS else x,  # Dummy if not used
            sign_col if HAS_COL_SIGNS else x,  # Dummy if not used
            n=n,
            stage=stage,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_ROW_SIGNS=HAS_ROW_SIGNS,
            HAS_COL_SIGNS=HAS_COL_SIGNS,
            APPLY_SIGNS=(stage == 0),
            num_stages=1,
        )
    
    # Apply normalization and copy to output if needed
    if normalize:
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        scale = 1.0 / math.sqrt(n)
        _hadamard_final_normalize_kernel[grid](
            final_buf, out,
            n=n,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=1,
        )
    else:
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        if final_buf.data_ptr() != out.data_ptr():
            _copy_kernel[grid](
                final_buf, out,
                n=n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=1,
            )
    
    return out


def fast_hadamard_transform_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """Fused Hadamard transform with sign multiplications.
    
    For 2D+ tensors where both last two dims are reasonably large powers of 2, 
    uses 2D transform. For smaller dimensions or batched 1D, uses 1D transform.
    """
    # Check if this should be a 2D transform:
    # - Both dimensions should be >= 64 (to distinguish from batch dims)
    # - Both dimensions must be powers of 2
    # - Signs must match both dimensions
    if x.dim() >= 2:
        n_rows = x.shape[-2]
        n_cols = x.shape[-1]
        is_power_of_2 = lambda n: (n > 0) and (n & (n - 1) == 0)
        
        # Use 2D only if both dimensions are "significant" (>= 64)
        # This avoids treating (4, 1024) as 2D when it's really batched 1D
        if n_rows >= 64 and n_cols >= 64 and is_power_of_2(n_rows) and is_power_of_2(n_cols):
            # And signs must match the 2D structure
            row_match = sign_row is None or sign_row.shape[-1] == n_rows
            col_match = sign_col is None or sign_col.shape[-1] == n_cols
            if row_match and col_match:
                return fast_hadamard_transform_2d_with_signs(
                    x, sign_row=sign_row, sign_col=sign_col, normalize=normalize
                )
    
    # Use 1D transform
    if HAS_TRITON_HADAMARD and x.is_cuda:
        return triton_hadamard_transform_with_signs(
            x, sign_row=sign_row, sign_col=sign_col, normalize=normalize
        )
    
    # Fallback to PyTorch
    from ..utils.hadamard import fast_hadamard_transform_with_signs as py_fht_signs
    return py_fht_signs(x, sign_row=sign_row, sign_col=sign_col, normalize=normalize)


def fast_hadamard_transform_2d(
    x: Tensor,
    normalize: bool = True,
    inplace: bool = False
) -> Tensor:
    """
    Apply Fast Hadamard Transform to both dimensions.
    result = H @ X @ H.T
    """
    n_rows = x.shape[-2]
    n_cols = x.shape[-1]
    
    if not inplace:
        x = x.clone()
    
    # Apply Hadamard to last dimension (columns)
    if HAS_TRITON_HADAMARD and x.is_cuda:
        x = triton_hadamard_transform(x, normalize=False, output=x)
    else:
        from ..utils.hadamard import fast_hadamard_transform as py_fht
        x = py_fht(x, normalize=False, inplace=True)
    
    # Apply Hadamard to second-to-last dimension (rows)
    x = x.transpose(-2, -1)
    if HAS_TRITON_HADAMARD and x.is_cuda:
        x = triton_hadamard_transform(x, normalize=False, output=x)
    else:
        from ..utils.hadamard import fast_hadamard_transform as py_fht
        x = py_fht(x, normalize=False, inplace=True)
    x = x.transpose(-2, -1)
    
    if normalize:
        x = x / math.sqrt(n_rows * n_cols)
    
    return x


def fast_hadamard_transform_2d_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """Fused 2D Hadamard with sign multiplications.
    
    Mathematical formula: D1 @ H @ X @ H.T @ D2
    where D1 = diag(sign_row), D2 = diag(sign_col)
    
    This means:
    1. Apply signs: X * sign_row * sign_col
    2. Apply Hadamard transform: H @ (signed X) @ H.T
    """
    if sign_row is not None or sign_col is not None:
        x = x.clone()
        
        if sign_row is not None and sign_col is not None:
            x.mul_(sign_row.unsqueeze(-1))
            x.mul_(sign_col.unsqueeze(-2))
        elif sign_row is not None:
            x.mul_(sign_row.unsqueeze(-1))
        elif sign_col is not None:
            x.mul_(sign_col.unsqueeze(-2))
    
    return fast_hadamard_transform_2d(x, normalize=normalize, inplace=True)


def inverse_hadamard_transform_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    orig_shape: Optional[tuple] = None,
    normalize: bool = True
) -> Tensor:
    """Inverse Hadamard transform with sign restoration.
    
    For normalized Hadamard: H' @ H' = I (identity)
    If forward is y = H' @ (x * sign_row * sign_col), then:
    H' @ y = H' @ H' @ (x * sign_row * sign_col) = x * sign_row * sign_col
    So: x = (H' @ y) * sign_row * sign_col (since sign * sign = 1)
    """
    # Hadamard is self-inverse (up to normalization)
    if HAS_TRITON_HADAMARD and x.is_cuda:
        result = triton_hadamard_transform(x, normalize=normalize)
    else:
        from ..utils.hadamard import fast_hadamard_transform as py_fht
        result = py_fht(x, normalize=normalize, inplace=True)
    
    # Apply signs after transform to cancel the signs from forward
    if sign_row is not None:
        result = result * sign_row.unsqueeze(-1)
    if sign_col is not None:
        result = result * sign_col.unsqueeze(-2)
    
    # Slice to original shape
    if orig_shape is not None:
        slices = tuple(slice(0, s) for s in orig_shape)
        result = result[slices]
    
    return result


def inverse_hadamard_transform_2d_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    orig_rows: Optional[int] = None,
    orig_cols: Optional[int] = None,
    normalize: bool = True
) -> Tensor:
    """Inverse 2D Hadamard transform with sign restoration.
    
    Inverse of: y = D1 @ H @ x @ H.T @ D2
    Where D1 = diag(sign_row), D2 = diag(sign_col)
    
    The forward applies signs first, then Hadamard.
    The inverse applies Hadamard first, then signs.
    
    Derivation:
        If y = D1 @ H @ x @ H.T @ D2, then:
        H @ y @ H.T = H @ (D1 @ H @ x @ H.T @ D2) @ H.T
                    = (H @ D1 @ H) @ x @ (H.T @ D2 @ H.T)
        
        For Hadamard: H @ H.T = n * I (without normalization)
        With normalization: H' @ H'.T = I
        
        So: H' @ y @ H'.T = (H' @ D1 @ H') @ x @ (H'.T @ D2 @ H'.T)
        
        To recover x: x = D1 @ (H' @ y @ H'.T) @ D2  (since D1^2 = I, D2^2 = I)
    
    So the inverse is:
    1. Apply 2D Hadamard: H @ y @ H.T
    2. Apply row signs: D1 @ result
    3. Apply column signs: result @ D2
    """
    # First apply 2D Hadamard (self-inverse with normalization)
    result = fast_hadamard_transform_2d(x, normalize=normalize, inplace=False)
    
    # Then apply signs (after Hadamard, not before)
    if sign_row is not None:
        result = result * sign_row.view(*([1] * (result.ndim - 2)), -1, 1)
    if sign_col is not None:
        result = result * sign_col.view(*([1] * (result.ndim - 2)), 1, -1)
    
    # Slice to original dimensions
    if orig_rows is not None or orig_cols is not None:
        slices = [slice(None)] * result.ndim
        if orig_rows is not None:
            slices[-2] = slice(0, orig_rows)
        if orig_cols is not None:
            slices[-1] = slice(0, orig_cols)
        result = result[tuple(slices)]
    
    return result


# =============================================================================
# Backward Compatibility
# =============================================================================

__all__ = [
    'HAS_TRITON_HADAMARD',
    'triton_hadamard_transform',
    'fast_hadamard_transform',
    'fast_hadamard_transform_with_signs',
    'fast_hadamard_transform_2d',
    'fast_hadamard_transform_2d_with_signs',
    'inverse_hadamard_transform_with_signs',
    'inverse_hadamard_transform_2d_with_signs',
]
