import torch
from torch import Tensor
import math
from typing import Optional, Dict
import os

# Fast Hadamard Transform (FHT) implementations:
# - Optimized PyTorch fallback with reduced allocations
# - Optional torch.compile with static shapes
# - Triton kernels in convert_to_quant.comfy.hadamard_kernels
# - Mixed-precision (BF16/FP16) for bandwidth reduction on Ampere+

_ENABLE_MIXED_PRECISION = os.environ.get("ENABLE_MIXED_HADAMARD", "auto")
_MIXED_PRECISION_DTYPE = os.environ.get("MIXED_HADAMARD_DTYPE", "bfloat16")
_LOW_MEMORY_MODE = False


def set_low_memory_mode(enabled: bool = True):
    """Set low memory mode flag. Called by quantization code when --low-memory is used."""
    global _LOW_MEMORY_MODE
    _LOW_MEMORY_MODE = enabled


def _should_use_mixed_precision() -> bool:
    """
    Determine if mixed precision should be used based on environment and hardware.
    
    Mixed precision is primarily beneficial in low_memory mode where the memory
    savings (50% bandwidth reduction) outweigh the performance cost.
    
    Now also checks BF16_COMPUTE_MODE for global BF16 configuration.
    
    Returns:
        True if mixed precision is enabled and supported
    """
    # Check BF16_COMPUTE_MODE first (new global configuration)
    bf16_mode = os.environ.get("BF16_COMPUTE_MODE", "auto")
    
    if bf16_mode in ("0", "off", "false"):
        return False
    
    if bf16_mode in ("1", "on", "true", "force"):
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return major >= 8  # Ampere or newer
        return False
    
    # Original mixed precision logic
    if _ENABLE_MIXED_PRECISION == "0":
        return False
    
    if _ENABLE_MIXED_PRECISION == "1":
        return True
    
    # Auto mode: enable only in low_memory mode on supported hardware
    if _ENABLE_MIXED_PRECISION == "auto":
        if not _LOW_MEMORY_MODE:
            # Not in low memory mode - use standard FP32 (faster with Triton)
            return False
        
        if torch.cuda.is_available():
            # Ampere (SM80+) and newer have native BF16 support
            major, minor = torch.cuda.get_device_capability()
            sm = major * 10 + minor
            return sm >= 80  # Ampere or newer
    
    return False


def _get_mixed_precision_dtype() -> torch.dtype:
    """Get the compute dtype for mixed precision from environment."""
    dtype_str = os.environ.get("MIXED_HADAMARD_DTYPE", "bfloat16").lower()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    else:
        return torch.float16


# Utility Functions

def is_power_of_two(n: int) -> bool:
    return (n > 0) and (n & (n - 1) == 0)


def next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 2**(n - 1).bit_length()


# Optimized PyTorch Implementation

def _fast_hadamard_transform_core_v1(x: Tensor, n: int, normalize: bool = True) -> Tensor:
    """Original core implementation using torch.stack (baseline for comparison)."""
    original_shape = x.shape
    x = x.reshape(-1, n)
    
    h = 1
    while h < n:
        x_view = x.view(-1, n // (h * 2), 2, h)
        a = x_view[:, :, 0, :]
        b = x_view[:, :, 1, :]
        sum_ab = a + b
        diff_ab = a - b
        x = torch.stack([sum_ab, diff_ab], dim=2).view(-1, n // h, h)
        h *= 2
    
    x = x.reshape(original_shape)
    if normalize:
        x = x / math.sqrt(n)
    return x


def _fast_hadamard_transform_core_optimized(x: Tensor, n: int, normalize: bool = True) -> Tensor:
    """
    OPTIMIZED core Hadamard transform implementation.
    
    Improvements over v1:
    - Pre-allocates output buffer (single allocation)
    - Uses in-place operations where possible
    - Reduces memory traffic by avoiding intermediate views
    - Uses contiguous memory access patterns
    """
    original_shape = x.shape
    x = x.reshape(-1, n)
    batch = x.shape[0]
    
    # Pre-allocate working buffer to avoid repeated allocations
    output = torch.empty_like(x)
    
    h = 1
    while h < n:
        half = n // (2 * h)
        
        # Reshape for butterfly operations
        x_view = x.view(batch, half, 2, h)
        out_view = output.view(batch, half, 2, h)
        
        # Butterfly: (a, b) -> (a+b, a-b)
        # Note: we clone 'a' to avoid modifying input during first iteration
        a = x_view[:, :, 0, :].contiguous()
        b = x_view[:, :, 1, :]
        
        # Butterfly operations in FP32 for speed
        # Note: Previously used FP64 on CPU for numerical stability, but
        # FP32 is significantly faster (~2x) and sufficient for inference
        out_view[:, :, 0, :] = a + b
        out_view[:, :, 1, :] = a - b
        
        # Swap input/output for next iteration (ping-pong)
        x, output = output, x
        h *= 2
    
    result = x.reshape(original_shape)
    if normalize:
        # In-place normalization
        result.mul_(1.0 / math.sqrt(n))
    
    return result


def _fast_hadamard_transform_core_inplace(x: Tensor, n: int, normalize: bool = True) -> Tensor:
    """
    In-place Hadamard transform - modifies input tensor.
    Zero extra allocations during transform.
    """
    original_shape = x.shape
    x = x.reshape(-1, n)
    batch = x.shape[0]
    
    h = 1
    while h < n:
        half = n // (2 * h)
        x_view = x.view(batch, half, 2, h)
        
        a = x_view[:, :, 0, :].clone()
        b = x_view[:, :, 1, :]
        
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        
        h *= 2
    
    x = x.reshape(original_shape)
    if normalize:
        x.mul_(1.0 / math.sqrt(n))
    return x


def _fast_hadamard_transform_core_mixed(
    x: Tensor,
    n: int,
    normalize: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """
    Mixed-precision Hadamard transform core.
    
    Uses BF16/FP16 for storage/bandwidth, FP32 for computation.
    This reduces memory bandwidth while maintaining precision.
    
    Args:
        x: Input tensor (FP32)
        n: Size of Hadamard transform (power of 2)
        normalize: Whether to normalize by 1/sqrt(n)
        compute_dtype: Lower precision dtype (BF16 or FP16)
    
    Returns:
        Transformed tensor in FP32
    """
    original_shape = x.shape
    x = x.reshape(-1, n)
    batch = x.shape[0]
    
    # Cast input to lower precision once and keep it there
    # This reduces memory bandwidth and avoids repeated casting
    x_low = x.to(compute_dtype)
    
    # Pre-allocate working buffer in lower precision for ping-pong
    temp = torch.empty_like(x_low)
    
    h = 1
    while h < n:
        half = n // (2 * h)
        
        # Reshape for butterfly
        x_view = x_low.view(batch, half, 2, h)
        temp_view = temp.view(batch, half, 2, h)
        
        # Butterfly in FP32 for precision, store back in lower precision
        a = x_view[:, :, 0, :].float()
        b = x_view[:, :, 1, :].float()
        
        temp_view[:, :, 0, :] = (a + b).to(compute_dtype)
        temp_view[:, :, 1, :] = (a - b).to(compute_dtype)
        
        # Swap buffers for next iteration
        x_low, temp = temp, x_low
        h *= 2
    
    # Final result: convert back to FP32
    result = x_low.float().reshape(original_shape)
    
    # DEBUG: Check for NaNs or zeros
    if torch.isnan(result).any():
        print(f"      [DEBUG] WARNING: NaNs detected in mixed-precision Hadamard transform (n={n})")
    if (result == 0).all() and n > 1:
        print(f"      [DEBUG] WARNING: All zeros detected in mixed-precision Hadamard transform (n={n})")
        
    if normalize:
        result.mul_(1.0 / math.sqrt(n))
    
    return result


# =============================================================================
# Static-Shape torch.compile Cache (Phase 1)
# =============================================================================

_compiled_transforms: Dict[int, callable] = {}
_DISABLE_COMPILE = os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1"
_ENABLE_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "1") == "1"  # Changed default to enabled


def _get_compiled_transform(n: int):
    """
    Get or create a torch.compile'd transform for size n.
    Uses static shapes to avoid dynamic shape compilation issues.
    """
    global _compiled_transforms
    
    if n in _compiled_transforms:
        return _compiled_transforms[n]
    
    if not _ENABLE_COMPILE or _DISABLE_COMPILE:
        return None
    
    try:
        # Create size-specific compiled function
        @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
        def compiled_transform(x):
            return _fast_hadamard_transform_core_optimized(x, n, normalize=True)
        
        # Warmup with dummy tensor
        dummy = torch.randn(2, n, device='cuda' if torch.cuda.is_available() else 'cpu')
        compiled_transform(dummy)
        
        _compiled_transforms[n] = compiled_transform
        return compiled_transform
    except Exception as e:
        import warnings
        warnings.warn(f"torch.compile failed for n={n}: {e}")
        return None


def clear_compile_cache():
    """Clear the torch.compile cache. Useful for memory management."""
    global _compiled_transforms
    _compiled_transforms.clear()


# Default core function selector
def _fast_hadamard_transform_core(x: Tensor, n: int, normalize: bool = True) -> Tensor:
    """Dispatch to optimal core implementation."""
    # Use optimized version by default
    return _fast_hadamard_transform_core_optimized(x, n, normalize)


def hadamard_matrix(n: int) -> Tensor:
    """
    Generate normalized Hadamard matrix of size n x n.
    n must be a power of 2.
    
    H_1 = [1]
    H_2 = 1/sqrt(2) * [[1,  1],
                       [1, -1]]
    H_2k = 1/sqrt(2) * [[H_k,  H_k],
                        [H_k, -H_k]]
    """
    if not is_power_of_two(n):
        raise ValueError(f"n must be a power of 2, got {n}")

    if n == 1:
        return torch.ones((1, 1))

    h_prev = hadamard_matrix(n // 2)
    h_n = torch.cat([
        torch.cat([h_prev, h_prev], dim=1),
        torch.cat([h_prev, -h_prev], dim=1)
    ], dim=0)
    
    return h_n / math.sqrt(2)


# =============================================================================
# Main Hadamard Transform Functions
# =============================================================================

def fast_hadamard_transform(
    x: Tensor,
    normalize: bool = True,
    inplace: bool = False,
    use_mixed_precision: Optional[bool] = None
) -> Tensor:
    """
    Apply Fast Hadamard Transform (FHT) in O(n log n) time.
    
    OPTIMIZED: Uses Triton when available, falls back to optimized PyTorch.
    Supports mixed-precision (BF16/FP16) for ~50% memory bandwidth reduction on Ampere+ GPUs.
    
    Args:
        x: Input tensor, last dimension must be power of 2
        normalize: Whether to divide by sqrt(n)
        inplace: Whether to modify input tensor (saves memory)
        use_mixed_precision: Override mixed precision mode (None = auto-detect)
    
    Returns:
        Transformed tensor
    """
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Determine mixed precision mode
    if use_mixed_precision is None:
        # Check global BF16 configuration first
        try:
            from ..constants import should_use_bf16_for_op
            use_mixed_precision = should_use_bf16_for_op(x.numel(), "hadamard")
        except ImportError:
            use_mixed_precision = _should_use_mixed_precision()

    # Use mixed precision core if enabled and on CUDA (check before Triton)
    # This ensures we actually use mixed precision when requested
    if use_mixed_precision and x.is_cuda:
        if not inplace:
            x = x.clone()
        return _fast_hadamard_transform_core_mixed(
            x, n, normalize, _get_mixed_precision_dtype()
        )

    # Try Triton acceleration if on CUDA (only for FP32 path)
    if x.is_cuda:
        try:
            from ..comfy.hadamard_kernels import (
                HAS_TRITON_HADAMARD,
                triton_hadamard_transform,
            )
            if HAS_TRITON_HADAMARD:
                # Triton kernel handles both normalize and inplace
                if inplace:
                    return triton_hadamard_transform(x, normalize=normalize, output=x)
                else:
                    return triton_hadamard_transform(x, normalize=normalize)
        except (ImportError, AttributeError):
            pass
    
    # Clone if not inplace
    if not inplace:
        x = x.clone()
    
    # Try torch.compile for repeated shapes (CUDA only)
    if x.is_cuda and not inplace and normalize:
        compiled = _get_compiled_transform(n)
        if compiled is not None:
            return compiled(x)
    
    return _fast_hadamard_transform_core(x, n, normalize)


def fast_hadamard_transform_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """
    Apply Fast Hadamard Transform with optional sign multiplications.
    
    This fused operation applies signs before the transform:
        result = FHT(x * sign_row[:, None] * sign_col[None, :])
    
    Args:
        x: Input tensor, last dimension must be power of 2
        sign_row: Optional row signs (applied to dim -2)
        sign_col: Optional column signs (applied to dim -1)
        normalize: Whether to normalize by 1/sqrt(n)
    
    Returns:
        Transformed tensor with same shape as input
    """
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be a power of 2, got {n}")
    
    # Apply signs before transform (fused into one operation)
    if sign_row is not None or sign_col is not None:
        # Clone once for signs + hadamard
        x = x.clone()
        
        # Ensure x is on the same device as the signs to avoid device mismatch
        if sign_row is not None and x.device != sign_row.device:
            x = x.to(sign_row.device)
        elif sign_col is not None and x.device != sign_col.device:
            x = x.to(sign_col.device)
        
        if sign_row is not None and sign_col is not None:
            # Both signs: x[i,j] *= row_signs[i] * col_signs[j]
            x.mul_(sign_row.unsqueeze(-1))
            x.mul_(sign_col.unsqueeze(-2))
        elif sign_row is not None:
            # Row signs only
            x.mul_(sign_row.unsqueeze(-1))
        elif sign_col is not None:
            # Column signs only
            x.mul_(sign_col.unsqueeze(-2))
        
        # Apply Hadamard transform in-place on the clone
        return fast_hadamard_transform(x, normalize=normalize, inplace=True)
    
    # No signs: Apply Hadamard transform (not in-place to respect API contract)
    # We must clone here because fast_hadamard_transform might still modify in-place
    # if it's not careful, and we want to be absolutely sure.
    return fast_hadamard_transform(x.clone(), normalize=normalize, inplace=True)


def fast_hadamard_transform_2d(
    x: Tensor,
    normalize: bool = True,
    inplace: bool = False
) -> Tensor:
    """
    Apply Fast Hadamard Transform to both dimensions of a 2D matrix.
    
    For matrix X: result = H @ X @ H.T
    where H is the Hadamard matrix.
    
    Args:
        x: Input tensor, last two dimensions must be powers of 2
        normalize: Whether to normalize by 1/sqrt(n*m)
        inplace: Whether to modify input tensor
    
    Returns:
        Transformed tensor
    """
    n_rows = x.shape[-2]
    n_cols = x.shape[-1]
    
    if not is_power_of_two(n_rows):
        raise ValueError(f"Second-to-last dimension must be power of 2, got {n_rows}")
    if not is_power_of_two(n_cols):
        raise ValueError(f"Last dimension must be power of 2, got {n_cols}")
    
    if not inplace:
        x = x.clone()
    
    # Apply Hadamard to last dimension (columns)
    # Note: When we apply to last dim first, we get: X @ H.T (not normalized yet)
    x = fast_hadamard_transform(x, normalize=False, inplace=True)
    
    # Apply Hadamard to second-to-last dimension (rows)
    # This gives us: H @ (X @ H.T) = H @ X @ H.T
    x = x.transpose(-2, -1)
    x = fast_hadamard_transform(x, normalize=False, inplace=True)
    x = x.transpose(-2, -1)
    
    if normalize:
        # Each 1D transform normalizes by 1/sqrt(n), so 2D normalizes by 1/sqrt(n*m)
        # The two transforms above didn't normalize, so we need to normalize by 1/sqrt(n*m)
        x = x / math.sqrt(n_rows * n_cols)
    
    return x


# =============================================================================
# Chunked Hadamard Transform (Memory-Efficient)
# =============================================================================

def fast_hadamard_transform_chunked(
    x: Tensor,
    normalize: bool = True,
    chunk_rows: int = 2048,
    use_mixed_precision: Optional[bool] = None
) -> Tensor:
    """
    Apply Fast Hadamard Transform in row chunks for memory efficiency.
    
    For large matrices, processes rows in chunks to keep peak memory bounded
    at O(chunk_rows * n) instead of O(m * n).
    
    Args:
        x: Input tensor (..., m, n) where n must be power of 2
        normalize: Whether to normalize by 1/sqrt(n)
        chunk_rows: Number of rows to process at once
        use_mixed_precision: Whether to use mixed precision (BF16/FP16)
        
    Returns:
        Transformed tensor with same shape as input
        
    Example:
        >>> x = torch.randn(10000, 4096)  # Large matrix
        >>> result = fast_hadamard_transform_chunked(x, chunk_rows=2048)
    """
    n = x.shape[-1]
    m = x.shape[-2] if x.ndim >= 2 else 1
    
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be power of 2, got {n}")
    
    # For small tensors, use standard implementation
    if m <= chunk_rows:
        return fast_hadamard_transform(x, normalize, use_mixed_precision=use_mixed_precision)
    
    # Determine mixed precision mode
    if use_mixed_precision is None:
        use_mixed_precision = _should_use_mixed_precision()
    
    # Pre-allocate output
    result = torch.empty_like(x)
    
    # Process in chunks
    for row_start in range(0, m, chunk_rows):
        row_end = min(row_start + chunk_rows, m)
        
        # Extract chunk
        if x.ndim == 2:
            chunk = x[row_start:row_end, :]
        else:
            # Handle higher dimensional tensors
            chunk = x[..., row_start:row_end, :]
        
        # Transform chunk
        transformed = fast_hadamard_transform(
            chunk, 
            normalize=False,  # We'll normalize at the end
            use_mixed_precision=use_mixed_precision
        )
        
        # Write back
        if x.ndim == 2:
            result[row_start:row_end, :] = transformed
        else:
            result[..., row_start:row_end, :] = transformed
        
        # Clean up intermediate
        del transformed
        if x.is_cuda and row_start % (chunk_rows * 4) == 0:
            torch.cuda.empty_cache()
    
    if normalize:
        result.mul_(1.0 / math.sqrt(n))
    
    return result


def fast_hadamard_transform_2d_chunked(
    x: Tensor,
    normalize: bool = True,
    chunk_rows: int = 2048,
    use_mixed_precision: Optional[bool] = None
) -> Tensor:
    """
    Apply 2D Fast Hadamard Transform with chunked processing.
    
    For matrix X: result = H @ X @ H.T
    
    .. warning::
        This function is experimental and currently falls back to the standard
        2D transform for correctness. The 1D chunked transform 
        (fast_hadamard_transform_chunked) is fully functional and recommended
        for memory-efficient processing.
    
    Args:
        x: Input tensor (..., m, n) where both m and n are powers of 2
        normalize: Whether to normalize by 1/sqrt(n*m)
        chunk_rows: Number of rows to process at once (currently unused)
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Transformed tensor with same shape as input
    """
    import warnings
    warnings.warn(
        "fast_hadamard_transform_2d_chunked is experimental and falls back to "
        "standard 2D transform. Use fast_hadamard_transform_chunked for 1D "
        "memory-efficient processing.",
        UserWarning,
        stacklevel=2
    )
    
    # Fall back to standard implementation for correctness
    return fast_hadamard_transform_2d(x, normalize)


def fast_hadamard_transform_2d_chunked_with_signs(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True,
    chunk_rows: int = 2048
) -> Tensor:
    """
    Apply 2D chunked Hadamard Transform with optional signs.
    
    This applies: H @ (x * sign_row[:, None] * sign_col[None, :]) @ H.T
    
    Args:
        x: Input tensor (..., m, n)
        sign_row: Row signs (length m)
        sign_col: Column signs (length n)
        normalize: Whether to normalize
        chunk_rows: Number of rows per chunk
        
    Returns:
        Transformed tensor
    """
    n_rows = x.shape[-2]
    n_cols = x.shape[-1]
    
    if not is_power_of_two(n_rows) or not is_power_of_two(n_cols):
        raise ValueError(f"Dimensions must be powers of 2, got {n_rows} x {n_cols}")
    
    # FIXED: The previous chunked implementation was mathematically incorrect
    # because it attempted to perform a row-wise Hadamard transform by chunking
    # the row dimension. A Hadamard transform of size N requires all N elements
    # to be present. For now, we fall back to the standard 2D transform for correctness.
    import warnings
    warnings.warn(
        "fast_hadamard_transform_2d_chunked_with_signs is experimental and falls back to "
        "standard 2D transform for correctness.",
        UserWarning,
        stacklevel=2
    )
    
    return fast_hadamard_transform_2d_chunked_with_signs_fallback(
        x, sign_row, sign_col, normalize
    )

def fast_hadamard_transform_2d_chunked_with_signs_fallback(
    x: Tensor,
    sign_row: Optional[Tensor] = None,
    sign_col: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """Fallback implementation for 2D Hadamard with signs."""
    if sign_row is not None or sign_col is not None:
        x = x.clone()
        if sign_row is not None:
            x.mul_(sign_row.view(*([1] * (x.ndim - 2)), -1, 1))
        if sign_col is not None:
            x.mul_(sign_col.view(*([1] * (x.ndim - 2)), 1, -1))
            
    return fast_hadamard_transform_2d(x, normalize=normalize, inplace=True)

def random_orthogonal_matrix(n: int, seed: Optional[int] = None, device: str = "cpu") -> Tensor:
    """
    Generate random orthogonal matrix via QR decomposition of random Gaussian.
    
    Algorithm:
    1. G = randn(n, n)
    2. Q, R = QR(G)
    3. Q = Q @ diag(sign(diag(R)))  # Ensure uniform distribution on O(n)
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    g = torch.randn(n, n, device=device)
    q, r = torch.linalg.qr(g)
    
    # Ensure uniform distribution
    d = torch.diag(r).sign()
    # Handle zeros in sign (unlikely but possible)
    d[d == 0] = 1
    
    q = q * d.unsqueeze(0)
    return q
