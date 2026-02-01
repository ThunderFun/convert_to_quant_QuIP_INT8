"""Memory utilities for GPU memory management and OOM prevention.

This module provides tools for:
- Pre-emptive OOM detection
- Memory availability checking
- Tensor size estimation
- Memory pressure monitoring
"""

import threading
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Dict, Iterator, Optional, Tuple, Union

import torch

# =============================================================================
# Constants
# =============================================================================

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

# Default thresholds
DEFAULT_SAFETY_FACTOR = 1.25
DEFAULT_MIN_FREE_MB = 512
DEFAULT_PRESSURE_THRESHOLD = 0.85
DEFAULT_HADAMARD_THRESHOLD_ELEMENTS = 50_000_000
DEFAULT_CACHE_CLEANUP_INTERVAL = 8


# =============================================================================
# Global State (Thread-Safe)
# =============================================================================

_state_lock = threading.Lock()
_no_memory_limits = False
_cache_cleanup_counter = 0
_cache_cleanup_interval = DEFAULT_CACHE_CLEANUP_INTERVAL
_last_memory_pressure = 0.0


def set_no_memory_limits(enabled: bool = True) -> None:
    """Enable or disable all memory limits and OOM checks globally.
    
    When enabled, all OOMGuard checks will return True (allowing GPU allocation)
    and `should_use_cpu_*` functions will return False (forcing GPU usage).
    
    Args:
        enabled: True to disable memory limits, False to enable them.
    
    Note:
        This function is thread-safe.
    """
    global _no_memory_limits
    with _state_lock:
        _no_memory_limits = enabled


def get_no_memory_limits() -> bool:
    """Check if memory limits are disabled.
    
    Returns:
        True if memory limits are disabled, False otherwise.
    """
    with _state_lock:
        return _no_memory_limits


@contextmanager
def memory_limits_disabled():
    """Context manager to temporarily disable memory limits.
    
    Example:
        >>> with memory_limits_disabled():
        ...     # All OOM checks bypassed here
        ...     large_tensor = torch.empty(1_000_000_000, device='cuda')
    """
    previous = get_no_memory_limits()
    set_no_memory_limits(True)
    try:
        yield
    finally:
        set_no_memory_limits(previous)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class MemorySummary(Mapping):
    """Immutable summary of GPU memory status.
    
    Behaves like both a dataclass (attribute access) and a dict (for backward compatibility).
    """
    available: bool
    device_name: str = ""
    total_gb: float = 0.0
    reserved_gb: float = 0.0
    allocated_gb: float = 0.0
    free_gb: float = 0.0
    fragmentation_ratio: float = 0.0
    utilization_percent: float = 0.0
    error: Optional[str] = None
    
    def __getitem__(self, key: str) -> Union[float, bool, str, None]:
        """Allow dict-style access: summary['available']."""
        return getattr(self, key)
    
    def __iter__(self) -> Iterator[str]:
        """Allow iteration over keys."""
        return (f.name for f in fields(self))
    
    def __len__(self) -> int:
        """Return number of fields."""
        return len(fields(self))
    
    def get(self, key: str, default=None) -> Union[float, bool, str, None]:
        """Dict-style .get() method with optional default."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Union[float, bool, str, None]]:
        """Convert to dictionary for backward compatibility."""
        return {
            'available': self.available,
            'device_name': self.device_name,
            'total_gb': self.total_gb,
            'reserved_gb': self.reserved_gb,
            'allocated_gb': self.allocated_gb,
            'free_gb': self.free_gb,
            'fragmentation_ratio': self.fragmentation_ratio,
            'utilization_percent': self.utilization_percent,
            'error': self.error,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_device(device: Union[int, str, torch.device]) -> int:
    """Parse device specification to integer index.
    
    Args:
        device: GPU device as int, 'cuda:X' string, or torch.device.
        
    Returns:
        Integer device index.
        
    Raises:
        ValueError: If device specification is invalid or unsupported.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        if device.startswith('cuda:'):
            try:
                return int(device.split(':')[1])
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid CUDA device string: '{device}'") from e
        if device == 'cuda':
            return 0
    raise ValueError(f"Invalid device specification: {device!r} (expected int, 'cuda:X', or torch.device)")


@lru_cache(maxsize=32)
def _get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes of a PyTorch dtype.
    
    Args:
        dtype: PyTorch data type.
        
    Returns:
        Size in bytes per element.
    """
    # Create a dummy tensor to get element size - most reliable method
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        # Fallback for edge cases
        dtype_sizes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.int8: 1,
            torch.uint8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
            torch.complex64: 8,
            torch.complex128: 16,
        }
        return dtype_sizes.get(dtype, 4)


def estimate_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    """Estimate the size of a tensor in bytes.
    
    Args:
        shape: Tensor shape tuple.
        dtype: PyTorch data type.
        
    Returns:
        Estimated size in bytes.
        
    Example:
        >>> estimate_tensor_size((1000, 1000), torch.float32)
        4000000
    """
    if not shape:
        return 0
    
    numel = 1
    for dim in shape:
        numel *= dim
    
    return numel * _get_dtype_size(dtype)


def format_bytes(size_bytes: Union[int, float]) -> str:
    """Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted string (e.g., "1.50 GB", "256.00 MB").
    """
    if size_bytes < 0:
        return f"-{format_bytes(-size_bytes)}"
    
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


# =============================================================================
# OOMGuard Class
# =============================================================================

class OOMGuard:
    """Pre-emptive OOM detection and prevention for GPU operations.
    
    Queries GPU memory availability before large allocations to predict
    and prevent OOM errors. Provides utilities for making CPU/GPU
    processing decisions based on memory pressure.
    
    Example:
        >>> from convert_to_quant.utils.memory_utils import OOMGuard
        >>> 
        >>> # Check before allocating a large tensor
        >>> if OOMGuard.check_allocation(500 * 1024 * 1024):  # 500MB
        ...     tensor = torch.empty(500_000_000, device='cuda')
        ... else:
        ...     tensor = torch.empty(500_000_000, device='cpu')
        >>> 
        >>> # Get memory summary
        >>> summary = OOMGuard.get_memory_summary()
        >>> print(f"Free: {summary.free_gb:.2f} GB")
    
    Attributes:
        DEFAULT_SAFETY_FACTOR: Multiplier for required memory (1.25).
        DEFAULT_MIN_FREE_MB: Minimum free memory in MB (512).
    """
    
    DEFAULT_SAFETY_FACTOR: float = DEFAULT_SAFETY_FACTOR
    DEFAULT_MIN_FREE_MB: int = DEFAULT_MIN_FREE_MB
    
    @staticmethod
    def check_allocation(
        size_bytes: int, 
        device: Union[int, str, torch.device] = 0,
        safety_factor: Optional[float] = None,
        min_free_mb: Optional[int] = None
    ) -> bool:
        """Check if a GPU allocation would likely succeed.
        
        Queries current GPU memory status and determines if an allocation
        of the specified size would succeed. Includes safety margins to
        account for fragmentation and allocation overhead.
        
        Args:
            size_bytes: Size of the desired allocation in bytes.
            device: GPU device index, 'cuda:X' string, or torch.device.
            safety_factor: Multiplier for required memory (default 1.25).
            min_free_mb: Minimum free memory required in MB (default 512).
            
        Returns:
            True if allocation should succeed, False if likely to OOM.
            
        Note:
            Returns True if memory limits are disabled globally.
            Returns False if CUDA is not available.
        """
        if get_no_memory_limits():
            return True
        
        if not torch.cuda.is_available():
            return False
        
        device_idx = _parse_device(device)
        
        try:
            total = torch.cuda.get_device_properties(device_idx).total_memory
            reserved = torch.cuda.memory_reserved(device_idx)
            
            # Available memory is reserved minus allocated (accounts for PyTorch's memory caching)
            # This gives the actual free memory within PyTorch's reserved pool
            free_memory = reserved - torch.cuda.memory_allocated(device_idx)
            
            # Calculate required memory with safety margins
            safety = safety_factor if safety_factor is not None else OOMGuard.DEFAULT_SAFETY_FACTOR
            min_free = (min_free_mb if min_free_mb is not None else OOMGuard.DEFAULT_MIN_FREE_MB)
            min_free_bytes = min_free * BYTES_PER_MB
            
            required = int(size_bytes * safety) + min_free_bytes
            
            return free_memory >= required
            
        except RuntimeError as e:
            warnings.warn(f"Failed to query GPU memory: {e}", stacklevel=2)
            return False
    
    @staticmethod
    def get_memory_summary(device: Union[int, str, torch.device] = 0) -> MemorySummary:
        """Get comprehensive memory status summary.
        
        Returns detailed information about GPU memory usage including
        total, reserved, allocated, and free memory. Also includes
        fragmentation ratio which indicates memory pressure.
        
        Args:
            device: GPU device index, 'cuda:X' string, or torch.device.
            
        Returns:
            MemorySummary dataclass with memory statistics.
        """
        if not torch.cuda.is_available():
            return MemorySummary(available=False)
        
        device_idx = _parse_device(device)
        
        try:
            props = torch.cuda.get_device_properties(device_idx)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            free = total - allocated
            
            return MemorySummary(
                available=True,
                device_name=props.name,
                total_gb=total / BYTES_PER_GB,
                reserved_gb=reserved / BYTES_PER_GB,
                allocated_gb=allocated / BYTES_PER_GB,
                free_gb=free / BYTES_PER_GB,
                fragmentation_ratio=reserved / max(1, allocated),
                utilization_percent=(allocated / max(1, total)) * 100,
            )
            
        except RuntimeError as e:
            warnings.warn(f"Failed to get memory summary: {e}", stacklevel=2)
            return MemorySummary(available=False, error=str(e))
    
    @staticmethod
    def estimate_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Estimate the size of a tensor in bytes.
        
        Args:
            shape: Tensor shape tuple.
            dtype: PyTorch data type.
            
        Returns:
            Estimated size in bytes.
        """
        return estimate_tensor_size(shape, dtype)
    
    @staticmethod
    def get_memory_pressure(device: Union[int, str, torch.device] = 0) -> float:
        """Get current memory pressure as a ratio (0.0 - 1.0).
        
        Memory pressure is the ratio of reserved memory to total memory.
        Higher values indicate more memory pressure.
        
        Args:
            device: GPU device index, 'cuda:X' string, or torch.device.
            
        Returns:
            Memory pressure ratio (0.0 = empty, 1.0 = full).
        """
        summary = OOMGuard.get_memory_summary(device)
        if not summary.available:
            return 0.0
        
        return summary.utilization_percent / 100.0
    
    @staticmethod
    def can_allocate_tensor(
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: Union[int, str, torch.device] = 0,
        safety_factor: Optional[float] = None,
    ) -> bool:
        """Check if a tensor with given shape and dtype can be allocated.
        
        Convenience method combining size estimation with allocation check.
        
        Args:
            shape: Tensor shape tuple.
            dtype: PyTorch data type.
            device: GPU device.
            safety_factor: Memory safety multiplier.
            
        Returns:
            True if tensor can likely be allocated.
        """
        size_bytes = estimate_tensor_size(shape, dtype)
        return OOMGuard.check_allocation(size_bytes, device, safety_factor)


# =============================================================================
# CPU/GPU Decision Functions
# =============================================================================

def should_use_cpu_for_hadamard(
    tensor_shape: Tuple[int, ...], 
    dtype: torch.dtype = torch.float32,
    force_cpu_threshold_elements: Optional[int] = None,
    device: Union[int, str, torch.device] = 0
) -> bool:
    """Determine if Hadamard transform should use CPU.
    
    Combines size-based threshold with memory availability check to
    decide whether to process on CPU or GPU.
    
    Args:
        tensor_shape: Shape of the tensor to transform.
        dtype: Data type of the tensor.
        force_cpu_threshold_elements: Element count threshold to force CPU usage.
        device: GPU device to check.
        
    Returns:
        True if CPU should be used, False for GPU.
    """
    if get_no_memory_limits():
        return False
    
    # Calculate total elements
    num_elements = 1
    for dim in tensor_shape:
        num_elements *= dim
    
    # Size-based check
    threshold = force_cpu_threshold_elements or DEFAULT_HADAMARD_THRESHOLD_ELEMENTS
    if num_elements > threshold:
        return True
    
    # Memory-based check
    # Hadamard needs ~2x the tensor size for intermediate results
    size_bytes = estimate_tensor_size(tensor_shape, dtype)
    if not OOMGuard.check_allocation(size_bytes * 2, device):
        return True
    
    return False


def should_use_cpu_for_ldlq(
    M: int, 
    N: int, 
    block_size: int,
    dtype: torch.dtype = torch.float32,
    device: Union[int, str, torch.device] = 0
) -> bool:
    """Determine if LDLQ quantization should use CPU.
    
    Estimates memory requirements for LDLQ and checks availability.
    
    Args:
        M: Output dimension (rows).
        N: Input dimension (cols).
        block_size: Quantization block size.
        dtype: Data type for computations.
        device: GPU device to check.
        
    Returns:
        True if CPU should be used, False for GPU.
    """
    if get_no_memory_limits():
        return False
    
    element_size = _get_dtype_size(dtype)
    
    # Estimate memory needed:
    # - W_buffer: M * N elements
    # - H_inv: N * N elements
    # - q_buffer: M * block_size elements (INT8)
    # - err_buffer: M * block_size elements
    w_size = M * N * element_size
    h_size = N * N * element_size
    q_size = M * block_size  # int8 = 1 byte
    err_size = M * block_size * element_size
    
    # Add overhead for intermediate computations (1.5x safety factor)
    total_bytes = int((w_size + h_size + q_size + err_size) * 1.5)
    
    return not OOMGuard.check_allocation(total_bytes, device)


def should_use_cpu_for_hessian(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: Union[int, str, torch.device] = 0
) -> bool:
    """Determine if Hessian operations should use CPU.
    
    Args:
        N: Hessian matrix dimension (N x N).
        dtype: Data type for computations.
        device: GPU device to check.
        
    Returns:
        True if CPU should be used, False for GPU.
    """
    if get_no_memory_limits():
        return False
    
    element_size = _get_dtype_size(dtype)
    size_bytes = N * N * element_size
    
    # Cholesky needs additional workspace (~2x)
    return not OOMGuard.check_allocation(size_bytes * 2, device)


# =============================================================================
# Smart Memory Cache Management
# =============================================================================

def maybe_empty_cache(
    force: bool = False,
    pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD,
    device: Union[int, str, torch.device] = 0
) -> bool:
    """Conditionally empty CUDA cache based on memory pressure.
    
    Unlike raw `torch.cuda.empty_cache()`, this function:
    
    1. Only empties cache when memory pressure exceeds threshold
    2. Rate-limits checks to avoid excessive synchronization
    3. Returns immediately if CUDA unavailable or device is CPU
    4. Can be forced to empty cache when needed
    
    This dramatically reduces GPU synchronization overhead while
    still preventing OOM errors during memory-intensive operations.
    
    Args:
        force: If True, always empty cache.
        pressure_threshold: Memory pressure threshold (0.0-1.0) to trigger cleanup.
        device: GPU device to check.
        
    Returns:
        True if cache was emptied, False otherwise.
    """
    global _cache_cleanup_counter, _last_memory_pressure
    
    if not torch.cuda.is_available():
        return False
        
    # Handle device specification - return early if CPU
    if isinstance(device, torch.device) and device.type == 'cpu':
        return False
    if isinstance(device, str) and device == 'cpu':
        return False
    
    if force:
        torch.cuda.empty_cache()
        return True
    
    # Rate-limit checks to reduce synchronization
    with _state_lock:
        _cache_cleanup_counter += 1
        counter = _cache_cleanup_counter
        interval = _cache_cleanup_interval
        cached_pressure = _last_memory_pressure
    
    # If we're not at the interval, only check if the LAST known pressure was high
    if counter % interval != 0:
        if cached_pressure < pressure_threshold:
            return False
    
    # Check current memory pressure
    try:
        device_idx = _parse_device(device)
        
        # Use memory_reserved vs total for a more stable pressure metric
        # allocated vs total can fluctuate too rapidly
        total = torch.cuda.get_device_properties(device_idx).total_memory
        reserved = torch.cuda.memory_reserved(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx)
        
        # We only want to empty cache if there's actually something to empty!
        # If reserved is close to allocated, empty_cache() does nothing.
        if reserved - allocated < 128 * 1024 * 1024: # Less than 128MB to reclaim
            return False
            
        pressure = allocated / max(1, total)
        
        with _state_lock:
            _last_memory_pressure = pressure
        
        if pressure > pressure_threshold:
            torch.cuda.empty_cache()
            return True
        
        return False
        
    except (RuntimeError, ValueError):
        return False


def set_cache_cleanup_interval(interval: int) -> None:
    """Set how often `maybe_empty_cache()` actually checks memory.
    
    Higher values = better performance but more memory usage.
    Lower values = more responsive but more synchronization overhead.
    
    Args:
        interval: Check memory every N calls (default: 8).
    """
    global _cache_cleanup_interval
    with _state_lock:
        _cache_cleanup_interval = max(1, interval)


def get_cache_cleanup_interval() -> int:
    """Get the current cache cleanup interval."""
    with _state_lock:
        return _cache_cleanup_interval


def empty_cache_if_needed(
    min_free_mb: int = 1024,
    device: Union[int, str, torch.device] = 0
) -> bool:
    """Empty cache if free memory is below threshold.
    
    More precise than pressure-based check - useful when you know
    exactly how much memory you need.
    
    Args:
        min_free_mb: Minimum free memory required in MB.
        device: GPU device to check.
        
    Returns:
        True if cache was emptied, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    
    device_idx = _parse_device(device)
    
    try:
        props = torch.cuda.get_device_properties(device_idx)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device_idx)
        free_bytes = total - allocated
        
        if free_bytes < min_free_mb * BYTES_PER_MB:
            torch.cuda.empty_cache()
            return True
        
        return False
        
    except RuntimeError:
        return False


def synchronize_and_empty_cache(device: Union[int, str, torch.device] = 0) -> None:
    """Synchronize CUDA stream and empty cache.
    
    Use this before measuring memory or when you need guaranteed cleanup.
    
    Args:
        device: GPU device to synchronize.
    """
    if not torch.cuda.is_available():
        return
    
    device_idx = _parse_device(device)
    torch.cuda.synchronize(device_idx)
    torch.cuda.empty_cache()
