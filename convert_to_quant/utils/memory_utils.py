"""Memory utilities for GPU memory management and OOM prevention.

This module provides tools for:
- Pre-emptive OOM detection
- Memory availability checking
- Tensor size estimation
- Memory pressure monitoring
"""
import torch
from typing import Optional, Tuple, Union, Dict
import warnings

# Global flag to disable all memory limits and OOM checks
_NO_MEMORY_LIMITS = False


def set_no_memory_limits(enabled: bool = True):
    """Enable or disable all memory limits and OOM checks globally.
    
    When enabled, all OOMGuard checks will return True (allowing GPU allocation)
    and should_use_cpu_* functions will return False (forcing GPU usage).
    
    Args:
        enabled: True to disable memory limits, False to enable them (default)
    """
    global _NO_MEMORY_LIMITS
    _NO_MEMORY_LIMITS = enabled


class OOMGuard:
    """
    Pre-emptive OOM detection and prevention for GPU operations.
    
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
        >>> print(f"Free: {summary['free_gb']:.2f} GB")
    """
    
    # Safety factor - require this much extra memory beyond allocation
    DEFAULT_SAFETY_FACTOR: float = 1.25
    
    # Minimum free memory to consider (in bytes)
    DEFAULT_MIN_FREE_MB: int = 512
    
    @staticmethod
    def check_allocation(
        size_bytes: int, 
        device: Union[int, str] = 0,
        safety_factor: Optional[float] = None,
        min_free_mb: Optional[int] = None
    ) -> bool:
        """
        Check if a GPU allocation would likely succeed.
        
        Queries current GPU memory status and determines if an allocation
        of the specified size would succeed. Includes safety margins to
        account for fragmentation and allocation overhead.
        
        Args:
            size_bytes: Size of the desired allocation in bytes
            device: GPU device index or 'cuda:X' string
            safety_factor: Multiplier for required memory (default 1.25)
            min_free_mb: Minimum free memory required in MB (default 512)
            
        Returns:
            True if allocation should succeed, False if likely to OOM
            
        Note:
            Returns False if CUDA is not available.
        """
        global _NO_MEMORY_LIMITS
        if _NO_MEMORY_LIMITS:
            return True  # Always allow when no memory limits
        
        if not torch.cuda.is_available():
            return False
        
        # Parse device
        if isinstance(device, str):
            if device.startswith('cuda:'):
                device = int(device.split(':')[1])
            else:
                device = 0
        
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            
            # Available memory is total minus reserved
            free_memory = total - reserved
            
            # Calculate required memory with safety margins
            safety = safety_factor or OOMGuard.DEFAULT_SAFETY_FACTOR
            min_free = (min_free_mb or OOMGuard.DEFAULT_MIN_FREE_MB) * 1024 * 1024
            required = int(size_bytes * safety) + min_free
            
            return free_memory >= required
            
        except Exception as e:
            # If we can't determine memory, be conservative
            warnings.warn(f"Failed to query GPU memory: {e}")
            return False
    
    @staticmethod
    def get_memory_summary(device: Union[int, str] = 0) -> Dict[str, Union[float, bool]]:
        """
        Get comprehensive memory status summary.
        
        Returns detailed information about GPU memory usage including
        total, reserved, allocated, and free memory. Also includes
        fragmentation ratio which indicates memory pressure.
        
        Args:
            device: GPU device index or 'cuda:X' string
            
        Returns:
            Dictionary with memory statistics:
                - available: Whether CUDA is available
                - total_gb: Total GPU memory in GB
                - reserved_gb: Reserved memory in GB
                - allocated_gb: Allocated memory in GB  
                - free_gb: Free memory in GB
                - fragmentation_ratio: reserved/allocated ratio
        """
        if not torch.cuda.is_available():
            return {'available': False}
        
        # Parse device
        if isinstance(device, str):
            if device.startswith('cuda:'):
                device = int(device.split(':')[1])
            else:
                device = 0
        
        try:
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free = total - reserved
            
            return {
                'available': True,
                'device_name': props.name,
                'total_gb': total / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'allocated_gb': allocated / (1024**3),
                'free_gb': free / (1024**3),
                'fragmentation_ratio': reserved / max(1, allocated),
                'utilization_percent': (reserved / total) * 100,
            }
            
        except Exception as e:
            warnings.warn(f"Failed to get memory summary: {e}")
            return {'available': False, 'error': str(e)}
    
    @staticmethod
    def estimate_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """
        Estimate the size of a tensor in bytes.
        
        Args:
            shape: Tensor shape tuple
            dtype: PyTorch data type
            
        Returns:
            Estimated size in bytes
        """
        # Get element size in bytes
        if dtype == torch.bool:
            element_size = 1
        elif dtype.is_floating_point:
            if dtype == torch.float16 or dtype == torch.bfloat16:
                element_size = 2
            elif dtype == torch.float32:
                element_size = 4
            elif dtype == torch.float64:
                element_size = 8
            else:
                element_size = 4  # default
        else:
            # Integer types
            if dtype == torch.int8 or dtype == torch.uint8:
                element_size = 1
            elif dtype == torch.int16 or dtype == torch.uint16:
                element_size = 2
            elif dtype == torch.int32 or dtype == torch.uint32:
                element_size = 4
            elif dtype == torch.int64 or dtype == torch.uint64:
                element_size = 8
            else:
                element_size = 4  # default
        
        # Calculate total elements
        numel = 1
        for dim in shape:
            numel *= dim
        
        return numel * element_size
    
    @staticmethod
    def get_memory_pressure(device: Union[int, str] = 0) -> float:
        """
        Get current memory pressure as a ratio (0.0 - 1.0).
        
        Memory pressure is the ratio of reserved memory to total memory.
        Higher values indicate more memory pressure.
        
        Args:
            device: GPU device index or 'cuda:X' string
            
        Returns:
            Memory pressure ratio (0.0 = empty, 1.0 = full)
        """
        summary = OOMGuard.get_memory_summary(device)
        if not summary.get('available', False):
            return 0.0
        
        return summary.get('utilization_percent', 0) / 100.0


def should_use_cpu_for_hadamard(
    tensor_shape: Tuple[int, ...], 
    dtype: torch.dtype = torch.float32,
    force_cpu_threshold_elements: Optional[int] = None,
    device: Union[int, str] = 0
) -> bool:
    """
    Determine if Hadamard transform should use CPU.
    
    Combines size-based threshold with memory availability check to
    decide whether to process on CPU or GPU.
    
    Args:
        tensor_shape: Shape of the tensor to transform
        dtype: Data type of the tensor
        force_cpu_threshold_elements: Size threshold to force CPU usage
        device: GPU device to check
        
    Returns:
        True if CPU should be used, False for GPU
    """
    global _NO_MEMORY_LIMITS
    if _NO_MEMORY_LIMITS:
        return False  # Always use GPU when no memory limits
    
    # Calculate total elements
    num_elements = 1
    for dim in tensor_shape:
        num_elements *= dim
    
    # Size-based check
    threshold = force_cpu_threshold_elements or 50_000_000
    if num_elements > threshold:
        return True
    
    # Memory-based check
    # Hadamard needs ~2x the tensor size for intermediate results
    size_bytes = OOMGuard.estimate_tensor_size(tensor_shape, dtype)
    if not OOMGuard.check_allocation(size_bytes * 2, device):
        return True
    
    return False


def should_use_cpu_for_ldlq(
    M: int, 
    N: int, 
    block_size: int,
    dtype: torch.dtype = torch.float32,
    device: Union[int, str] = 0
) -> bool:
    """
    Determine if LDLQ quantization should use CPU.
    
    Estimates memory requirements for LDLQ and checks availability.
    
    Args:
        M: Output dimension (rows)
        N: Input dimension (cols)
        block_size: Quantization block size
        dtype: Data type for computations
        device: GPU device to check
        
    Returns:
        True if CPU should be used, False for GPU
    """
    global _NO_MEMORY_LIMITS
    if _NO_MEMORY_LIMITS:
        return False  # Always use GPU when no memory limits
    
    # Estimate memory needed:
    # - W_buffer: M * N elements (FP32)
    # - H_inv: N * N elements (FP32)
    # - q_buffer: M * block_size elements (INT8)
    # - err_buffer: M * block_size elements (FP32)
    element_size = 4 if dtype == torch.float32 else 2
    
    w_size = M * N * element_size
    h_size = N * N * element_size
    q_size = M * block_size * 1  # int8
    err_size = M * block_size * element_size
    
    # Add overhead for intermediate computations
    total_bytes = w_size + h_size + q_size + err_size
    total_bytes = int(total_bytes * 1.5)  # 1.5x safety factor
    
    return not OOMGuard.check_allocation(total_bytes, device)


def should_use_cpu_for_hessian(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: Union[int, str] = 0
) -> bool:
    """
    Determine if Hessian operations should use CPU.
    
    Args:
        N: Hessian matrix dimension (N x N)
        dtype: Data type for computations
        device: GPU device to check
        
    Returns:
        True if CPU should be used, False for GPU
    """
    global _NO_MEMORY_LIMITS
    if _NO_MEMORY_LIMITS:
        return False  # Always use GPU when no memory limits
    
    # Hessian is N x N
    element_size = 4 if dtype == torch.float32 else 2
    size_bytes = N * N * element_size
    
    # Cholesky needs additional workspace
    return not OOMGuard.check_allocation(size_bytes * 2, device)


def format_bytes(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
