"""Pinned memory pool for efficient CPU→GPU transfers.

This module provides a pool of reusable pinned (page-locked) memory buffers
that enable faster DMA transfers to GPU. Buffers are organized by size
buckets to maximize reuse and minimize allocation overhead.

Example:
    >>> from convert_to_quant.utils.pinned_pool import get_pinned_pool
    >>> pool = get_pinned_pool()
    >>> 
    >>> # Get a pinned buffer
    >>> buffer = pool.acquire(1000, torch.float32)
    >>> buffer[:] = my_data
    >>> 
    >>> # Transfer to GPU
    >>> gpu_tensor = buffer.cuda(non_blocking=True)
    >>> 
    >>> # Return buffer to pool for reuse
    >>> pool.release(buffer)
"""
from collections import deque
from typing import Dict, Optional, List
import torch
import threading


class PinnedMemoryPool:
    """
    Thread-safe pool of reusable pinned memory buffers.
    
    Buffers are organized into power-of-2 size buckets for efficient reuse.
    The pool has a maximum total size to prevent unbounded memory growth.
    Each bucket maintains a limited number of buffers (LRU eviction).
    
    Args:
        max_size_mb: Maximum total pool size in megabytes
        min_bucket_size: Minimum bucket size in bytes (default 1KB)
        max_bucket_size: Maximum bucket size in bytes (default 128MB)
        buffers_per_bucket: Maximum buffers to keep per bucket (default 4)
        
    Example:
        >>> pool = PinnedMemoryPool(max_size_mb=512)
        >>> buffer = pool.acquire(10000, torch.float32)
        >>> # Use buffer...
        >>> pool.release(buffer)
    """
    
    def __init__(
        self, 
        max_size_mb: int = 512,
        min_bucket_size: int = 1024,      # 1KB minimum
        max_bucket_size: int = 134217728,  # 128MB maximum
        buffers_per_bucket: int = 4
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.min_bucket_size = min_bucket_size
        self.max_bucket_size = max_bucket_size
        self.buffers_per_bucket = buffers_per_bucket
        
        # size -> deque of pinned buffers
        self._buckets: Dict[int, deque] = {}
        self._current_size = 0
        self._lock = threading.Lock()
        
        # Statistics
        self._allocations = 0
        self._reuses = 0
        self._evictions = 0
        self._total_allocated_bytes = 0
    
    def _get_bucket_size(self, size_bytes: int) -> int:
        """
        Round up to next power of 2 within bounds.
        
        Args:
            size_bytes: Required size in bytes
            
        Returns:
            Bucket size (power of 2)
        """
        if size_bytes <= self.min_bucket_size:
            return self.min_bucket_size
        if size_bytes >= self.max_bucket_size:
            return self.max_bucket_size
        return 1 << (size_bytes - 1).bit_length()
    
    def acquire(
        self, 
        numel: int, 
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Get a pinned buffer from the pool or allocate new.
        
        The returned buffer may be larger than requested but will have
        at least numel elements. The buffer is pinned (page-locked) for
        fast GPU transfers.
        
        Args:
            numel: Minimum number of elements needed
            dtype: Data type for the buffer
            
        Returns:
            Pinned memory tensor (may be larger than requested)
            
        Example:
            >>> buffer = pool.acquire(1000, torch.float32)
            >>> buffer[:1000] = my_data  # Use only what you need
            >>> gpu_tensor = buffer[:1000].cuda(non_blocking=True)
        """
        # Calculate required size
        element_size = self._get_element_size(dtype)
        size_bytes = numel * element_size
        bucket_size = self._get_bucket_size(size_bytes)
        
        with self._lock:
            # Try to get from pool
            if bucket_size in self._buckets and self._buckets[bucket_size]:
                self._reuses += 1
                buffer = self._buckets[bucket_size].popleft()
                self._current_size -= bucket_size
                return buffer
        
        # Allocate new buffer
        self._allocations += 1
        numel_padded = bucket_size // element_size
        
        try:
            buffer = torch.empty(numel_padded, dtype=dtype).pin_memory()
            self._total_allocated_bytes += bucket_size
            return buffer
        except RuntimeError as e:
            # If allocation fails, try with smaller bucket
            if bucket_size > self.min_bucket_size:
                smaller_bucket = max(self.min_bucket_size, bucket_size // 2)
                numel_padded = smaller_bucket // element_size
                if numel_padded >= numel:
                    return torch.empty(numel_padded, dtype=dtype).pin_memory()
            raise e
    
    def release(self, buffer: torch.Tensor) -> bool:
        """
        Return a buffer to the pool for reuse.
        
        If the pool is full or the buffer is not pinned, it will be
        discarded (garbage collected).
        
        Args:
            buffer: Pinned memory tensor to return to pool
            
        Returns:
            True if buffer was added to pool, False if discarded
        """
        if not buffer.is_pinned():
            return False  # Don't cache non-pinned buffers
        
        size_bytes = buffer.numel() * buffer.element_size()
        bucket_size = self._get_bucket_size(size_bytes)
        
        with self._lock:
            # Check if adding this would exceed max size
            if self._current_size + bucket_size > self.max_size_bytes:
                # Pool is full
                return False
            
            if bucket_size not in self._buckets:
                self._buckets[bucket_size] = deque(maxlen=self.buffers_per_bucket)
            
            # Check if bucket is full (deque will auto-evict, but track it)
            if len(self._buckets[bucket_size]) >= self.buffers_per_bucket:
                self._evictions += 1
            
            self._buckets[bucket_size].append(buffer)
            self._current_size += bucket_size
            return True
    
    def _get_element_size(self, dtype: torch.dtype) -> int:
        """Get element size in bytes for a dtype."""
        if dtype == torch.float32:
            return 4
        elif dtype == torch.float64:
            return 8
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            return 2
        elif dtype == torch.int32 or dtype == torch.uint32:
            return 4
        elif dtype == torch.int64 or dtype == torch.uint64:
            return 8
        elif dtype == torch.int16 or dtype == torch.uint16:
            return 2
        elif dtype == torch.int8 or dtype == torch.uint8:
            return 1
        else:
            return 4  # default
    
    def get_stats(self) -> dict:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool statistics:
                - allocations: Number of new buffer allocations
                - reuses: Number of buffer reuses from pool
                - evictions: Number of buffers evicted from pool
                - reuse_rate: Ratio of reuses to total requests
                - current_size_mb: Current pool size in MB
                - total_allocated_mb: Total bytes ever allocated
                - bucket_count: Number of active buckets
        """
        total_requests = self._allocations + self._reuses
        return {
            'allocations': self._allocations,
            'reuses': self._reuses,
            'evictions': self._evictions,
            'reuse_rate': self._reuses / max(1, total_requests),
            'current_size_mb': self._current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'total_allocated_mb': self._total_allocated_bytes / (1024 * 1024),
            'bucket_count': len(self._buckets),
            'total_requests': total_requests,
        }
    
    def clear(self) -> int:
        """
        Clear all buffers from the pool.
        
        Returns:
            Number of buffers cleared
        """
        with self._lock:
            count = sum(len(bucket) for bucket in self._buckets.values())
            self._buckets.clear()
            self._current_size = 0
            return count
    
    def resize(self, max_size_mb: int) -> None:
        """
        Resize the pool maximum size.
        
        If the new size is smaller than current usage, buffers will be
        evicted until the pool fits within the new limit.
        
        Args:
            max_size_mb: New maximum size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        with self._lock:
            # Evict buffers if we're over the new limit
            while self._current_size > self.max_size_bytes:
                # Find the largest bucket and evict from it
                if not self._buckets:
                    break
                
                largest_bucket = max(self._buckets.keys())
                if self._buckets[largest_bucket]:
                    self._buckets[largest_bucket].pop()
                    self._current_size -= largest_bucket
                    self._evictions += 1
                else:
                    del self._buckets[largest_bucket]


# Global pool instance
_global_pinned_pool: Optional[PinnedMemoryPool] = None
_pool_lock = threading.Lock()


def get_pinned_pool(
    max_size_mb: Optional[int] = None,
    force_new: bool = False
) -> PinnedMemoryPool:
    """
    Get the global pinned memory pool.
    
    Lazily initializes on first call. The global pool is shared across
    the application for maximum buffer reuse.
    
    Args:
        max_size_mb: Maximum pool size (only used on first call)
        force_new: If True, create a new pool even if one exists
        
    Returns:
        The global PinnedMemoryPool instance
        
    Example:
        >>> pool = get_pinned_pool()
        >>> buffer = pool.acquire(10000)
        >>> # Use buffer...
        >>> pool.release(buffer)
    """
    global _global_pinned_pool
    
    with _pool_lock:
        if _global_pinned_pool is None or force_new:
            size = max_size_mb or 512
            _global_pinned_pool = PinnedMemoryPool(max_size_mb=size)
        return _global_pinned_pool


def set_pinned_pool(pool: PinnedMemoryPool) -> None:
    """
    Set a custom global pinned memory pool.
    
    Args:
        pool: PinnedMemoryPool instance to use globally
    """
    global _global_pinned_pool
    with _pool_lock:
        _global_pinned_pool = pool


def transfer_with_pool(
    tensor: torch.Tensor,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True
) -> torch.Tensor:
    """
    Transfer tensor to GPU using pooled pinned memory.
    
    Uses the global PinnedMemoryPool for efficient buffer reuse.
    This is faster than standard transfer for repeated operations
    because it reuses pinned memory buffers.
    
    Args:
        tensor: CPU tensor to transfer
        device: Target device (default 'cuda')
        dtype: Optional dtype conversion
        non_blocking: Use non-blocking transfer (default True)
        
    Returns:
        Tensor on target device
        
    Example:
        >>> cpu_tensor = torch.randn(1000, 1000)
        >>> gpu_tensor = transfer_with_pool(cpu_tensor)
    """
    # Skip if not a CPU tensor or CUDA unavailable
    if tensor.device.type != 'cpu' or not torch.cuda.is_available():
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    
    # Skip if target is not CUDA
    if not str(device).startswith('cuda'):
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    
    # Get pool and acquire buffer
    pool = get_pinned_pool()
    pinned = pool.acquire(tensor.numel(), tensor.dtype)
    
    try:
        # Copy data to pinned buffer
        view = pinned[:tensor.numel()]
        view.copy_(tensor.view(-1))
        
        # Reshape to original shape
        pinned_reshaped = view.view(tensor.shape)
        
        # Transfer to GPU
        if dtype is not None:
            result = pinned_reshaped.to(device=device, dtype=dtype, non_blocking=non_blocking)
        else:
            result = pinned_reshaped.to(device=device, non_blocking=non_blocking)
        
        return result
        
    finally:
        # Always return buffer to pool
        pool.release(pinned)


def clear_pinned_pool() -> int:
    """
    Clear all buffers from the global pinned pool.
    
    Returns:
        Number of buffers cleared
    """
    global _global_pinned_pool
    if _global_pinned_pool is not None:
        return _global_pinned_pool.clear()
    return 0


def get_pinned_pool_stats() -> Optional[dict]:
    """
    Get statistics for the global pinned pool.
    
    Returns:
        Dictionary of statistics or None if pool not initialized
    """
    global _global_pinned_pool
    if _global_pinned_pool is not None:
        return _global_pinned_pool.get_stats()
    return None
