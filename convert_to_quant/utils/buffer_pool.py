"""Buffer pool for LDLQ quantization operations.

This module provides a pool of reusable buffers for LDLQ quantization,
eliminating repeated allocations when processing multiple layers with
different dimensions.

Example:
    >>> from convert_to_quant.utils.buffer_pool import LDLQBufferPool
    >>> pool = LDLQBufferPool(max_entries=8)
    >>> 
    >>> # Get buffers for a specific size
    >>> q_buf, err_buf, w_buf = pool.get_buffers(4096, 4096, 128, 'cuda')
    >>> 
    >>> # Use buffers for quantization...
    >>> 
    >>> # Buffers are automatically cached for reuse
    >>> q_buf2, err_buf2, w_buf2 = pool.get_buffers(4096, 4096, 128, 'cuda')
    >>> # These may be the same buffers as before!
"""
from collections import deque, OrderedDict
from typing import Dict, Tuple, Optional, Union
import torch
import threading


class LDLQBufferPool:
    """
    Manages reusable buffers for LDLQ quantization.
    
    Buffers are keyed by (rounded_M, rounded_N, block_size, device) and reused
    across quantization calls to minimize allocations. Uses LRU eviction when
    the pool reaches its maximum size.
    
    Args:
        max_entries: Maximum number of buffer sets to cache
        round_size: Round dimensions to nearest multiple of this value
        
    Example:
        >>> pool = LDLQBufferPool(max_entries=8)
        >>> q, err, w = pool.get_buffers(4096, 4096, 128, 'cuda')
        >>> # Use buffers...
        >>> # On next call with same dimensions, cached buffers are returned
        >>> q2, err2, w2 = pool.get_buffers(4096, 4096, 128, 'cuda')
    """
    
    def __init__(self, max_entries: int = 8, round_size: int = 1024):
        self.max_entries = max_entries
        self.round_size = round_size
        
        # OrderedDict for LRU: most recent at the end
        self._buffers: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._total_allocated_mb = 0
    
    def _round_dims(self, M: int, N: int, block_size: int) -> Tuple[int, int, int]:
        """
        Round dimensions to reduce unique buffer sizes.
        
        Args:
            M: Number of rows
            N: Number of columns
            block_size: Block size for quantization
            
        Returns:
            Tuple of (rounded_M, rounded_N, rounded_block)
        """
        M_rounded = ((M + self.round_size - 1) // self.round_size) * self.round_size
        N_rounded = ((N + self.round_size - 1) // self.round_size) * self.round_size
        block_rounded = ((block_size + 127) // 128) * 128
        return (M_rounded, N_rounded, block_rounded)
    
    def _make_key(
        self, 
        M: int, 
        N: int, 
        block_size: int, 
        device: Union[str, torch.device]
    ) -> Tuple:
        """Create cache key from dimensions and device."""
        rounded = self._round_dims(M, N, block_size)
        device_str = str(device)
        return (*rounded, device_str)
    
    def get_buffers(
        self, 
        M: int, 
        N: int, 
        block_size: int, 
        device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get or create quantization buffers.
        
        Returns buffers sized for the requested dimensions. If buffers
        for these (rounded) dimensions exist in the pool, they are reused.
        Otherwise, new buffers are allocated and added to the pool.
        
        Args:
            M: Number of rows (output dimension)
            N: Number of columns (input dimension)
            block_size: Quantization block size
            device: Device for buffers ('cpu', 'cuda', etc.)
            
        Returns:
            Tuple of (q_buffer, err_buffer, W_buffer) where:
                - q_buffer: INT8 buffer of shape (M, block_size)
                - err_buffer: FP32 buffer of shape (M, block_size)
                - W_buffer: FP32 buffer of shape (M, N)
                
        Example:
            >>> q_buf, err_buf, w_buf = pool.get_buffers(4096, 4096, 128, 'cuda')
            >>> print(q_buf.shape)  # torch.Size([4096, 128])
            >>> print(w_buf.shape)  # torch.Size([4096, 4096])
        """
        # Validate dimensions - zero dimensions indicate an upstream issue
        if M <= 0:
            raise ValueError(f"Number of rows (M) must be positive, got {M}")
        if N <= 0:
            raise ValueError(f"Number of columns (N) must be positive, got {N}")
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")
        
        key = self._make_key(M, N, block_size, device)
        
        with self._lock:
            if key in self._buffers:
                # Cache hit - move to end (most recent)
                self._hit_count += 1
                entry = self._buffers.pop(key)
                self._buffers[key] = entry
                
                # Return slices of the cached buffers
                q_buf = entry['q'][:M, :block_size]
                err_buf = entry['err'][:M, :block_size]
                W_buf = entry['W'][:M, :N]
                return q_buf, err_buf, W_buf
            
            self._miss_count += 1
        
        # Cache miss - create new buffers with rounded sizes
        M_r, N_r, block_r = self._round_dims(M, N, block_size)
        
        # Use pinned memory for CPU buffers to accelerate transfers
        pin_memory = (str(device) == 'cpu' and torch.cuda.is_available())
        
        try:
            q_buffer = torch.empty(M_r, block_r, dtype=torch.int8, device=device, pin_memory=pin_memory)
            err_buffer = torch.empty(M_r, block_r, dtype=torch.float32, device=device, pin_memory=pin_memory)
            W_buffer = torch.empty(M_r, N_r, dtype=torch.float32, device=device, pin_memory=pin_memory)
        except (RuntimeError, TypeError) as e:
            # If allocation fails or pin_memory is not supported, try with exact sizes and no pinning
            # First, try to reclaim memory if on CUDA
            if str(device) != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                q_buffer = torch.empty(M, block_size, dtype=torch.int8, device=device)
                err_buffer = torch.empty(M, block_size, dtype=torch.float32, device=device)
                W_buffer = torch.empty(M, N, dtype=torch.float32, device=device)
                M_r, N_r, block_r = M, N, block_size
            except Exception:
                # Re-raise the original error if fallback also fails
                raise e
        
        # Calculate allocation size for statistics
        alloc_mb = (
            q_buffer.numel() * 1 +  # int8
            err_buffer.numel() * 4 +  # float32
            W_buffer.numel() * 4  # float32
        ) / (1024 * 1024)
        self._total_allocated_mb += alloc_mb
        
        # Cache the full buffers
        entry = {
            'q': q_buffer,
            'err': err_buffer,
            'W': W_buffer,
            'M': M_r,
            'N': N_r,
            'block_size': block_r,
        }
        
        with self._lock:
            # Evict oldest if over limit
            while len(self._buffers) >= self.max_entries:
                self._eviction_count += 1
                self._buffers.popitem(last=False)  # Remove oldest
            
            self._buffers[key] = entry
        
        # Return slices
        return q_buffer[:M, :block_size], err_buffer[:M, :block_size], W_buffer[:M, :N]
    
    def release_buffers(
        self, 
        q_buf: torch.Tensor, 
        err_buf: torch.Tensor, 
        w_buf: torch.Tensor
    ) -> None:
        """
        Explicitly release buffers back to the pool.
        
        This is optional - buffers are automatically managed. However,
        calling this after use can help with memory tracking.
        
        Args:
            q_buf: Quantization buffer
            err_buf: Error buffer
            w_buf: Weight buffer
        """
        # Buffers are already in the pool, this is just for API compatibility
        pass
    
    def get_stats(self) -> Dict:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with statistics:
                - hit_count: Number of cache hits
                - miss_count: Number of cache misses
                - hit_rate: Cache hit rate (0.0 - 1.0)
                - cached_entries: Number of buffer sets in pool
                - evictions: Number of evicted buffer sets
                - total_allocated_mb: Total memory allocated
        """
        total = self._hit_count + self._miss_count
        return {
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': self._hit_count / max(1, total),
            'cached_entries': len(self._buffers),
            'max_entries': self.max_entries,
            'evictions': self._eviction_count,
            'total_allocated_mb': self._total_allocated_mb,
        }
    
    def clear(self) -> int:
        """
        Clear all cached buffers.
        
        Returns:
            Number of buffer sets cleared
        """
        with self._lock:
            count = len(self._buffers)
            self._buffers.clear()
            return count
    
    def resize(self, max_entries: int) -> None:
        """
        Resize the pool maximum entries.
        
        If the new size is smaller than current entries, oldest
        entries are evicted.
        
        Args:
            max_entries: New maximum number of entries
        """
        self.max_entries = max_entries
        
        with self._lock:
            while len(self._buffers) > self.max_entries:
                self._eviction_count += 1
                self._buffers.popitem(last=False)


# Global pool instance
_global_buffer_pool: Optional[LDLQBufferPool] = None
_pool_lock = threading.Lock()


def get_buffer_pool(
    max_entries: Optional[int] = None,
    force_new: bool = False
) -> LDLQBufferPool:
    """
    Get the global LDLQ buffer pool.
    
    Lazily initializes on first call. The global pool is shared
    across the application for maximum buffer reuse.
    
    Args:
        max_entries: Maximum pool entries (only used on first call)
        force_new: If True, create a new pool even if one exists
        
    Returns:
        The global LDLQBufferPool instance
        
    Example:
        >>> pool = get_buffer_pool()
        >>> q, err, w = pool.get_buffers(4096, 4096, 128, 'cuda')
    """
    global _global_buffer_pool
    
    with _pool_lock:
        if _global_buffer_pool is None or force_new:
            entries = max_entries or 8
            _global_buffer_pool = LDLQBufferPool(max_entries=entries)
        return _global_buffer_pool


def set_buffer_pool(pool: LDLQBufferPool) -> None:
    """
    Set a custom global buffer pool.
    
    Args:
        pool: LDLQBufferPool instance to use globally
    """
    global _global_buffer_pool
    with _pool_lock:
        _global_buffer_pool = pool


def clear_buffer_pool() -> int:
    """
    Clear all buffers from the global pool.
    
    Returns:
        Number of buffer sets cleared
    """
    global _global_buffer_pool
    if _global_buffer_pool is not None:
        return _global_buffer_pool.clear()
    return 0


def get_buffer_pool_stats() -> Optional[Dict]:
    """
    Get statistics for the global buffer pool.
    
    Returns:
        Dictionary of statistics or None if pool not initialized
    """
    global _global_buffer_pool
    if _global_buffer_pool is not None:
        return _global_buffer_pool.get_stats()
    return None
