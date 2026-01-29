"""Async tensor prefetching for quantization workflows.

Overlaps tensor loading from disk with GPU quantization computation.
While tensor N is being quantized, tensor N+1 is loaded from disk to CPU.
"""
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Callable, Tuple, Any
import torch
import threading
import time
import logging

from convert_to_quant.utils.pinned_pool import transfer_with_pool

logger = logging.getLogger(__name__)


class TensorPrefetchStats:
    """Statistics for tensor prefetching."""
    
    def __init__(self):
        self.tensors_loaded = 0
        self.tensors_prefetched = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_wait_time_ms = 0.0
        self.total_load_time_ms = 0.0
    
    def record_prefetch_hit(self):
        self.prefetch_hits += 1
    
    def record_prefetch_miss(self):
        self.prefetch_misses += 1
    
    def record_wait_time(self, ms: float):
        self.total_wait_time_ms += ms
    
    def record_load_time(self, ms: float):
        self.total_load_time_ms += ms
    
    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / max(1, total)
    
    @property
    def avg_wait_ms(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.total_wait_time_ms / max(1, total)
    
    def __str__(self) -> str:
        return (
            f"TensorPrefetchStats(hit_rate={self.hit_rate:.1%}, "
            f"avg_wait={self.avg_wait_ms:.1f}ms, "
            f"tensors={self.tensors_loaded})"
        )


class AsyncTensorPrefetcher:
    """Asynchronously prefetch tensors from a loader while processing current tensor.
    
    This overlaps disk I/O (loading next tensor) with GPU computation 
    (quantizing current tensor) for better throughput.
    
    Args:
        loader: Tensor loader with get_tensor() method (e.g., MemoryEfficientSafeOpen)
        keys: List of tensor keys to process in order
        device: Target device for tensor transfers
        max_workers: Number of prefetch threads (default 1)
        enable_pin_memory: Use pinned memory for GPU transfers
        
    Example:
        >>> with MemoryEfficientSafeOpen("model.safetensors") as loader:
        ...     prefetcher = AsyncTensorPrefetcher(
        ...         loader, weight_keys, device='cuda'
        ...     )
        ...     for key in prefetcher:
        ...         # Tensor is already on GPU here
        ...         quantize_tensor(key, prefetcher.current_tensor())
    """
    
    def __init__(
        self,
        loader: Any,
        keys: List[str],
        device: str = 'cuda',
        max_workers: int = 1,
        enable_pin_memory: bool = True
    ):
        self.loader = loader
        self.keys = keys
        self.device = device
        self.max_workers = max_workers
        self.enable_pin_memory = enable_pin_memory and torch.cuda.is_available()
        
        self._current_idx = -1
        self._current_tensor: Optional[torch.Tensor] = None
        self._current_key: Optional[str] = None
        self._prefetch_future: Optional[Future] = None
        self._prefetched_key: Optional[str] = None
        self._prefetched_tensor: Optional[torch.Tensor] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stats = TensorPrefetchStats()
        self._lock = threading.Lock()
        self._closed = False
        
        if len(keys) > 0 and max_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.debug(f"Initialized AsyncTensorPrefetcher with {max_workers} workers")
    
    def _load_tensor(self, key: str) -> Tuple[str, torch.Tensor]:
        """Load a tensor from the loader.
        
        Runs in background thread.
        
        Returns:
            (key, tensor_on_device)
        """
        start_time = time.time()
        
        # Load from disk (CPU)
        tensor = self.loader.get_tensor(key)
        
        # Transfer to target device
        if tensor.device.type == 'cpu' and self.device.startswith('cuda'):
            if self.enable_pin_memory:
                # Use pinned memory for async transfer
                tensor = transfer_with_pool(tensor, device=self.device, non_blocking=True)
            else:
                tensor = tensor.to(device=self.device)
        
        load_time = (time.time() - start_time) * 1000
        self._stats.record_load_time(load_time)
        
        return key, tensor
    
    def prefetch_next(self) -> bool:
        """Start prefetching the next tensor.
        
        Returns:
            True if prefetch started, False if no more tensors
        """
        if self._closed:
            raise RuntimeError("AsyncTensorPrefetcher has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.keys):
            return False
        
        # Cancel any existing prefetch
        if self._prefetch_future is not None:
            if not self._prefetch_future.done():
                self._prefetch_future.cancel()
            self._prefetch_future = None
        
        # Start new prefetch
        next_key = self.keys[next_idx]
        if self._executor is not None:
            self._prefetch_future = self._executor.submit(self._load_tensor, next_key)
            self._prefetched_key = next_key
            self._stats.tensors_prefetched += 1
            logger.debug(f"Started prefetching tensor {next_key}")
            return True
        return False
    
    def current(self) -> Tuple[str, torch.Tensor]:
        """Get current key and tensor.
        
        On first call, loads first tensor synchronously.
        Subsequent calls return cached values.
        
        Returns:
            (key, tensor_on_device)
        """
        if self._closed:
            raise RuntimeError("AsyncTensorPrefetcher has been closed")
        
        if self._current_idx < 0:
            # First call - load synchronously
            self._current_idx = 0
            key = self.keys[0]
            _, tensor = self._load_tensor(key)
            self._current_key = key
            self._current_tensor = tensor
            self._stats.tensors_loaded += 1
            
            # Trigger prefetch of next
            if len(self.keys) > 1:
                self.prefetch_next()
        
        return self._current_key, self._current_tensor
    
    def advance(self) -> Optional[Tuple[str, torch.Tensor]]:
        """Move to next tensor and return it.
        
        Returns:
            (key, tensor) or None if no more tensors
        """
        if self._closed:
            raise RuntimeError("AsyncTensorPrefetcher has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.keys):
            self._current_idx = len(self.keys)
            self._current_key = None
            self._current_tensor = None
            return None
        
        wait_start = time.time()
        
        # Check if we have a prefetched result
        if self._prefetch_future is not None:
            try:
                key, tensor = self._prefetch_future.result()
                self._stats.record_prefetch_hit()
                logger.debug(f"Tensor {key} was prefetched")
            except Exception as e:
                logger.warning(f"Prefetch failed: {e}")
                # Fall back to synchronous load
                key = self.keys[next_idx]
                _, tensor = self._load_tensor(key)
                self._stats.record_prefetch_miss()
            finally:
                self._prefetch_future = None
                self._prefetched_key = None
        else:
            # No prefetch, load synchronously
            key = self.keys[next_idx]
            _, tensor = self._load_tensor(key)
            self._stats.record_prefetch_miss()
        
        wait_time = (time.time() - wait_start) * 1000
        self._stats.record_wait_time(wait_time)
        
        self._current_idx = next_idx
        self._current_key = key
        self._current_tensor = tensor
        self._stats.tensors_loaded += 1
        
        # Trigger prefetch of next
        if next_idx + 1 < len(self.keys):
            self.prefetch_next()
        
        return key, tensor
    
    def __iter__(self):
        """Make prefetcher iterable.
        
        Yields:
            (key, tensor) pairs
        """
        self._current_idx = -1
        self._current_key = None
        self._current_tensor = None
        
        try:
            key, tensor = self.current()
            while key is not None:
                yield key, tensor
                result = self.advance()
                if result is None:
                    break
                key, tensor = result
        finally:
            pass  # Don't close here
    
    def current_key(self) -> Optional[str]:
        """Get current key without loading."""
        if self._current_idx < 0:
            return None
        return self._current_key
    
    def current_tensor(self) -> Optional[torch.Tensor]:
        """Get current tensor."""
        return self._current_tensor
    
    def close(self):
        """Close prefetcher and cleanup."""
        if self._closed:
            return
        
        if self._prefetch_future is not None:
            if not self._prefetch_future.done():
                self._prefetch_future.cancel()
            self._prefetch_future = None
        
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    @property
    def stats(self) -> TensorPrefetchStats:
        """Get prefetching statistics."""
        return self._stats
    
    @property
    def current_index(self) -> int:
        """Current tensor index."""
        return self._current_idx
    
    @property
    def num_tensors(self) -> int:
        """Total number of tensors."""
        return len(self.keys)