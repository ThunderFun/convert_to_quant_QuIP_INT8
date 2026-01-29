"""Async prefetching for layer-by-layer quantization.

This module provides asynchronous layer loading to overlap data transfer
with GPU computation. Uses double buffering to prefetch the next layer
while the current layer is being quantized.

Example:
    >>> from convert_to_quant.utils.async_prefetch import AsyncLayerLoader
    >>> loader = AsyncLayerLoader(model_layers, device='cuda')
    >>> 
    >>> for i in range(len(model_layers)):
    ...     # Process current layer
    ...     quantize_layer(loader.current())
    ...     
    ...     # Prefetch next layer (non-blocking)
    ...     loader.prefetch_next()
    ...     
    ...     # Move to next (may wait if prefetch not done)
    ...     loader.advance()
"""
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Callable, Any, Dict
import torch
import threading
import time
import logging

from convert_to_quant.utils.pinned_pool import get_pinned_pool, transfer_with_pool

logger = logging.getLogger(__name__)


class PrefetchStats:
    """Statistics for async prefetching operations."""
    
    def __init__(self):
        self.layers_loaded = 0
        self.layers_prefetched = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_wait_time_ms = 0.0
        self.total_load_time_ms = 0.0
        self._load_start_time: Optional[float] = None
    
    def record_prefetch_hit(self):
        """Record when a prefetched layer was ready."""
        self.prefetch_hits += 1
    
    def record_prefetch_miss(self):
        """Record when we had to wait for a layer to load."""
        self.prefetch_misses += 1
    
    def record_wait_time(self, ms: float):
        """Record time spent waiting for a layer."""
        self.total_wait_time_ms += ms
    
    def record_load_time(self, ms: float):
        """Record time spent loading a layer."""
        self.total_load_time_ms += ms
    
    @property
    def hit_rate(self) -> float:
        """Prefetch hit rate (0.0 to 1.0)."""
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / max(1, total)
    
    @property
    def avg_wait_ms(self) -> float:
        """Average wait time per layer."""
        total = self.prefetch_hits + self.prefetch_misses
        return self.total_wait_time_ms / max(1, total)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'layers_loaded': self.layers_loaded,
            'layers_prefetched': self.layers_prefetched,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'hit_rate': self.hit_rate,
            'total_wait_time_ms': self.total_wait_time_ms,
            'total_load_time_ms': self.total_load_time_ms,
            'avg_wait_ms': self.avg_wait_ms,
        }
    
    def __str__(self) -> str:
        return (
            f"PrefetchStats(hit_rate={self.hit_rate:.1%}, "
            f"avg_wait={self.avg_wait_ms:.1f}ms, "
            f"layers={self.layers_loaded})"
        )


class AsyncLayerLoader:
    """Asynchronously prefetch next layer while processing current layer.
    
    Uses a background thread to load and transfer the next layer to GPU
    while the main thread quantizes the current layer. This overlaps
    I/O operations with GPU computation for better throughput.
    
    Args:
        layers: List of layers to load (can be nn.Module or any object with weights)
        device: Target device for layer transfers (default 'cuda')
        max_workers: Number of background threads (default 1)
        enable_pin_memory: Use pinned memory for faster transfers (default True)
        prefetch_ahead: Number of layers to prefetch ahead (default 1)
        
    Example:
        >>> loader = AsyncLayerLoader(model_layers, device='cuda')
        >>> 
        >>> # Get first layer (triggers prefetch of layer 1)
        >>> current = loader.current()
        >>> 
        >>> for i in range(len(model_layers)):
        ...     # Quantize current layer (GPU compute)
        ...     quantize_layer(current)
        ...     
        ...     # Start prefetching next layer
        ...     loader.prefetch_next()
        ...     
        ...     # Get next layer (waits if prefetch not done)
        ...     current = loader.advance()
        >>> 
        >>> # Get statistics
        >>> print(loader.stats)
    """
    
    def __init__(
        self,
        layers: List[Any],
        device: str = 'cuda',
        max_workers: int = 1,
        enable_pin_memory: bool = True,
        prefetch_ahead: int = 1
    ):
        self.layers = layers
        self.device = device
        self.max_workers = max_workers
        self.enable_pin_memory = enable_pin_memory and torch.cuda.is_available()
        self.prefetch_ahead = prefetch_ahead
        
        self._current_idx = -1
        self._current_layer: Optional[Any] = None
        self._prefetched_layer: Optional[Any] = None
        self._prefetch_future: Optional[Future] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stats = PrefetchStats()
        self._lock = threading.Lock()
        self._closed = False
        
        # Initialize executor if we have layers
        if len(layers) > 0 and max_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.debug(f"Initialized AsyncLayerLoader with {max_workers} workers")
    
    def _load_layer(self, idx: int) -> Any:
        """Load a single layer and transfer to device.
        
        This runs in the background thread.
        
        Args:
            idx: Index of layer to load
            
        Returns:
            Layer on target device
        """
        if idx < 0 or idx >= len(self.layers):
            return None
        
        start_time = time.time()
        layer = self.layers[idx]
        
        # Transfer layer to device if needed
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight = layer.weight
            if weight.device.type == 'cpu':
                if self.enable_pin_memory:
                    # Use pinned memory pool for async transfer
                    weight = transfer_with_pool(
                        weight,
                        device=self.device,
                        non_blocking=True
                    )
                else:
                    # Standard transfer
                    weight = weight.to(device=self.device)
                # Update layer weight
                layer.weight = torch.nn.Parameter(weight)
        
        # Transfer bias if present
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias = layer.bias
            if bias.device.type == 'cpu':
                if self.enable_pin_memory:
                    bias = transfer_with_pool(
                        bias,
                        device=self.device,
                        non_blocking=True
                    )
                else:
                    bias = bias.to(device=self.device)
                layer.bias = torch.nn.Parameter(bias)
        
        # For generic objects, try to move entire object
        if hasattr(layer, 'to') and callable(getattr(layer, 'to')):
            try:
                layer = layer.to(device=self.device)
            except Exception:
                pass  # Some layers don't support .to()
        
        load_time = (time.time() - start_time) * 1000
        self._stats.record_load_time(load_time)
        
        return layer
    
    def prefetch_next(self) -> bool:
        """Start prefetching the next layer in background.
        
        This is non-blocking and returns immediately. The actual
        loading happens in a background thread.
        
        Returns:
            True if prefetch was started, False if no more layers
            
        Example:
            >>> loader.prefetch_next()  # Start loading next layer
            >>> # ... do other work ...
            >>> next_layer = loader.advance()  # May return immediately
        """
        if self._closed:
            raise RuntimeError("AsyncLayerLoader has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.layers):
            return False
        
        # Cancel any existing prefetch
        if self._prefetch_future is not None:
            if not self._prefetch_future.done():
                self._prefetch_future.cancel()
            self._prefetch_future = None
        
        # Start new prefetch
        if self._executor is not None:
            self._prefetch_future = self._executor.submit(self._load_layer, next_idx)
            self._stats.layers_prefetched += 1
            logger.debug(f"Started prefetching layer {next_idx}")
            return True
        else:
            # No executor, will load synchronously
            return False
    
    def current(self) -> Any:
        """Get the current layer.
        
        On first call, loads the first layer synchronously.
        Subsequent calls return the same layer until advance() is called.
        
        Returns:
            Current layer on target device
            
        Raises:
            IndexError: If no layers available
        """
        if self._closed:
            raise RuntimeError("AsyncLayerLoader has been closed")
        
        if self._current_idx < 0:
            # First call - load first layer synchronously
            self._current_idx = 0
            self._current_layer = self._load_layer(0)
            self._stats.layers_loaded += 1
            
            # Trigger prefetch of next layer
            if len(self.layers) > 1:
                self.prefetch_next()
        
        if self._current_layer is None:
            raise IndexError("No current layer available")
        
        return self._current_layer
    
    def advance(self) -> Optional[Any]:
        """Move to the next layer and return it.
        
        If the next layer has been prefetched, returns immediately.
        Otherwise, waits for the layer to load.
        
        Returns:
            Next layer on target device, or None if no more layers
            
        Example:
            >>> loader = AsyncLayerLoader(layers)
            >>> current = loader.current()  # First layer
            >>> while current is not None:
            ...     process_layer(current)
            ...     loader.prefetch_next()  # Start loading next
            ...     current = loader.advance()  # Get next layer
        """
        if self._closed:
            raise RuntimeError("AsyncLayerLoader has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.layers):
            self._current_idx = len(self.layers)
            self._current_layer = None
            return None
        
        wait_start = time.time()
        
        # Check if we have a prefetched result
        if self._prefetch_future is not None:
            try:
                layer = self._prefetch_future.result()  # Wait if not done
                self._stats.record_prefetch_hit()
                logger.debug(f"Layer {next_idx} was prefetched")
            except Exception as e:
                logger.warning(f"Prefetch failed for layer {next_idx}: {e}")
                # Fall back to synchronous load
                layer = self._load_layer(next_idx)
                self._stats.record_prefetch_miss()
            finally:
                self._prefetch_future = None
        else:
            # No prefetch, load synchronously
            layer = self._load_layer(next_idx)
            self._stats.record_prefetch_miss()
        
        wait_time = (time.time() - wait_start) * 1000
        self._stats.record_wait_time(wait_time)
        
        self._current_idx = next_idx
        self._current_layer = layer
        self._stats.layers_loaded += 1
        
        # Trigger prefetch of next layer
        if next_idx + 1 < len(self.layers):
            self.prefetch_next()
        
        return layer
    
    def __iter__(self):
        """Make loader iterable.
        
        Yields:
            Layers in order, with automatic prefetching
            
        Example:
            >>> loader = AsyncLayerLoader(layers)
            >>> for layer in loader:
            ...     process_layer(layer)
        """
        self._current_idx = -1
        self._current_layer = None
        
        try:
            layer = self.current()
            while layer is not None:
                yield layer
                self.prefetch_next()
                layer = self.advance()
        finally:
            pass  # Don't close here, let user close explicitly
    
    def reset(self):
        """Reset loader to beginning.
        
        Clears any pending prefetches and resets to initial state.
        """
        if self._prefetch_future is not None:
            if not self._prefetch_future.done():
                self._prefetch_future.cancel()
            self._prefetch_future = None
        
        self._current_idx = -1
        self._current_layer = None
        self._prefetched_layer = None
        self._stats = PrefetchStats()
    
    def close(self):
        """Close the loader and cleanup resources.
        
        Shuts down the background executor. Should be called when
done using the loader.
        """
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
        logger.debug("AsyncLayerLoader closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    @property
    def stats(self) -> PrefetchStats:
        """Get prefetching statistics."""
        return self._stats
    
    @property
    def current_index(self) -> int:
        """Current layer index."""
        return self._current_idx
    
    @property
    def num_layers(self) -> int:
        """Total number of layers."""
        return len(self.layers)
    
    @property
    def is_prefetching(self) -> bool:
        """True if a prefetch is currently in progress."""
        return self._prefetch_future is not None and not self._prefetch_future.done()


class BatchedAsyncLoader:
    """Async loader for batch quantization scenarios.
    
    Extends AsyncLayerLoader with support for processing layers
    in groups/batches, prefetching entire batches ahead of time.
    
    Args:
        layer_groups: List of layer groups (each group is a list of layers)
        device: Target device
        max_workers: Number of background threads
        enable_pin_memory: Use pinned memory transfers
        
    Example:
        >>> groups = [[layer1, layer2], [layer3, layer4], ...]
        >>> loader = BatchedAsyncLoader(groups)
        >>> for batch in loader:
        ...     process_batch(batch)  # Each batch is a list of layers
    """
    
    def __init__(
        self,
        layer_groups: List[List[Any]],
        device: str = 'cuda',
        max_workers: int = 1,
        enable_pin_memory: bool = True
    ):
        self.layer_groups = layer_groups
        self.device = device
        self.max_workers = max_workers
        self.enable_pin_memory = enable_pin_memory
        
        self._current_idx = -1
        self._current_batch: Optional[List[Any]] = None
        self._prefetch_future: Optional[Future] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stats = PrefetchStats()
        self._closed = False
        
        if len(layer_groups) > 0 and max_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _load_batch(self, idx: int) -> List[Any]:
        """Load a batch of layers."""
        if idx < 0 or idx >= len(self.layer_groups):
            return []
        
        start_time = time.time()
        batch = self.layer_groups[idx]
        loaded = []
        
        for layer in batch:
            # Transfer layer to device
            if hasattr(layer, 'weight') and layer.weight is not None:
                weight = layer.weight
                if weight.device.type == 'cpu':
                    if self.enable_pin_memory:
                        weight = transfer_with_pool(
                            weight,
                            device=self.device,
                            non_blocking=True
                        )
                    else:
                        weight = weight.to(device=self.device)
                    layer.weight = torch.nn.Parameter(weight)
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias
                if bias.device.type == 'cpu':
                    if self.enable_pin_memory:
                        bias = transfer_with_pool(
                            bias,
                            device=self.device,
                            non_blocking=True
                        )
                    else:
                        bias = bias.to(device=self.device)
                    layer.bias = torch.nn.Parameter(bias)
            
            loaded.append(layer)
        
        load_time = (time.time() - start_time) * 1000
        self._stats.record_load_time(load_time)
        
        return loaded
    
    def prefetch_next(self) -> bool:
        """Prefetch next batch."""
        if self._closed:
            raise RuntimeError("BatchedAsyncLoader has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.layer_groups):
            return False
        
        if self._prefetch_future is not None:
            if not self._prefetch_future.done():
                self._prefetch_future.cancel()
            self._prefetch_future = None
        
        if self._executor is not None:
            self._prefetch_future = self._executor.submit(self._load_batch, next_idx)
            self._stats.layers_prefetched += 1
            return True
        return False
    
    def current(self) -> List[Any]:
        """Get current batch."""
        if self._closed:
            raise RuntimeError("BatchedAsyncLoader has been closed")
        
        if self._current_idx < 0:
            self._current_idx = 0
            self._current_batch = self._load_batch(0)
            self._stats.layers_loaded += len(self._current_batch)
            
            if len(self.layer_groups) > 1:
                self.prefetch_next()
        
        return self._current_batch or []
    
    def advance(self) -> Optional[List[Any]]:
        """Move to next batch."""
        if self._closed:
            raise RuntimeError("BatchedAsyncLoader has been closed")
        
        next_idx = self._current_idx + 1
        if next_idx >= len(self.layer_groups):
            self._current_idx = len(self.layer_groups)
            self._current_batch = None
            return None
        
        wait_start = time.time()
        
        if self._prefetch_future is not None:
            try:
                batch = self._prefetch_future.result()
                self._stats.record_prefetch_hit()
            except Exception as e:
                logger.warning(f"Batch prefetch failed: {e}")
                batch = self._load_batch(next_idx)
                self._stats.record_prefetch_miss()
            finally:
                self._prefetch_future = None
        else:
            batch = self._load_batch(next_idx)
            self._stats.record_prefetch_miss()
        
        wait_time = (time.time() - wait_start) * 1000
        self._stats.record_wait_time(wait_time)
        
        self._current_idx = next_idx
        self._current_batch = batch
        self._stats.layers_loaded += len(batch)
        
        if next_idx + 1 < len(self.layer_groups):
            self.prefetch_next()
        
        return batch
    
    def __iter__(self):
        """Make loader iterable."""
        self._current_idx = -1
        self._current_batch = None
        
        try:
            batch = self.current()
            while batch:
                yield batch
                self.prefetch_next()
                batch = self.advance()
        finally:
            pass
    
    def close(self):
        """Close loader and cleanup."""
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
    def stats(self) -> PrefetchStats:
        """Get statistics."""
        return self._stats


def create_async_loader(
    layers: List[Any],
    device: str = 'cuda',
    batch_size: int = 1,
    enable_async: bool = True
) -> Any:
    """Factory function to create appropriate async loader.
    
    Args:
        layers: List of layers to load
        device: Target device
        batch_size: Number of layers per batch (1 = layer-by-layer)
        enable_async: Enable async prefetching
        
    Returns:
        AsyncLayerLoader or BatchedAsyncLoader instance
        
    Example:
        >>> loader = create_async_loader(layers, batch_size=4)
        >>> for batch in loader:
        ...     process_batch(batch)
    """
    if not enable_async or not torch.cuda.is_available():
        # Return a synchronous wrapper
        return _SynchronousLoader(layers, device, batch_size)
    
    if batch_size == 1:
        return AsyncLayerLoader(layers, device=device)
    else:
        # Group layers into batches
        groups = [
            layers[i:i + batch_size]
            for i in range(0, len(layers), batch_size)
        ]
        return BatchedAsyncLoader(groups, device=device)


class _SynchronousLoader:
    """Synchronous fallback loader when async is disabled."""
    
    def __init__(self, layers: List[Any], device: str, batch_size: int):
        self.layers = layers
        self.device = device
        self.batch_size = batch_size
        self._idx = -1
        self._stats = PrefetchStats()
    
    def _load(self, idx: int):
        if idx >= len(self.layers):
            return None
        
        if self.batch_size == 1:
            layer = self.layers[idx]
            if hasattr(layer, 'to'):
                layer = layer.to(device=self.device)
            return layer
        else:
            batch = self.layers[idx:idx + self.batch_size]
            return [l.to(device=self.device) if hasattr(l, 'to') else l for l in batch]
    
    def current(self):
        if self._idx < 0:
            self._idx = 0
            return self._load(0)
        return self._load(self._idx)
    
    def advance(self):
        self._idx += 1
        if self._idx >= len(self.layers):
            return None
        return self._load(self._idx)
    
    def __iter__(self):
        for i in range(len(self.layers)):
            yield self._load(i)
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    @property
    def stats(self):
        return self._stats