"""Configuration for QuIP optimizations.

This module provides centralized configuration for memory and performance
optimizations in the QuIP INT8 quantization converter.

Environment Variables:
    QUIP_OPTIMIZATIONS: Global enable/disable ("1" or "0", default "1")
    QUIP_PINNED_POOL: Enable pinned memory pool ("1" or "0", default "1")
    QUIP_PINNED_POOL_MB: Max pool size in MB (default "512")
    QUIP_BUFFER_POOL: Enable buffer pool ("1" or "0", default "1")
    QUIP_BUFFER_POOL_ENTRIES: Max pool entries (default "8")
    QUIP_OOM_GUARD: Enable OOM guard ("1" or "0", default "1")
    QUIP_CHUNKED_HADAMARD: Enable chunked Hadamard ("1" or "0", default "1")
    QUIP_CHUNKED_HADAMARD_ROWS: Chunk size in rows (default "2048")
    QUIP_FP16_INTERMEDIATES: Enable FP16 storage ("1" or "0", default "1")
    QUIP_ASYNC_PREFETCH: Enable async layer prefetching ("1" or "0", default "1")
    QUIP_ASYNC_WORKERS: Number of async workers (default "1")
    QUIP_ASYNC_BATCH_SIZE: Layers per batch (default "1")
    QUIP_OUTLIER_AWARE: Enable outlier-aware scaling ("1" or "0", default "1")
    QUIP_OUTLIER_PERCENTILE: Percentile for scale calculation (default "1.0")
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class OptimizationConfig:
    """Configuration for QuIP INT8 optimizations.
    
    All optimizations are enabled by default but can be disabled individually
    via environment variables or programmatically.
    
    Attributes:
        enable_pinned_pool: Use pooled pinned memory for CPU→GPU transfers
        pinned_pool_max_mb: Maximum size of pinned memory pool
        enable_buffer_pool: Reuse quantization buffers across layers
        buffer_pool_max_entries: Maximum number of cached buffer sets
        enable_oom_guard: Check memory before large allocations
        oom_safety_factor: Multiplier for required memory check
        enable_chunked_hadamard: Process Hadamard in row chunks
        hadamard_chunk_rows: Number of rows per chunk
        hadamard_chunk_threshold: Element threshold for chunking
        enable_fp16_intermediates: Use FP16 for intermediate storage
        fp16_threshold_elements: Element threshold for FP16 mode
        enable_async_prefetch: Use async layer prefetching
        async_prefetch_workers: Number of prefetch worker threads
        async_prefetch_batch_size: Number of layers per prefetch batch
    """
    
    # Global enable/disable
    enable_optimizations: bool = True
    
    # Pinned Memory Pool
    enable_pinned_pool: bool = True
    pinned_pool_max_mb: int = 512
    pinned_pool_min_bucket: int = 1024  # 1KB
    pinned_pool_max_bucket: int = 134217728  # 128MB
    
    # Buffer Pool
    enable_buffer_pool: bool = True
    buffer_pool_max_entries: int = 8
    buffer_pool_rounding: int = 1024  # Round dims to nearest
    
    # OOM Guard
    enable_oom_guard: bool = True
    oom_safety_factor: float = 1.25
    oom_min_free_mb: int = 512  # 512MB minimum free
    
    # Chunked Hadamard
    enable_chunked_hadamard: bool = True
    hadamard_chunk_rows: int = 2048
    hadamard_chunk_threshold: int = 33_000_000  # elements
    
    # FP16 Intermediates
    enable_fp16_intermediates: bool = True
    fp16_threshold_elements: int = 50_000_000  # elements
    
    # Async Prefetching
    enable_async_prefetch: bool = True
    async_prefetch_workers: int = 1
    async_prefetch_batch_size: int = 1
    
    # Outlier-aware scaling
    enable_outlier_aware_scaling: bool = True
    outlier_percentile: float = 1.0  # Default to 1.0 (disabled) for QuIP precision
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.pinned_pool_max_mb < 64:
            raise ValueError(f"pinned_pool_max_mb must be >= 64, got {self.pinned_pool_max_mb}")
        if self.buffer_pool_max_entries < 1:
            raise ValueError(f"buffer_pool_max_entries must be >= 1, got {self.buffer_pool_max_entries}")
        if self.oom_safety_factor < 1.0:
            raise ValueError(f"oom_safety_factor must be >= 1.0, got {self.oom_safety_factor}")
        if self.hadamard_chunk_rows < 512:
            raise ValueError(f"hadamard_chunk_rows must be >= 512, got {self.hadamard_chunk_rows}")
        if self.async_prefetch_workers < 1:
            raise ValueError(f"async_prefetch_workers must be >= 1, got {self.async_prefetch_workers}")
        if self.async_prefetch_batch_size < 1:
            raise ValueError(f"async_prefetch_batch_size must be >= 1, got {self.async_prefetch_batch_size}")
        if not 0.5 <= self.outlier_percentile <= 1.0:
            raise ValueError(f"outlier_percentile must be between 0.5 and 1.0, got {self.outlier_percentile}")
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Create configuration from environment variables.
        
        Reads all QUIP_* environment variables and constructs a config.
        Unset variables use default values.
        
        Returns:
            OptimizationConfig with values from environment
        """
        def get_bool(name: str, default: str) -> bool:
            return os.getenv(name, default) == '1'
        
        def get_int(name: str, default: str) -> int:
            try:
                return int(os.getenv(name, default))
            except ValueError:
                return int(default)
        
        def get_float(name: str, default: str) -> float:
            try:
                return float(os.getenv(name, default))
            except ValueError:
                return float(default)
        
        # Check global disable first
        global_enable = get_bool('QUIP_OPTIMIZATIONS', '1')
        
        if not global_enable:
            return cls.disable_all()
        
        return cls(
            enable_optimizations=True,
            enable_pinned_pool=get_bool('QUIP_PINNED_POOL', '1'),
            pinned_pool_max_mb=get_int('QUIP_PINNED_POOL_MB', '512'),
            enable_buffer_pool=get_bool('QUIP_BUFFER_POOL', '1'),
            buffer_pool_max_entries=get_int('QUIP_BUFFER_POOL_ENTRIES', '8'),
            enable_oom_guard=get_bool('QUIP_OOM_GUARD', '1'),
            oom_safety_factor=get_float('QUIP_OOM_SAFETY_FACTOR', '1.25'),
            oom_min_free_mb=get_int('QUIP_OOM_MIN_FREE_MB', '512'),
            enable_chunked_hadamard=get_bool('QUIP_CHUNKED_HADAMARD', '1'),
            hadamard_chunk_rows=get_int('QUIP_CHUNKED_HADAMARD_ROWS', '2048'),
            hadamard_chunk_threshold=get_int('QUIP_CHUNKED_HADAMARD_THRESHOLD', '33000000'),
            enable_fp16_intermediates=get_bool('QUIP_FP16_INTERMEDIATES', '1'),
            fp16_threshold_elements=get_int('QUIP_FP16_THRESHOLD', '50000000'),
            enable_async_prefetch=get_bool('QUIP_ASYNC_PREFETCH', '1'),
            async_prefetch_workers=get_int('QUIP_ASYNC_WORKERS', '1'),
            async_prefetch_batch_size=get_int('QUIP_ASYNC_BATCH_SIZE', '1'),
            enable_outlier_aware_scaling=get_bool('QUIP_OUTLIER_AWARE', '1'),
            outlier_percentile=get_float('QUIP_OUTLIER_PERCENTILE', '1.0'),
        )
    
    @classmethod
    def disable_all(cls) -> 'OptimizationConfig':
        """Create configuration with all optimizations disabled.
        
        Useful for debugging or benchmarking baseline performance.
        
        Returns:
            OptimizationConfig with all optimizations disabled
        """
        return cls(
            enable_optimizations=False,
            enable_pinned_pool=False,
            enable_buffer_pool=False,
            enable_oom_guard=False,
            enable_chunked_hadamard=False,
            enable_fp16_intermediates=False,
            enable_async_prefetch=False,
        )
    
    @classmethod
    def maximum_performance(cls) -> 'OptimizationConfig':
        """Create configuration optimized for maximum performance.
        
        Enables all optimizations with aggressive settings.
        May use more memory for better speed.
        
        Returns:
            OptimizationConfig for maximum performance
        """
        return cls(
            enable_optimizations=True,
            enable_pinned_pool=True,
            pinned_pool_max_mb=1024,
            enable_buffer_pool=True,
            buffer_pool_max_entries=16,
            enable_oom_guard=True,
            enable_chunked_hadamard=True,
            hadamard_chunk_rows=4096,
            enable_fp16_intermediates=True,
            enable_async_prefetch=True,
            async_prefetch_workers=2,
            async_prefetch_batch_size=1,
        )
    
    @classmethod
    def minimum_memory(cls) -> 'OptimizationConfig':
        """Create configuration optimized for minimum memory usage.
        
        Uses more aggressive memory-saving settings.
        May be slower but uses less VRAM.
        
        Returns:
            OptimizationConfig for minimum memory
        """
        return cls(
            enable_optimizations=True,
            enable_pinned_pool=True,
            pinned_pool_max_mb=256,
            enable_buffer_pool=True,
            buffer_pool_max_entries=4,
            enable_oom_guard=True,
            oom_safety_factor=1.5,
            enable_chunked_hadamard=True,
            hadamard_chunk_rows=1024,
            hadamard_chunk_threshold=10_000_000,
            enable_fp16_intermediates=True,
            fp16_threshold_elements=10_000_000,
            enable_async_prefetch=False,  # Disable to save memory
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'enable_optimizations': self.enable_optimizations,
            'enable_pinned_pool': self.enable_pinned_pool,
            'pinned_pool_max_mb': self.pinned_pool_max_mb,
            'enable_buffer_pool': self.enable_buffer_pool,
            'buffer_pool_max_entries': self.buffer_pool_max_entries,
            'enable_oom_guard': self.enable_oom_guard,
            'oom_safety_factor': self.oom_safety_factor,
            'oom_min_free_mb': self.oom_min_free_mb,
            'enable_chunked_hadamard': self.enable_chunked_hadamard,
            'hadamard_chunk_rows': self.hadamard_chunk_rows,
            'hadamard_chunk_threshold': self.hadamard_chunk_threshold,
            'enable_fp16_intermediates': self.enable_fp16_intermediates,
            'fp16_threshold_elements': self.fp16_threshold_elements,
            'enable_async_prefetch': self.enable_async_prefetch,
            'async_prefetch_workers': self.async_prefetch_workers,
            'async_prefetch_batch_size': self.async_prefetch_batch_size,
            'enable_outlier_aware_scaling': self.enable_outlier_aware_scaling,
            'outlier_percentile': self.outlier_percentile,
        }
    
    def __str__(self) -> str:
        """String representation showing enabled optimizations."""
        if not self.enable_optimizations:
            return "OptimizationConfig(all_disabled)"
        
        enabled = []
        if self.enable_pinned_pool:
            enabled.append(f"pinned_pool({self.pinned_pool_max_mb}MB)")
        if self.enable_buffer_pool:
            enabled.append(f"buffer_pool({self.buffer_pool_max_entries})")
        if self.enable_oom_guard:
            enabled.append("oom_guard")
        if self.enable_chunked_hadamard:
            enabled.append(f"chunked_hadamard({self.hadamard_chunk_rows})")
        if self.enable_fp16_intermediates:
            enabled.append("fp16_intermediates")
        if self.enable_async_prefetch:
            enabled.append(f"async_prefetch({self.async_prefetch_workers}w)")
        if self.enable_outlier_aware_scaling:
            enabled.append(f"outlier_aware({self.outlier_percentile})")
        
        return f"OptimizationConfig({', '.join(enabled)})"


# Global configuration instance
_global_config: Optional[OptimizationConfig] = None


def get_optimization_config() -> OptimizationConfig:
    """Get the global optimization configuration.
    
    Lazily initializes from environment variables on first call.
    
    Returns:
        The global OptimizationConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = OptimizationConfig.from_env()
    return _global_config


def set_optimization_config(config: OptimizationConfig) -> None:
    """Set the global optimization configuration.
    
    Args:
        config: New configuration to use globally
    """
    global _global_config
    _global_config = config


def reset_optimization_config() -> None:
    """Reset global configuration to defaults from environment."""
    global _global_config
    _global_config = OptimizationConfig.from_env()
