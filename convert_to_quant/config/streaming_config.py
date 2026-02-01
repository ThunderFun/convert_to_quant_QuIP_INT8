"""Streaming configuration for memory-efficient quantization."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple
import torch
import time
import sys

# Import BF16 check from constants
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from constants import _get_bf16_if_supported


@dataclass
class StreamingThresholds:
    """Thresholds for CPU offloading decisions."""
    hadamard_elements: Union[int, float]  # Elements threshold for Hadamard transform
    hessian_dim: Union[int, float]        # Dimension threshold for Hessian inversion
    ldlq_dim: Union[int, float]           # Dimension threshold for LDLQ quantization
    ortho_dim: Union[int, float]          # Dimension threshold for orthogonal matrix generation


class StreamingConfig:
    """Configuration for streaming modes."""
    
    # Predefined tier thresholds
    TIERS: Dict[str, StreamingThresholds] = {
        'off': StreamingThresholds(
            hadamard_elements=10**15,
            hessian_dim=10**15,
            ldlq_dim=10**15,
            ortho_dim=10**15
        ),
        'minimal': StreamingThresholds(
            hadamard_elements=150_000_000,  # ~600MB
            hessian_dim=32768,              # Increased: Block Cholesky handles large N
            ldlq_dim=65536,                 # Increased: Tiled propagation handles large N
            ortho_dim=16384
        ),
        'balanced': StreamingThresholds(
            hadamard_elements=100_000_000,  # ~400MB
            hessian_dim=24576,              # Increased: Block Cholesky handles large N
            ldlq_dim=49152,                 # Increased: Tiled propagation handles large N
            ortho_dim=12288
        ),
        'aggressive': StreamingThresholds(
            hadamard_elements=25_000_000,   # ~100MB
            hessian_dim=8192,               # Increased: Block Cholesky handles large N
            ldlq_dim=16384,                 # Increased: Tiled propagation handles large N
            ortho_dim=4096
        ),
        'extreme': StreamingThresholds(
            hadamard_elements=12_500_000,   # ~50MB - very aggressive offloading
            hessian_dim=2048,
            ldlq_dim=4096,
            ortho_dim=2048
        ),
    }
    
    @classmethod
    def get_thresholds(cls, tier: str) -> StreamingThresholds:
        """Get thresholds for a named tier.
        
        Args:
            tier: One of 'off', 'minimal', 'balanced', 'aggressive'
            
        Returns:
            StreamingThresholds for the tier
            
        Raises:
            ValueError: If tier is not recognized
        """
        if tier not in cls.TIERS:
            raise ValueError(
                f"Unknown streaming tier: {tier}. "
                f"Choose from {list(cls.TIERS.keys())}"
            )
        return cls.TIERS[tier]
    
    @classmethod
    def auto_detect_tier(cls, vram_gb: Optional[float] = None) -> str:
        """Auto-detect optimal tier based on available VRAM.
        
        With memory optimizations (pinned pool, buffer pool, OOM guard,
        chunked Hadamard, FP16 intermediates, BF16 compute), we can be more aggressive
        in utilizing GPU memory without risking OOM errors.
        
        BF16 compute mode saves ~50% memory on compute operations (Ampere+ GPUs),
        allowing us to use more aggressive streaming thresholds.
        
        Args:
            vram_gb: Available VRAM in GB. If None, detects automatically.
            
        Returns:
            Recommended tier name based on VRAM:
            - 'minimal' for < 6GB VRAM (conservative, memory-safe)
            - 'balanced' for 6-12GB VRAM (good performance/safety tradeoff)
            - 'aggressive' for 12-20GB VRAM (high performance)
            - 'extreme' for > 20GB VRAM (maximum performance)
        """
        if vram_gb is None:
            if not torch.cuda.is_available():
                return 'balanced'
            try:
                # Use free memory if available, otherwise total memory
                # This is more accurate if other processes are using the GPU
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                
                # Use a more conservative estimate:
                # We take the minimum of (total memory) and (free memory + 2GB buffer for cache)
                # This prevents choosing a high tier when free memory is very low,
                # while still allowing for some cache reclamation.
                effective_free = free_mem + (2 * 1024**3) # Assume we can reclaim 2GB of cache
                vram_gb = min(total_mem, effective_free) / (1024**3)
            except Exception:
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    vram_gb = total_memory / (1024**3)
                except Exception:
                    return 'balanced'
        
        # Check if BF16 is available (Ampere+ GPUs)
        bf16_available = _get_bf16_if_supported() == torch.bfloat16
        
        # BF16 saves ~50% memory on compute operations, so we can be more aggressive.
        # Adjust thresholds conservatively as if we have +2GB VRAM when BF16 is available.
        # This provides realistic memory savings without overcommitting GPU memory.
        effective_vram = vram_gb + (2.0 if bf16_available else 0.0)
        
        # Aggressive auto-detection with memory optimizations:
        # - Pinned memory pool for efficient CPU<->GPU transfers
        # - Buffer pool for quantization buffer reuse
        # - OOM guard for pre-emptive memory checking
        # - Chunked Hadamard for large tensor processing
        # - FP16 intermediates for reduced memory usage
        # - BF16 compute for 2x memory efficiency on compute ops (Ampere+)
        if effective_vram < 6:
            return 'minimal'      # Conservative for very low VRAM (< 6GB)
        elif effective_vram <= 12:
            return 'balanced'     # Balanced for mid-range (6-12GB)
        elif effective_vram <= 20:
            return 'aggressive'   # Aggressive for high VRAM (12-20GB)
        else:
            return 'extreme'      # Extreme for ultra-high VRAM (> 20GB)
    
    @classmethod
    def get_available_tiers(cls) -> list:
        """Get list of available tier names."""
        return list(cls.TIERS.keys())
    
    @classmethod
    def get_safety_threshold(cls, tier: str) -> int:
        """Get safety threshold for Hadamard transform based on tier.
        
        This threshold is used as a hard limit to force CPU usage for extremely
        large tensors, even if the adaptive manager is optimistic.
        
        Args:
            tier: Tier name
            
        Returns:
            Threshold in elements
        """
        # Base safety thresholds (FP32 elements)
        # Higher tiers allow larger tensors on GPU
        safety_map = {
            'off': 10**15,            # Use a very large integer instead of float('inf')
            'minimal': 50_000_000,    # ~200MB
            'balanced': 75_000_000,   # ~300MB
            'aggressive': 150_000_000, # ~600MB
            'extreme': 300_000_000,    # ~1.2GB
        }
        
        base_threshold = safety_map.get(tier, 75_000_000)
        
        # Adjust for BF16 if available
        bf16_available = _get_bf16_if_supported() == torch.bfloat16
        
        # Handle infinity or very large values safely
        # Use math.isinf for robustness
        import math as py_math
        
        # Ensure base_threshold is a number
        try:
            base_threshold = float(base_threshold)
        except (ValueError, TypeError):
            base_threshold = 75_000_000.0
            
        if py_math.isinf(base_threshold) or py_math.isnan(base_threshold) or base_threshold > 10**14:
            return 10**15
            
        if bf16_available:
            # CRITICAL: Ensure we don't multiply infinity by 2.0 before checking
            try:
                val = base_threshold * 2.0
                if py_math.isinf(val):
                    return 10**15
                return int(val)
            except (OverflowError, ValueError):
                return 10**15
        return int(base_threshold)


class AdaptiveStreamingManager:
    """Adaptive streaming manager that monitors memory pressure and adjusts thresholds dynamically.
    
    This class provides real-time adaptive streaming that:
    1. Monitors GPU memory pressure before each operation
    2. Dynamically adjusts thresholds based on available memory
    3. Learns from OOM events to establish safe operating boundaries
    4. Tracks performance to optimize GPU vs CPU decisions
    
    Usage:
        manager = AdaptiveStreamingManager()
        
        # Before each operation, get the current adaptive threshold
        threshold = manager.get_hadamard_threshold()
        
        # If OOM occurs, report it to learn from the failure
        manager.report_oom(operation='hadamard', attempted_elements=50000000)
        
        # Get statistics at the end
        stats = manager.get_stats()
    """
    
    # Memory pressure thresholds (fraction of total VRAM)
    # Conservative thresholds to leave headroom for driver and other applications:
    # - Leave 10-12% VRAM headroom to prevent driver crashes
    MEMORY_CRITICAL = 0.90  # Above this, use CPU aggressively (10% headroom)
    MEMORY_HIGH = 0.80      # Above this, lower thresholds (20% headroom)
    MEMORY_NORMAL = 0.70    # Above this, use balanced thresholds
    MEMORY_LOW = 0.50       # Below this, can be more aggressive
    
    # Adjustment factors for thresholds
    ADJUSTMENT_UP = 1.15    # Increase threshold by 15%
    ADJUSTMENT_DOWN = 0.85  # Decrease threshold by 15%
    ADJUSTMENT_OOM = 0.75   # Decrease threshold by 25% after OOM
    
    def __init__(
        self,
        base_tier: str = "balanced",
        enable_adaptation: bool = True,
        memory_check_interval: float = 0.5,  # Minimum seconds between memory checks
        oom_recovery_enabled: bool = True,
    ):
        """Initialize adaptive streaming manager.
        
        Args:
            base_tier: Base tier to start from ('minimal', 'balanced', 'aggressive')
            enable_adaptation: Whether to enable dynamic threshold adjustment
            memory_check_interval: Minimum time between memory pressure checks
            oom_recovery_enabled: Whether to learn from and recover from OOM events
        """
        self.base_tier = base_tier
        self.enable_adaptation = enable_adaptation
        self.memory_check_interval = memory_check_interval
        self.oom_recovery_enabled = oom_recovery_enabled
        
        # Get base thresholds
        self._base_thresholds = StreamingConfig.get_thresholds(base_tier)
        
        # Current adaptive thresholds (start with base)
        self._current_thresholds = StreamingThresholds(
            hadamard_elements=float(self._base_thresholds.hadamard_elements),
            hessian_dim=float(self._base_thresholds.hessian_dim),
            ldlq_dim=float(self._base_thresholds.ldlq_dim),
            ortho_dim=float(self._base_thresholds.ortho_dim),
        )
        
        # Minimum thresholds (don't go below these)
        self._min_thresholds = StreamingThresholds(
            hadamard_elements=1_000_000,   # ~4MB minimum
            hessian_dim=512,
            ldlq_dim=1024,
            ortho_dim=512,
        )
        
        # Maximum thresholds (don't exceed base tier * 2)
        self._max_thresholds = StreamingThresholds(
            hadamard_elements=float(self._base_thresholds.hadamard_elements) * 2,
            hessian_dim=float(self._base_thresholds.hessian_dim) * 2,
            ldlq_dim=float(self._base_thresholds.ldlq_dim) * 2,
            ortho_dim=float(self._base_thresholds.ortho_dim) * 2,
        )
        
        # Tracking state
        self._last_memory_check: float = 0.0
        self._last_memory_pressure: float = 0.0
        self._last_free_memory: int = 0
        self._last_total_memory: int = 0
        self._oom_history: List[Dict] = []
        self._decision_history: List[Dict] = []
        self._total_operations: int = 0
        self._gpu_operations: int = 0
        self._cpu_operations: int = 0
        
    def _get_memory_info(self, force_refresh: bool = False) -> Tuple[float, int, int]:
        """Get current GPU memory info.
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh value.
        
        Returns:
            Tuple of (pressure, free_bytes, total_bytes)
        """
        if not torch.cuda.is_available():
            return 0.0, 0, 0
        
        current_time = time.time()
        time_since_last = current_time - self._last_memory_check
        
        # Dynamic check interval: check more frequently when pressure is high
        effective_interval = self.memory_check_interval
        if self._last_memory_pressure > self.MEMORY_HIGH:
            effective_interval /= 4.0 # Check 4x more often
        elif self._last_memory_pressure > self.MEMORY_NORMAL:
            effective_interval /= 2.0 # Check 2x more often
            
        if not force_refresh and time_since_last < effective_interval:
            return self._last_memory_pressure, self._last_free_memory, self._last_total_memory
        
        self._last_memory_check = current_time
        
        try:
            # Use mem_get_info for actual free memory (including unreserved)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            reserved_mem = torch.cuda.memory_reserved(0)
            
            # Pressure is still based on reserved vs total for stability
            pressure = reserved_mem / total_mem
            
            self._last_memory_pressure = pressure
            self._last_free_memory = free_mem
            self._last_total_memory = total_mem
            return pressure, free_mem, total_mem
        except Exception:
            return self._last_memory_pressure, getattr(self, '_last_free_memory', 0), getattr(self, '_last_total_memory', 0)

    def _get_memory_pressure(self, force_refresh: bool = False) -> float:
        """Get current GPU memory pressure as a fraction (0.0 - 1.0)."""
        pressure, _, _ = self._get_memory_info(force_refresh)
        return pressure
    
    def _adjust_threshold(self, current: float, target_multiplier: float, min_val: float, max_val: float) -> float:
        """Adjust a threshold towards a target multiplier of base.
        
        Args:
            current: Current threshold value
            target_multiplier: Target as multiplier of base threshold
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Adjusted threshold value
        """
        if not self.enable_adaptation:
            return current
        
        # Gradual adjustment towards target
        adjustment = self.ADJUSTMENT_UP if target_multiplier > 1.0 else self.ADJUSTMENT_DOWN
        new_value = current * adjustment
        
        # Clamp to bounds
        return max(min_val, min(max_val, new_value))
    
    def refresh_base_tier(self):
        """Dynamically refresh the base tier based on current environment."""
        if not self.enable_adaptation:
            return
            
        new_tier = StreamingConfig.auto_detect_tier()
        if new_tier != self.base_tier:
            from ..utils.logging import verbose
            verbose(f"Adaptive Manager: Switching base tier from {self.base_tier} to {new_tier}")
            self.base_tier = new_tier
            self._base_thresholds = StreamingConfig.get_thresholds(new_tier)
            # Update max thresholds too
            self._max_thresholds = StreamingThresholds(
                hadamard_elements=float(self._base_thresholds.hadamard_elements) * 2,
                hessian_dim=float(self._base_thresholds.hessian_dim) * 2,
                ldlq_dim=float(self._base_thresholds.ldlq_dim) * 2,
                ortho_dim=float(self._base_thresholds.ortho_dim) * 2,
            )

    def update_thresholds(self):
        """Update all thresholds based on current memory pressure and available VRAM."""
        if not self.enable_adaptation or not torch.cuda.is_available():
            return
        
        # Periodically refresh base tier to account for external memory changes
        self.refresh_base_tier()
        
        pressure, free_mem, total_mem = self._get_memory_info()
        
        # 1. Pressure-based multiplier (more granular)
        if pressure >= self.MEMORY_CRITICAL:
            target_multiplier = 0.5
        elif pressure >= self.MEMORY_HIGH:
            target_multiplier = 0.75
        elif pressure >= 0.70:
            target_multiplier = 0.9
        elif pressure >= self.MEMORY_NORMAL:
            target_multiplier = 1.0
        elif pressure >= 0.45:
            target_multiplier = 1.3
        elif pressure >= self.MEMORY_LOW:
            target_multiplier = 1.6
        else:
            target_multiplier = 2.2
            
        # 2. Direct memory-based adjustment
        # If we have lots of free memory, we can be even more aggressive
        # regardless of the "pressure" (which is reserved/total)
        free_gb = free_mem / (1024**3)
        if free_gb > 8.0:
            target_multiplier *= 1.2
        elif free_gb < 1.0:
            target_multiplier *= 0.8
        
        # Apply multipliers directly to base thresholds for immediate response
        base = self._base_thresholds
        current = self._current_thresholds
        min_t = self._min_thresholds
        max_t = self._max_thresholds
        
        current.hadamard_elements = max(min_t.hadamard_elements,
                                        min(max_t.hadamard_elements,
                                            base.hadamard_elements * target_multiplier))
        current.hessian_dim = max(min_t.hessian_dim,
                                  min(max_t.hessian_dim,
                                      base.hessian_dim * target_multiplier))
        current.ldlq_dim = max(min_t.ldlq_dim,
                               min(max_t.ldlq_dim,
                                   base.ldlq_dim * target_multiplier))
        current.ortho_dim = max(min_t.ortho_dim,
                                min(max_t.ortho_dim,
                                    base.ortho_dim * target_multiplier))
    
    def _get_bf16_multiplier(self, operation: str) -> float:
        """Get multiplier for thresholds when BF16 is active.
        
        BF16 uses 2 bytes vs FP32's 4 bytes, so we can handle 2x larger
        tensors when BF16 is being used for the operation.
        
        Args:
            operation: Type of operation ('hadamard', 'hessian', 'ldlq', 'ortho')
            
        Returns:
            Multiplier (2.0 if BF16 is active, 1.0 otherwise)
        """
        # BF16 is available on Ampere+ (SM80+)
        if _get_bf16_if_supported() != torch.bfloat16:
            return 1.0
        
        # BF16 uses half the memory of FP32, so we can handle 2x larger tensors
        # Use conservative 2x multiplier (not 3x) to avoid overcommitting memory
        return 2.0
    
    def get_hadamard_threshold(self) -> float:
        """Get current adaptive Hadamard threshold."""
        self.update_thresholds()
        base = self._current_thresholds.hadamard_elements
        # BF16 saves 50% memory, so we can handle 2x larger tensors
        return base * self._get_bf16_multiplier('hadamard')
    
    def get_hessian_threshold(self) -> float:
        """Get current adaptive Hessian threshold."""
        self.update_thresholds()
        base = self._current_thresholds.hessian_dim
        return base * self._get_bf16_multiplier('hessian')
    
    def get_ldlq_threshold(self) -> float:
        """Get current adaptive LDLQ threshold."""
        self.update_thresholds()
        base = self._current_thresholds.ldlq_dim
        return base * self._get_bf16_multiplier('ldlq')
    
    def get_ortho_threshold(self) -> float:
        """Get current adaptive orthogonal matrix threshold."""
        self.update_thresholds()
        base = self._current_thresholds.ortho_dim
        return base * self._get_bf16_multiplier('ortho')
    
    def get_all_thresholds(self) -> StreamingThresholds:
        """Get all current adaptive thresholds."""
        self.update_thresholds()
        return StreamingThresholds(
            hadamard_elements=self._current_thresholds.hadamard_elements,
            hessian_dim=self._current_thresholds.hessian_dim,
            ldlq_dim=self._current_thresholds.ldlq_dim,
            ortho_dim=self._current_thresholds.ortho_dim,
        )
    
    def report_oom(self, operation: str, attempted_size: Optional[Union[int, Tuple]] = None):
        """Report an OOM event to learn from it.
        
        Args:
            operation: Operation that caused OOM ('hadamard', 'hessian', 'ldlq', 'ortho')
            attempted_size: Size of the tensor/operation that failed
        """
        if not self.oom_recovery_enabled:
            return
        
        # Record OOM event
        self._oom_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'attempted_size': attempted_size,
            'thresholds': self.get_all_thresholds(),
            'memory_pressure': self._get_memory_pressure(),
        })
        
        # Reduce threshold for the failing operation
        current = self._current_thresholds
        min_t = self._min_thresholds
        
        if operation == 'hadamard':
            current.hadamard_elements = max(
                min_t.hadamard_elements,
                current.hadamard_elements * self.ADJUSTMENT_OOM
            )
        elif operation == 'hessian':
            current.hessian_dim = max(
                min_t.hessian_dim,
                current.hessian_dim * self.ADJUSTMENT_OOM
            )
        elif operation == 'ldlq':
            current.ldlq_dim = max(
                min_t.ldlq_dim,
                current.ldlq_dim * self.ADJUSTMENT_OOM
            )
        elif operation == 'ortho':
            current.ortho_dim = max(
                min_t.ortho_dim,
                current.ortho_dim * self.ADJUSTMENT_OOM
            )
    
    def log_decision(self, operation: str, device: str, tensor_shape: Tuple, reason: str = ""):
        """Log a streaming decision for tracking.
        
        Args:
            operation: Type of operation
            device: 'cpu' or 'gpu' 
            tensor_shape: Shape of the tensor being processed
            reason: Reason for the decision
        """
        self._total_operations += 1
        if device == 'gpu':
            self._gpu_operations += 1
        else:
            self._cpu_operations += 1
        
        self._decision_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'device': device,
            'tensor_shape': tensor_shape,
            'reason': reason,
            'memory_pressure': self._get_memory_pressure(),
            'thresholds': self.get_all_thresholds(),
        })
        
        # Keep only last 1000 decisions to prevent memory bloat
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-1000:]
    
    def get_stats(self) -> Dict:
        """Get statistics about adaptive streaming decisions.
        
        Returns:
            Dictionary with statistics including:
            - total_operations
            - gpu_operations
            - cpu_operations
            - oom_events
            - current_thresholds
            - current_memory_pressure
        """
        return {
            'total_operations': self._total_operations,
            'gpu_operations': self._gpu_operations,
            'cpu_operations': self._cpu_operations,
            'gpu_percentage': (self._gpu_operations / self._total_operations * 100) if self._total_operations > 0 else 0,
            'oom_events': len(self._oom_history),
            'current_memory_pressure': self._get_memory_pressure(),
            'current_thresholds': {
                'hadamard_elements': self._current_thresholds.hadamard_elements,
                'hessian_dim': self._current_thresholds.hessian_dim,
                'ldlq_dim': self._current_thresholds.ldlq_dim,
                'ortho_dim': self._current_thresholds.ortho_dim,
            },
            'base_tier': self.base_tier,
            'adaptation_enabled': self.enable_adaptation,
        }
    
    def reset_thresholds(self):
        """Reset all thresholds to base tier values."""
        base = self._base_thresholds
        self._current_thresholds = StreamingThresholds(
            hadamard_elements=float(base.hadamard_elements),
            hessian_dim=float(base.hessian_dim),
            ldlq_dim=float(base.ldlq_dim),
            ortho_dim=float(base.ortho_dim),
        )
        self._oom_history.clear()
    
    def _compute_size_ratio(self, size: Union[int, float], threshold: float) -> float:
        """Compute how close the size is to the threshold (0.0 to 1.0+).
        
        Returns:
            Ratio of size to threshold. >1.0 means exceeds threshold.
        """
        if threshold <= 0:
            return float('inf')
        return size / threshold
    
    def _get_adaptive_score(self, operation: str, size: Union[int, float]) -> float:
        """Compute an adaptive score for the operation (0.0 = definitely GPU, 1.0 = definitely CPU).
        
        This uses a sigmoid-like curve to provide smoother transitions between GPU and CPU,
        and considers historical performance data.
        
        Args:
            operation: Type of operation
            size: Size metric
            
        Returns:
            Score between 0.0 and 1.0
        """
        self.update_thresholds()
        
        # Get threshold for this operation
        if operation == 'hadamard':
            threshold = self._current_thresholds.hadamard_elements
        elif operation == 'hessian':
            threshold = self._current_thresholds.hessian_dim
        elif operation == 'ldlq':
            threshold = self._current_thresholds.ldlq_dim
        elif operation == 'ortho':
            threshold = self._current_thresholds.ortho_dim
        else:
            return 0.0  # Unknown operation, default to GPU
        
        # Compute ratio of size to threshold
        ratio = self._compute_size_ratio(size, threshold)
        
        # Apply memory pressure multiplier
        # Higher pressure = more aggressive about using CPU
        # Lower pressure = can be more aggressive with GPU
        # Conservative factors to leave headroom for driver/other apps
        pressure = self._get_memory_pressure()
        if pressure >= self.MEMORY_CRITICAL:
            pressure_factor = 2.0  # Strongly favor CPU
        elif pressure >= self.MEMORY_HIGH:
            pressure_factor = 1.5  # Favor CPU
        elif pressure >= 0.70:
            pressure_factor = 1.2  # Slightly favor CPU
        elif pressure >= self.MEMORY_NORMAL:
            pressure_factor = 1.0  # Balanced
        elif pressure >= 0.45:
            pressure_factor = 0.8  # Slightly favor GPU
        elif pressure >= self.MEMORY_LOW:
            pressure_factor = 0.7  # Favor GPU
        else:
            pressure_factor = 0.6  # Moderately favor GPU (capped to avoid over-allocation)
        
        # Adjusted ratio considering memory pressure
        adjusted_ratio = ratio * pressure_factor
        
        # Apply sigmoid-like curve for smooth transition
        # At ratio = 1.0, score = 0.5 (undecided)
        # At ratio = 0.5, score ≈ 0.27 (favor GPU)
        # At ratio = 2.0, score ≈ 0.88 (strongly favor CPU)
        import math
        score = 1.0 / (1.0 + math.exp(-4.0 * (adjusted_ratio - 1.0)))
        
        # Consider historical performance if available
        history_score = self._get_historical_bias(operation, size)
        if history_score is not None:
            # Blend current score with historical bias (70% current, 30% history)
            score = 0.7 * score + 0.3 * history_score
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _get_historical_bias(self, operation: str, size: Union[int, float]) -> Optional[float]:
        """Get historical bias from past decisions for similar-sized operations.
        
        Returns:
            Score between 0.0 and 1.0, or None if no relevant history.
        """
        if not self._decision_history:
            return None
        
        # Look at recent decisions for same operation with similar size (±20%)
        relevant_decisions = []
        for d in self._decision_history[-100:]:  # Look at last 100 decisions
            if d['operation'] != operation:
                continue
            
            # Get size from tensor_shape
            if isinstance(d['tensor_shape'], tuple):
                if operation == 'hadamard':
                    hist_size = d['tensor_shape'][0] * d['tensor_shape'][1] if len(d['tensor_shape']) >= 2 else d['tensor_shape'][0]
                else:
                    hist_size = max(d['tensor_shape']) if d['tensor_shape'] else 0
            else:
                hist_size = d['tensor_shape']
            
            # Check if within ±20% of current size
            if hist_size > 0 and 0.8 <= size / hist_size <= 1.25:
                relevant_decisions.append(d)
        
        if len(relevant_decisions) < 3:  # Need at least 3 data points
            return None
        
        # Calculate success rate for GPU
        gpu_decisions = [d for d in relevant_decisions if d['device'] == 'gpu']
        if not gpu_decisions:
            return 1.0  # All CPU, strongly favor CPU
        
        # Score based on proportion of GPU decisions
        gpu_ratio = len(gpu_decisions) / len(relevant_decisions)
        return 1.0 - gpu_ratio  # Higher score = more CPU bias
    
    def should_use_cpu(self, operation: str, size: Union[int, Tuple[int, ...]]) -> bool:
        """Determine if CPU should be used for an operation using adaptive scoring.
        
        Args:
            operation: Type of operation ('hadamard', 'hessian', 'ldlq', 'ortho')
            size: Size metric for the operation (elements for hadamard, dim for others)
            
        Returns:
            True if CPU should be used, False for GPU
        """
        # Handle tuple sizes (extract meaningful metric)
        if isinstance(size, tuple):
            if operation == 'hadamard':
                size = size[0] * size[1] if len(size) >= 2 else size[0]
            else:
                size = max(size)
        
        # Get adaptive score
        score = self._get_adaptive_score(operation, size)
        
        # Decision threshold can also be adaptive
        # If we've had recent OOMs, be more conservative (lower threshold)
        if self._oom_history:
            recent_ooms = sum(1 for oom in self._oom_history[-5:] if oom['operation'] == operation)
            if recent_ooms > 0:
                # Lower threshold if we've had recent OOMs for this operation
                decision_threshold = 0.3  # More likely to choose CPU
            else:
                decision_threshold = 0.5
        else:
            decision_threshold = 0.5
        
        use_cpu = score >= decision_threshold
        
        # Force CPU if memory pressure is critical regardless of score
        # Use force_refresh=True for this critical safety check
        pressure = self._get_memory_pressure(force_refresh=True)
        if pressure >= self.MEMORY_CRITICAL:
            use_cpu = True
            
        # Log a warning for large CPU operations as they will be slow
        if use_cpu and operation in ('ldlq', 'hessian') and size > 4096:
            from ..utils.logging import verbose
            verbose(f"WARNING: High memory pressure ({pressure:.1%}) forcing {operation} to CPU. "
                    f"This will be significantly slower than GPU.")
        
        return use_cpu
    
    def get_adaptive_decision_info(self, operation: str, size: Union[int, Tuple[int, ...]]) -> Dict:
        """Get detailed information about an adaptive decision.
        
        Returns:
            Dictionary with score, threshold, ratio, and recommendation.
        """
        if isinstance(size, tuple):
            if operation == 'hadamard':
                size = size[0] * size[1] if len(size) >= 2 else size[0]
            else:
                size = max(size)
        
        score = self._get_adaptive_score(operation, size)
        
        # Get current threshold
        if operation == 'hadamard':
            threshold = self._current_thresholds.hadamard_elements
        elif operation == 'hessian':
            threshold = self._current_thresholds.hessian_dim
        elif operation == 'ldlq':
            threshold = self._current_thresholds.ldlq_dim
        elif operation == 'ortho':
            threshold = self._current_thresholds.ortho_dim
        else:
            threshold = 0
        
        ratio = self._compute_size_ratio(size, threshold) if threshold > 0 else 0
        
        return {
            'operation': operation,
            'size': size,
            'threshold': threshold,
            'ratio': ratio,
            'score': score,
            'recommend_cpu': score >= 0.5,
            'memory_pressure': self._get_memory_pressure(),
        }
