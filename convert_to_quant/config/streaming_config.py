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
            hadamard_elements=float('inf'),
            hessian_dim=float('inf'),
            ldlq_dim=float('inf'),
            ortho_dim=float('inf')
        ),
        'minimal': StreamingThresholds(
            hadamard_elements=100_000_000,  # ~400MB
            hessian_dim=16384,
            ldlq_dim=32768,
            ortho_dim=16384
        ),
        'balanced': StreamingThresholds(
            hadamard_elements=50_000_000,   # ~200MB (was 33M)
            hessian_dim=6144,               # Was 4096
            ldlq_dim=12288,                 # Was 8192
            ortho_dim=6144                  # Was 4096
        ),
        'aggressive': StreamingThresholds(
            hadamard_elements=25_000_000,   # ~100MB - more aggressive offloading
            hessian_dim=4096,               # Lower threshold for Hessian
            ldlq_dim=8192,                  # Lower threshold for LDLQ
            ortho_dim=4096                  # Lower threshold for orthogonal
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
            - 'aggressive' for > 12GB VRAM (maximum performance)
        """
        if vram_gb is None:
            if not torch.cuda.is_available():
                return 'balanced'
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_memory / (1024**3)  # Convert to GB correctly (not 1e9)
            except Exception:
                return 'balanced'
        
        # Check if BF16 is available (Ampere+ GPUs)
        bf16_available = _get_bf16_if_supported() == torch.bfloat16
        
        # BF16 saves ~50% memory on compute operations, so we can be more aggressive.
        # Adjust thresholds as if we have +5GB VRAM when BF16 is available.
        # This shifts 8GB GPUs to 'aggressive' tier for maximum performance.
        effective_vram = vram_gb + (5.0 if bf16_available else 0.0)
        
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
        else:
            return 'aggressive'   # Aggressive for high VRAM (> 12GB)
    
    @classmethod
    def get_available_tiers(cls) -> list:
        """Get list of available tier names."""
        return list(cls.TIERS.keys())


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
    # More aggressive thresholds with memory optimizations:
    # - Pinned pool, buffer pool, OOM guard, chunked Hadamard, FP16
    MEMORY_CRITICAL = 0.95  # Above this, use CPU aggressively (was 0.90)
    MEMORY_HIGH = 0.85      # Above this, lower thresholds (was 0.75)
    MEMORY_NORMAL = 0.60    # Above this, use balanced thresholds (was 0.50)
    MEMORY_LOW = 0.40       # Below this, can be more aggressive (was 0.30)
    
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
        self._oom_history: List[Dict] = []
        self._decision_history: List[Dict] = []
        self._total_operations: int = 0
        self._gpu_operations: int = 0
        self._cpu_operations: int = 0
        
    def _get_memory_pressure(self, force_refresh: bool = False) -> float:
        """Get current GPU memory pressure as a fraction (0.0 - 1.0).
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh value.
                          Use for critical decisions where stale data is unacceptable.
        
        Returns:
            Memory pressure: allocated_memory / total_memory
            Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Rate limit checks (unless force_refresh is True)
        current_time = time.time()
        time_since_last = current_time - self._last_memory_check
        
        if not force_refresh and time_since_last < self.memory_check_interval:
            return self._last_memory_pressure
        
        self._last_memory_check = current_time
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            pressure = allocated_memory / total_memory
            self._last_memory_pressure = pressure
            return pressure
        except Exception:
            return self._last_memory_pressure
    
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
    
    def update_thresholds(self):
        """Update all thresholds based on current memory pressure."""
        if not self.enable_adaptation or not torch.cuda.is_available():
            return
        
        pressure = self._get_memory_pressure()
        
        # Determine target multiplier based on memory pressure
        # Lower pressure = higher thresholds = more GPU usage
        # Set directly rather than gradually adjusting for more responsive behavior
        # With memory optimizations, we can be more aggressive at all levels
        if pressure >= self.MEMORY_CRITICAL:
            # Critical: Use conservative thresholds (50% of base, was 40%)
            target_multiplier = 0.5
        elif pressure >= self.MEMORY_HIGH:
            # High: Use slightly conservative thresholds (75% of base, was 60%)
            target_multiplier = 0.75
        elif pressure >= 0.70:
            # Moderately high: Near base (90% of base, was 80%)
            target_multiplier = 0.9
        elif pressure >= self.MEMORY_NORMAL:
            # Normal: Use base thresholds (100% of base)
            target_multiplier = 1.0
        elif pressure >= 0.45:
            # Low (45-60%): Be aggressive (175% of base, was 150%)
            target_multiplier = 1.75
        elif pressure >= self.MEMORY_LOW:
            # Very low (40-45%): Be very aggressive (250% of base, was 200%)
            target_multiplier = 2.5
        else:
            # Extremely low (<40%): Be extremely aggressive (400% of base, was 300%)
            target_multiplier = 4.0
        
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
        
        # All these operations benefit from BF16 memory savings.
        # Use 3x multiplier for more aggressive GPU utilization.
        return 3.0
    
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
        # Lower pressure = can be much more aggressive with GPU
        # With memory optimizations, favor GPU more aggressively
        pressure = self._get_memory_pressure()
        if pressure >= self.MEMORY_CRITICAL:
            pressure_factor = 2.5  # Strongly favor CPU (was 3.0)
        elif pressure >= self.MEMORY_HIGH:
            pressure_factor = 1.75  # Favor CPU (was 2.0)
        elif pressure >= 0.70:
            pressure_factor = 1.25  # Slightly favor CPU (was 1.5)
        elif pressure >= self.MEMORY_NORMAL:
            pressure_factor = 0.9  # Slightly favor GPU (was 1.0)
        elif pressure >= 0.45:
            pressure_factor = 0.5  # Favor GPU - 50% more aggressive (was 0.6)
        elif pressure >= self.MEMORY_LOW:
            pressure_factor = 0.3  # Strongly favor GPU - 70% more aggressive (was 0.4)
        else:
            pressure_factor = 0.2  # Very strongly favor GPU - 80% more aggressive
        
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
        if self._get_memory_pressure(force_refresh=True) >= self.MEMORY_CRITICAL:
            use_cpu = True
        
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
