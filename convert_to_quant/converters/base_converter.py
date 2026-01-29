"""
Base converter class for learned rounding quantization.

Provides shared infrastructure for LearnedRoundingConverter (FP8/INT8)
and LearnedNVFP4Converter (NVFP4). Contains common initialization,
SVD computation, LR scheduling, and early stopping logic.
"""
import gc
import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE
from ..utils.tensor_utils import adaptive_lr_update
from ..pinned_transfer import transfer_to_gpu_pinned


class CudaGraphRunner:
    """
    CUDA Graph capture and replay utility for optimization loops.
    
    Captures GPU operations into a graph for efficient replay,
    eliminating CPU launch overhead in iterative optimization.
    """
    
    def __init__(self, warmup_iters: int = 3):
        self.graph = None
        self.static_inputs = []
        self.static_outputs = []
        self.warmup_iters = warmup_iters
        self.is_captured = False
        
    def capture_graph(self, model_fn, sample_inputs):
        """
        Capture a CUDA graph for the given model function.
        
        Args:
            model_fn: Callable that performs GPU operations
            sample_inputs: List of input tensors for tracing
            
        Returns:
            Tuple of (graph, static_inputs, static_outputs)
        """
        if not torch.cuda.is_available():
            return None, sample_inputs, None
            
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                model_fn(*sample_inputs)
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture
        self.static_inputs = sample_inputs
        self.graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.graph):
            self.static_outputs = model_fn(*sample_inputs)
            if not isinstance(self.static_outputs, (tuple, list)):
                self.static_outputs = (self.static_outputs,)
                
        self.is_captured = True
        return self.graph, self.static_inputs, self.static_outputs
    
    def replay(self, inputs=None):
        """
        Replay the captured graph.
        
        Args:
            inputs: Optional new inputs (must match captured shapes)
            
        Returns:
            Static outputs from graph replay
        """
        if not self.is_captured:
            raise RuntimeError("Graph not captured. Call capture_graph first.")
            
        if inputs is not None:
            # Copy new inputs to static input buffers
            for static_inp, new_inp in zip(self.static_inputs, inputs):
                static_inp.copy_(new_inp)
                
        self.graph.replay()
        return self.static_outputs


def maybe_use_cuda_graphs(enable: bool = True):
    """
    Decorator factory to optionally wrap optimization with CUDA graphs.
    
    Usage:
        @maybe_use_cuda_graphs(enable=True)
        def optimize_loop(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not enable or not torch.cuda.is_available():
                return func(*args, **kwargs)
            # CUDA graph logic would be integrated here
            # For now, just pass through
            return func(*args, **kwargs)
        return wrapper
    return decorator


class BaseLearnedConverter(ABC):
    """
    Abstract base class for learned rounding quantization converters.

    Provides shared infrastructure:
    - SVD computation for principal component optimization
    - Tier-based adaptive LR scheduling
    - Early stopping with configurable thresholds

    Subclasses must implement:
    - convert(): Format-specific quantization logic
    """

    def __init__(
        self,
        optimizer: str = "original",
        num_iter: int = 500,
        top_p: float = 0.01,
        min_k: int = 1,
        max_k: int = 16,
        full_matrix: bool = False,
        no_learned_rounding: bool = False,
        lr_schedule: str = "adaptive",
        lr_gamma: float = 0.99,
        lr_patience: int = 50,
        lr_factor: float = 0.5,
        lr_min: float = 1e-8,
        lr_cooldown: int = 0,
        lr_threshold: float = 0.0,
        lr_adaptive_mode: str = "simple-reset",
        lr_shape_influence: float = 1.0,
        lr_threshold_mode: str = "rel",
        early_stop_loss: float = 1e-8,
        early_stop_lr: float = 1e-10,
        early_stop_stall: int = 1000,
        **kwargs,
    ):
        """
        Initialize base converter with shared optimization parameters.

        Args:
            optimizer: Optimization algorithm ("original", "adamw", "radam")
            num_iter: Number of optimization iterations
            top_p: Proportion of principal components to use
            min_k: Minimum number of SVD components
            max_k: Maximum number of SVD components
            full_matrix: Use full SVD instead of lowrank
            no_learned_rounding: Skip optimization, use simple quantization
            lr_schedule: LR schedule ("adaptive", "exponential", "plateau")
            lr_gamma: Decay factor for exponential schedule
            lr_patience: Steps before decay for plateau schedule
            lr_factor: LR reduction factor for plateau
            lr_min: Minimum learning rate
            lr_cooldown: Cooldown steps after LR reduction
            lr_threshold: Minimum improvement threshold
            lr_adaptive_mode: Counter reset mode ("simple-reset", "no-reset")
            lr_shape_influence: Shape-aware LR scaling (0.0-1.0)
            lr_threshold_mode: Threshold mode ("rel", "abs")
            early_stop_loss: Stop when loss drops below this
            early_stop_lr: Stop when LR drops below this
            early_stop_stall: Stop after this many steps without improvement
            **kwargs: Additional optimizer-specific parameters (e.g., lr)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SVD parameters
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.full_matrix = full_matrix

        # Optimizer configuration
        self.optimizer_choice = optimizer
        self.num_iter = num_iter
        self.no_learned_rounding = no_learned_rounding
        self.optimizer_kwargs = kwargs

        # LR schedule configuration
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_cooldown = lr_cooldown
        self.lr_threshold = lr_threshold
        self.lr_adaptive_mode = lr_adaptive_mode
        self.lr_shape_influence = lr_shape_influence
        self.lr_threshold_mode = lr_threshold_mode

        # Early stopping thresholds
        self.early_stop_loss = early_stop_loss
        self.early_stop_lr = early_stop_lr
        self.early_stop_stall = early_stop_stall

    def _compute_svd_components(
        self, W_float32: torch.Tensor, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute SVD components for optimization.

        Args:
            W_float32: Input tensor (2D)
            verbose: Print SVD computation info

        Returns:
            Tuple of (U_k, Vh_k, k) where k is number of components used
        """
        M, N = W_float32.shape
        max_rank = min(M, N)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        if verbose:
            print(f"    - Tensor shape: [{M}, {N}], Max rank: {max_rank}. Using k={k} components.")

        if self.full_matrix:
            if verbose:
                print("    - Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                if verbose:
                    print("    - Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                if verbose:
                    print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        return U[:, :k], Vh[:k, :], k

    def _adaptive_lr_update(
        self,
        curr_lr: float,
        improved: bool,
        counter_for_tier: int,
        worse_loss_counter: int,
        small_mult: float = 1.0,
    ) -> float:
        """
        Tier-based adaptive LR update schedule.

        Delegates to centralized adaptive_lr_update() from tensor_utils.

        Args:
            curr_lr: Current learning rate
            improved: Whether loss improved this iteration
            counter_for_tier: Counter for tier selection
            worse_loss_counter: Current worse loss counter
            small_mult: Optional multiplier for square matrices

        Returns:
            Updated learning rate
        """
        return adaptive_lr_update(
            curr_lr, improved, counter_for_tier, worse_loss_counter, small_mult
        )

    def _compute_shape_aware_plateau_params(
        self, M: int, N: int
    ) -> Tuple[int, float, int]:
        """
        Compute shape-aware parameters for plateau LR schedule.

        Args:
            M: First dimension
            N: Second dimension

        Returns:
            Tuple of (effective_patience, effective_factor, effective_cooldown)
        """
        if self.lr_schedule == "plateau" and self.lr_shape_influence > 0:
            aspect_ratio = max(M, N) / min(M, N)
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence

            effective_patience = self.lr_patience
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor ** ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        return effective_patience, effective_factor, effective_cooldown

    def _check_improvement(
        self, current_loss: float, best_loss: float
    ) -> bool:
        """
        Check if current loss is a significant improvement.

        Supports both relative and absolute threshold modes.

        Args:
            current_loss: Current iteration loss
            best_loss: Best loss seen so far

        Returns:
            True if improvement is significant
        """
        if self.lr_threshold > 0:
            if self.lr_threshold_mode == "rel":
                return current_loss < best_loss * (1.0 - self.lr_threshold)
            else:  # 'abs'
                return (best_loss - current_loss) > self.lr_threshold
        return current_loss < best_loss

    def _cleanup_tensors(self, *tensors) -> None:
        """Delete tensors and clear GPU cache."""
        for t in tensors:
            del t
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @abstractmethod
    def convert(self, W_orig: torch.Tensor) -> Tuple:
        """
        Convert tensor to quantized format.

        Args:
            W_orig: Input tensor (2D)

        Returns:
            Tuple of quantized tensors (format-specific)
        """
        pass
