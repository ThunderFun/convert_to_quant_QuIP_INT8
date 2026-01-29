import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE, TARGET_INT8_DTYPE, INT8_SYMMETRIC_MAX
from ..utils.logging import verbose, debug
from ..utils.hadamard import (
    is_power_of_two,
    random_orthogonal_matrix,
)
from ..comfy.hadamard_kernels import fast_hadamard_transform


class QuIPInt8Converter:
    """
    QuIP (Quantization with Incoherence Processing) for INT8.
    
    Key advantages over GPTQ:
    - Incoherence processing makes weights more quantization-friendly
    - Near-lossless INT8 quantization
    - Same computational cost as GPTQ (transform overhead is negligible)
    
    Reference: https://arxiv.org/abs/2307.13304
    """
    
    def __init__(
        self,
        block_size: int = 128,
        percdamp: float = 0.01,
        device: str = "cuda",
        actorder: bool = True,
        use_hadamard: bool = True,
        seed: Optional[int] = None,
        use_triton: bool = False,
        lazy_updates: bool = True,
        use_learned_rounding: bool = False,
        ldlq_iterations: int = 1,
        store_transformed: bool = False,
        streaming_mode: str = "balanced",
        streaming_thresholds: Optional[Dict[str, Optional[int]]] = None,
        no_memory_limits: bool = False,
        use_checkpointed_ldlq: bool = False,
        checkpointed_ldlq_threshold: int = 8192,
        checkpoint_segments: int = 4,
    ):
        self.block_size = block_size
        self.percdamp = percdamp
        self.device = device if torch.cuda.is_available() else "cpu"
        self.actorder = actorder
        self.use_checkpointed_ldlq = use_checkpointed_ldlq
        self.checkpointed_ldlq_threshold = checkpointed_ldlq_threshold
        self.checkpoint_segments = checkpoint_segments
        self.use_hadamard = use_hadamard
        self.seed = seed
        self.use_triton = use_triton
        self.lazy_updates = lazy_updates
        self.use_learned_rounding = use_learned_rounding
        self.ldlq_iterations = ldlq_iterations
        self.store_transformed = store_transformed
        # Keep original streaming_mode value intact for downstream checks
        self.streaming_mode = streaming_mode
        # Store a display name for logging purposes (doesn't affect logic)
        self._streaming_mode_display = streaming_mode
        
        # Get thresholds from streaming config
        from ..config.streaming_config import StreamingConfig, AdaptiveStreamingManager
        
        # Initialize adaptive streaming manager (None for non-adaptive modes)
        self._adaptive_manager: Optional[AdaptiveStreamingManager] = None
        
        # Handle "auto" mode - now uses adaptive streaming
        if streaming_mode == "auto":
            # Detect base tier from VRAM
            base_tier = StreamingConfig.auto_detect_tier()
            # Apply threshold overrides to base tier before creating adaptive manager
            # This allows custom base thresholds while maintaining adaptive behavior
            base_thresholds = StreamingConfig.get_thresholds(base_tier)
            
            # Apply custom overrides to base thresholds (if provided)
            if streaming_thresholds:
                if streaming_thresholds.get("hadamard"):
                    base_thresholds.hadamard_elements = streaming_thresholds["hadamard"]
                if streaming_thresholds.get("hessian"):
                    base_thresholds.hessian_dim = streaming_thresholds["hessian"]
                if streaming_thresholds.get("ldlq"):
                    base_thresholds.ldlq_dim = streaming_thresholds["ldlq"]
                if streaming_thresholds.get("ortho"):
                    base_thresholds.ortho_dim = streaming_thresholds["ortho"]
            
            # Create adaptive manager with customized base tier
            self._adaptive_manager = AdaptiveStreamingManager(
                base_tier=base_tier,
                enable_adaptation=not no_memory_limits,
                oom_recovery_enabled=not no_memory_limits,
            )
            # Override the base thresholds in the adaptive manager with our customized values
            if streaming_thresholds:
                self._adaptive_manager._base_thresholds = base_thresholds
                self._adaptive_manager.reset_thresholds()  # Apply the new base values
            
            # Set display name for logging (original mode is preserved)
            limits_status = "no-limits" if no_memory_limits else "adaptive"
            self._streaming_mode_display = f"auto ({limits_status}, base={base_tier})"
            # Get initial thresholds from adaptive manager
            thresholds = self._adaptive_manager.get_all_thresholds()
        else:
            # Get base thresholds from tier
            thresholds = StreamingConfig.get_thresholds(streaming_mode)
            
            # Apply custom overrides if provided (for non-auto modes)
            if streaming_thresholds:
                if streaming_thresholds.get("hadamard"):
                    thresholds.hadamard_elements = streaming_thresholds["hadamard"]
                if streaming_thresholds.get("hessian"):
                    thresholds.hessian_dim = streaming_thresholds["hessian"]
                if streaming_thresholds.get("ldlq"):
                    thresholds.ldlq_dim = streaming_thresholds["ldlq"]
                if streaming_thresholds.get("ortho"):
                    thresholds.ortho_dim = streaming_thresholds["ortho"]
        
        self.hadamard_threshold = thresholds.hadamard_elements
        self.ldlq_threshold = thresholds.ldlq_dim
        self.hessian_threshold = thresholds.hessian_dim
        self.ortho_threshold = thresholds.ortho_dim
        
        # When no_memory_limits is enabled, override all thresholds to infinity to disable CPU offloading
        self._no_memory_limits = no_memory_limits
        if no_memory_limits:
            self.hadamard_threshold = float('inf')
            self.ldlq_threshold = float('inf')
            self.hessian_threshold = float('inf')
            self.ortho_threshold = float('inf')
        
        # streaming_enabled is True for all modes except "off"
        self.streaming = streaming_mode != "off"
        
        # Cache for orthogonal matrices
        self._ortho_cache = {}
        self.low_memory = False # Will be set by quantization.py

        # Pre-allocated buffers for memory optimization
        self._W_buffer = None
        self._H_buffer = None
        self._err_buffer = None
        self._q_buffer = None

    def cleanup(self):
        """Clear caches and free memory. Call after processing each tensor in low_memory mode."""
        if hasattr(self, '_ortho_cache'):
            self._ortho_cache.clear()
        self._W_buffer = None
        self._H_buffer = None
        self._err_buffer = None
        self._q_buffer = None
        
        # Clear global buffer pool to free cached buffers between layers
        from ..utils.buffer_pool import clear_buffer_pool
        cleared = clear_buffer_pool()
        if cleared > 0:
            verbose(f"[QuIP] Cleaned up {cleared} buffer sets from pool")
        
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _get_hadamard_threshold(self) -> float:
        """Get current Hadamard threshold (adaptive or static)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_hadamard_threshold()
        return self.hadamard_threshold

    def _get_hessian_threshold(self) -> float:
        """Get current Hessian threshold (adaptive or static)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_hessian_threshold()
        return self.hessian_threshold

    def _get_ldlq_threshold(self) -> float:
        """Get current LDLQ threshold (adaptive or static)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_ldlq_threshold()
        return self.ldlq_threshold

    def _get_ortho_threshold(self) -> float:
        """Get current orthogonal matrix threshold (adaptive or static)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_ortho_threshold()
        return self.ortho_threshold

    def _should_use_cpu(self, operation: str, size) -> bool:
        """Determine if CPU should be used (adaptive or static decision)."""
        # When no_memory_limits is enabled, always use GPU, never CPU
        if self._no_memory_limits:
            return False
        
        if self._adaptive_manager is not None:
            return self._adaptive_manager.should_use_cpu(operation, size)
        
        # Static fallback
        if operation == 'hadamard':
            return size > self.hadamard_threshold
        elif operation == 'hessian':
            return size > self.hessian_threshold
        elif operation == 'ldlq':
            return size > self.ldlq_threshold
        elif operation == 'ortho':
            return size > self.ortho_threshold
        return False

    def _log_streaming_decision(self, operation: str, device: str, tensor_shape: tuple, reason: str = ""):
        """Log a streaming decision if using adaptive mode."""
        if self._adaptive_manager is not None:
            self._adaptive_manager.log_decision(operation, device, tensor_shape, reason)

    def _report_oom(self, operation: str, attempted_size = None):
        """Report an OOM event to adaptive manager for learning."""
        if self._adaptive_manager is not None:
            self._adaptive_manager.report_oom(operation, attempted_size)

    def get_adaptive_stats(self) -> Optional[Dict]:
        """Get adaptive streaming statistics if in auto mode."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_stats()
        return None

    def _ensure_buffers(self, M: int, N: int, block_size: int, ldlq_device: str = None) -> None:
        """Allocate or reuse buffers for quantization loop using Buffer Pool."""
        buf_device = ldlq_device if ldlq_device else self.device
        
        # Use Buffer Pool for efficient buffer reuse across layers
        from ..utils.buffer_pool import get_buffer_pool
        pool = get_buffer_pool()
        
        # Get buffers from pool (sliced to exact size needed)
        self._q_buffer, self._err_buffer, self._W_buffer = pool.get_buffers(
            M, N, block_size, buf_device
        )

    def _get_random_signs(self, n: int) -> Tensor:
        """Get random signs (+1/-1) for QuIP. Does NOT reset seed - that's done once in __init__."""
        return torch.randint(0, 2, (n,), device=self.device).float() * 2 - 1

    def _apply_orthogonal(self, x: Tensor, dim: int) -> Tensor:
        """
        Apply orthogonal transformation to x along dim.
        Uses Fast Hadamard Transform if possible, otherwise full matrix.
        """
        n = x.shape[dim]
        if self.use_hadamard and is_power_of_two(n):
            # Apply random signs then Hadamard
            signs = self._get_random_signs(n)
            
            # Reshape signs for broadcasting
            shape = [1] * x.ndim
            shape[dim] = n
            x = x * signs.view(*shape)
            
            if dim == -1 or dim == x.ndim - 1:
                return fast_hadamard_transform(x)
            else:
                # Move dim to last, transform, move back
                x = x.transpose(dim, -1)
                x = fast_hadamard_transform(x)
                return x.transpose(dim, -1)
        else:
            # Fallback to full matrix
            Q = self._get_orthogonal_matrix(n)
            if dim == 0:
                return Q @ x
            else:
                return x @ Q.t()

    def _get_orthogonal_matrix(self, n: int) -> Tensor:
        """Get or compute orthogonal matrix of size n."""
        if n in self._ortho_cache:
            return self._ortho_cache[n]
        
        # For QuIP, we use Hadamard if power of 2, otherwise random orthogonal
        # Use dynamic threshold from adaptive manager if available
        ortho_threshold = self._get_ortho_threshold()
        use_cpu = self._should_use_cpu('ortho', n) or (n > ortho_threshold and self.low_memory)
        gen_device = "cpu" if use_cpu else self.device
        
        if self.use_hadamard and is_power_of_two(n):
            from ..utils.hadamard import hadamard_matrix
            Q = hadamard_matrix(n).to(gen_device)
        else:
            Q = random_orthogonal_matrix(n, seed=self.seed, device=gen_device)
        
        if not self.low_memory:
            self._ortho_cache[n] = Q
        return Q

    def _apply_incoherence(self, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Transform W and H to make them more "incoherent" (uniform).
        W' = U @ W @ V^T
        H' = V @ H @ V^T
        
        For non-power-of-2 dimensions with Hadamard transform:
        - Returns FULL padded tensor (M_pad x N_pad) to preserve transform information
        - Original dimensions should be tracked by caller for final slicing
        - Sign vectors are returned at full padded length
        """
        M, N = W.shape
        
        if self.use_hadamard:
            # Use Fast Hadamard Transform for speed and low VRAM
            from ..utils.hadamard import next_power_of_two
            M_pad = next_power_of_two(M)
            N_pad = next_power_of_two(N)
            
            # Use CPU for large matrices if in low_memory or streaming mode
            # Check BOTH weight size (M_pad * N_pad) AND Hessian size (N_pad * N_pad)
            # The Hessian can be much larger for wide matrices
            weight_elements = M_pad * N_pad
            hessian_elements = N_pad * N_pad  # Hessian is N×N
            total_elements = max(weight_elements, hessian_elements)  # Use larger of the two
            
            # Use dynamic threshold from adaptive manager if available
            # Otherwise fall back to static thresholds based on streaming mode
            
            # Safety threshold: Force CPU for very large tensors to prevent OOM
            # Even in auto mode, tensors > ~380MB (FP32) should go to CPU
            # With BF16, we can handle 2x larger tensors
            from ..constants import _get_bf16_if_supported
            bf16_available = _get_bf16_if_supported() == torch.bfloat16
            bf16_multiplier = 2.0 if bf16_available else 1.0
            
            # BF16 uses 2 bytes vs FP32's 4 bytes = 50% memory savings
            # So we can handle 2x larger tensors on GPU
            if bf16_available:
                SAFETY_THRESHOLD_ELEMENTS = 75_000_000  # ~300MB with BF16
            else:
                SAFETY_THRESHOLD_ELEMENTS = 50_000_000  # ~200MB FP32 minimum
            
            if self._no_memory_limits:
                # Never use CPU when no_memory_limits is enabled, process everything on GPU
                use_cpu = False
                cpu_threshold = float('inf')
            elif self._adaptive_manager is not None:
                use_cpu = self._should_use_cpu('hadamard', total_elements)
                cpu_threshold = self._get_hadamard_threshold()
                # Safety override: Force CPU for very large tensors regardless of adaptive decision
                # This prevents OOM when the adaptive manager is too optimistic
                safety_limit = SAFETY_THRESHOLD_ELEMENTS * 2  # ~400MB FP32, ~1200MB BF16
                if total_elements > safety_limit:
                    if not use_cpu:
                        verbose(f"[QuIP] Safety override: Large tensor (W:{weight_elements} H:{hessian_elements} elements) forcing CPU")
                    use_cpu = True
            else:
                # Static threshold based on streaming mode
                if self._no_memory_limits:
                    use_cpu = False
                else:
                    # Lower threshold = more aggressive CPU offloading
                    if self.streaming_mode == 'aggressive':
                        cpu_threshold = 25_000_000   # Most aggressive - offload smaller tensors
                    elif self.streaming_mode == 'balanced':
                        cpu_threshold = 33_000_000   # Moderate offloading
                    elif self.streaming_mode == 'minimal':
                        cpu_threshold = 50_000_000   # Conservative - only large tensors
                    else:
                        cpu_threshold = 33_000_000   # Default to balanced
                    
                    # Apply BF16 multiplier for consistent behavior with auto mode
                    if bf16_available:
                        cpu_threshold = int(cpu_threshold * 2.0)
                    
                    use_cpu = self.low_memory or (total_elements > cpu_threshold)
            
            calc_device = "cpu" if use_cpu else self.device
            
            # Log streaming decision for adaptive mode
            self._log_streaming_decision(
                operation='hadamard',
                device='cpu' if use_cpu else 'gpu',
                tensor_shape=(M_pad, N_pad),
                reason=f"{total_elements} elements vs {cpu_threshold:.0f} threshold"
            )
            
            # Inform about large tensor processing decisions
            if total_elements > 50_000_000:
                verbose(f"[QuIP] Large tensor: using {calc_device}")
            
            # Set seed once before generating all random signs
            if self.seed is not None:
                torch.manual_seed(self.seed)
            
            # Compute signs (small tensors, OK to keep)
            s_u = self._get_random_signs(M_pad).to(calc_device)
            s_v = self._get_random_signs(N_pad).to(calc_device)
            
            # Memory cleanup before large transform operations
            if calc_device == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try Hadamard on calc_device, fallback to CPU on OOM
            try:
                # For W: pad -> apply signs -> 2D Hadamard (fused for efficiency)
                if M != M_pad or N != N_pad:
                    import torch.nn.functional as F
                    W_padded = F.pad(W.to(calc_device), (0, N_pad - N, 0, M_pad - M))
                    
                    # Apply signs
                    W_padded.mul_(s_u.view(-1, 1))
                    W_padded.mul_(s_v.view(1, -1))
                    
                    # Apply H to rows and cols (matching working version order)
                    W_prime = fast_hadamard_transform(W_padded.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)
                    
                    # Keep full padded tensor for correct inverse transform
                    del W_padded
                else:
                    # Apply signs
                    W_prime = W.to(calc_device) * s_u.view(-1, 1) * s_v.view(1, -1)
                    
                    # Apply H to rows and cols
                    W_prime = fast_hadamard_transform(W_prime.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)
            except torch.cuda.OutOfMemoryError:
                verbose(f"[QuIP] OOM in Hadamard W transform on {calc_device}, falling back to CPU")
                self._report_oom('hadamard_w', M_pad * N_pad)
                torch.cuda.empty_cache()
                calc_device = "cpu"
                # Retry on CPU - move sign vectors to CPU
                s_u = s_u.to(calc_device)
                s_v = s_v.to(calc_device)
                if M != M_pad or N != N_pad:
                    import torch.nn.functional as F
                    W_padded = F.pad(W.to(calc_device), (0, N_pad - N, 0, M_pad - M))
                    W_padded.mul_(s_u.view(-1, 1))
                    W_padded.mul_(s_v.view(1, -1))
                    W_prime = fast_hadamard_transform(W_padded.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)
                    del W_padded
                else:
                    W_prime = W.to(calc_device) * s_u.view(-1, 1) * s_v.view(1, -1)
                    W_prime = fast_hadamard_transform(W_prime.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)

            # Try H Hadamard on calc_device, fallback to CPU on OOM
            try:
                # For H: pad -> apply signs -> 2D Hadamard (fused for efficiency)
                if N != N_pad:
                    import torch.nn.functional as F
                    H_padded = F.pad(H.to(calc_device), (0, N_pad - N, 0, N_pad - N))
                    # For H, we apply s_v to both dimensions
                    
                    # Apply signs
                    H_padded.mul_(s_v.view(-1, 1))
                    H_padded.mul_(s_v.view(1, -1))
                    
                    # Apply H to rows and cols
                    H_prime = fast_hadamard_transform(H_padded.t()).t()
                    H_prime = fast_hadamard_transform(H_prime)
                    
                    # DO NOT slice - keep full padded tensor
                    del H_padded
                else:
                    # Apply signs
                    H_prime = H.to(calc_device) * s_v.view(-1, 1) * s_v.view(1, -1)
                    
                    # Apply H to rows and cols
                    H_prime = fast_hadamard_transform(H_prime.t(), inplace=True).t()
                    H_prime = fast_hadamard_transform(H_prime, inplace=True)
            except torch.cuda.OutOfMemoryError:
                verbose(f"[QuIP] OOM in Hadamard H transform on {calc_device}, falling back to CPU")
                self._report_oom('hadamard_h', N_pad * N_pad)
                torch.cuda.empty_cache()
                calc_device = "cpu"
                # Retry on CPU - move s_v to CPU as well
                s_v = s_v.to(calc_device)
                if N != N_pad:
                    import torch.nn.functional as F
                    H_padded = F.pad(H.to(calc_device), (0, N_pad - N, 0, N_pad - N))
                    H_padded.mul_(s_v.view(-1, 1))
                    H_padded.mul_(s_v.view(1, -1))
                    H_prime = fast_hadamard_transform(H_padded.t()).t()
                    H_prime = fast_hadamard_transform(H_prime)
                    del H_padded
                else:
                    H_prime = H.to(calc_device) * s_v.view(-1, 1) * s_v.view(1, -1)
                    H_prime = fast_hadamard_transform(H_prime.t(), inplace=True).t()
                    H_prime = fast_hadamard_transform(H_prime, inplace=True)
            
            # Return on calc_device to preserve memory savings (caller handles device placement)
            return W_prime, H_prime, None, None, s_u, s_v
        else:
            U = self._get_orthogonal_matrix(M)
            V = self._get_orthogonal_matrix(N)
            
            # Apply transformation
            # W_prime = U @ W @ V.T
            W_prime = (U @ W.to(U.device) @ V.t().to(U.device)).to(self.device)
            
            # H_prime = V @ H @ V.T
            H_prime = (V @ H.to(V.device) @ V.t().to(V.device)).to(self.device)
            
            return W_prime, H_prime, U, V, None, None

    def _quantize_block_lazy(self, W_block, Q_block, Err_block, Hinv_block, scale):
        """Vectorized column processing with lazy updates and per-row scale."""
        count = W_block.shape[1]
        # Reduced batch size to reduce memory pressure and prevent OOM
        BATCH = 16
        
        # Ensure device consistency
        if W_block.device != scale.device:
            scale = scale.to(W_block.device)
        if Hinv_block.device != W_block.device:
            Hinv_block = Hinv_block.to(W_block.device)
        for j in range(0, count, BATCH):
            end_j = min(j + BATCH, count)
            
            # Quantize batch
            w_batch = W_block[:, j:end_j]
            # scale is (M, 1), w_batch is (M, batch)
            q_batch = torch.round(w_batch / scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            Q_block[:, j:end_j] = q_batch.to(torch.int8)
            
            # Compute errors
            d_batch = torch.diag(Hinv_block[j:end_j, j:end_j])
            err_batch = (w_batch - q_batch * scale) / d_batch.unsqueeze(0)
            Err_block[:, j:end_j] = err_batch
            
            # Batched update for remaining columns in block
            if end_j < count:
                update_factors = Hinv_block[j:end_j, end_j:]
                W_block[:, end_j:] -= err_batch @ update_factors
            
            # Periodic cleanup during quantization to prevent OOM accumulation
            if j % (BATCH * 8) == 0 and W_block.device.type == 'cuda':
                torch.cuda.empty_cache()

    def _refine_with_learned_rounding(
        self,
        Q_prime: Tensor,  # Quantized in transformed space
        W_target: Tensor,  # Original transformed weights
        scale: Tensor,
        num_iter: int = 100,
        lr: float = 8e-3
    ) -> Tensor:
        """AdamW refinement of rounding decisions."""
        from torch.optim import AdamW
        
        Q_float = Q_prime.float()
        # delta is the rounding adjustment (-0.5 to 0.5)
        delta = torch.zeros_like(Q_float, requires_grad=True)
        optimizer = AdamW([delta], lr=lr)
        
        best_loss = float('inf')
        worse_loss_counter = 0
        
        for i in range(num_iter):
            optimizer.zero_grad()
            
            # Dequantize with current delta
            dq = (Q_float + delta) * scale
            
            # MSE loss against target
            loss = torch.nn.functional.mse_loss(dq, W_target)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                worse_loss_counter = 0
            else:
                worse_loss_counter += 1
            
            # Early stopping if no improvement
            if worse_loss_counter > 20:
                break
        
        return (Q_float + delta.detach()).round().clamp(-127, 127).to(torch.int8)

    def _single_ldlq_pass(self, W: Tensor, Hinv: Tensor, scale: Tensor) -> Tensor:
        """Single pass of LDLQ quantization."""
        M_work, N_work = W.shape
        Q_prime = torch.zeros_like(W, dtype=torch.int8)
        
        # Log memory status before clone
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            verbose(f"[QuIP-DEBUG] Before W.clone(): W.shape=({M_work},{N_work}), device={W.device}, "
                    f"W_size={W.numel() * 4 / (1024**3):.2f}GB, "
                    f"GPU allocated={allocated:.2f}GB/{total:.2f}GB ({allocated/total*100:.1f}%)")
        
        # Try GPU first, fallback to CPU on OOM
        original_device = W.device
        try:
            W_work = W.clone()
        except torch.cuda.OutOfMemoryError:
            verbose(f"[QuIP] OOM on GPU W.clone(), falling back to CPU")
            self._report_oom('ldlq_clone', W.numel())
            torch.cuda.empty_cache()
            # Move inputs to CPU and retry
            W = W.cpu()
            Hinv = Hinv.cpu()
            scale = scale.cpu()
            W_work = W.clone()
            verbose(f"[QuIP] Successfully cloned on CPU, device={W_work.device}")
        
        # Ensure all inputs are on the same device
        target_device = W_work.device
        if Hinv.device != target_device:
            Hinv = Hinv.to(target_device)
        if scale.device != target_device:
            scale = scale.to(target_device)
        
        pbar = tqdm(range(0, N_work, self.block_size), desc="    LDLQ Pass", leave=False)
        for i1 in pbar:
            i2 = min(i1 + self.block_size, N_work)
            count = i2 - i1
            
            W_block = W_work[:, i1:i2].clone()
            Q_block = self._q_buffer[:, :count]
            Err_block = self._err_buffer[:, :count]
            Hinv_block = Hinv[i1:i2, i1:i2]
            
            from ..comfy.gptq_kernels import HAS_TRITON
            if self.use_triton and HAS_TRITON and scale.numel() == 1:
                from ..comfy.gptq_kernels import triton_gptq_quant_block
                Q_block, Err_block = triton_gptq_quant_block(W_block, Hinv_block, scale)
            else:
                self._quantize_block_lazy(W_block, Q_block, Err_block, Hinv_block, scale)
            
            Q_prime[:, i1:i2] = Q_block
            if i2 < N_work:
                update_slice = Hinv[i1:i2, i2:]
                if Err_block.device != W_work.device:
                    Err_block = Err_block.to(W_work.device)
                if update_slice.device != W_work.device:
                    update_slice = update_slice.to(W_work.device)
                
                # Memory-efficient update: process in chunks to avoid large intermediate tensors
                # Err_block @ update_slice can be huge (e.g., [4096, 128] @ [128, 3968] = [4096, 3968])
                # Process in column chunks to reduce peak memory
                remaining_cols = N_work - i2
                chunk_size = min(512, remaining_cols)  # Process 512 columns at a time
                
                # BF16 optimization check for LDLQ updates
                from ..constants import should_use_bf16_for_op
                use_bf16 = should_use_bf16_for_op(Err_block.numel() * update_slice.numel(), "ldlq")
                device_type = 'cuda' if Err_block.is_cuda else 'cpu'
                
                for col_start in range(0, remaining_cols, chunk_size):
                    col_end = min(col_start + chunk_size, remaining_cols)
                    try:
                        # Compute partial result for this chunk only
                        # Use BF16 for large matrix multiplications
                        if use_bf16:
                            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                                chunk_update = Err_block @ update_slice[:, col_start:col_end]
                            chunk_update = chunk_update.float()
                        else:
                            chunk_update = Err_block @ update_slice[:, col_start:col_end]
                        W_work[:, i2 + col_start:i2 + col_end] -= chunk_update
                        # Free the chunk update immediately
                        del chunk_update
                        if W_work.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError:
                        verbose(f"[QuIP] OOM in update chunk {col_start}, forcing CPU")
                        self._report_oom('ldlq_update_chunk', chunk_size * Err_block.shape[0])
                        torch.cuda.empty_cache()
                        # Move to CPU and continue
                        if W_work.device.type == 'cuda':
                            W_work = W_work.cpu()
                        if Err_block.device.type == 'cuda':
                            Err_block = Err_block.cpu()
                        if update_slice.device.type == 'cuda':
                            update_slice = update_slice.cpu()
                        # Retry this chunk on CPU
                        if use_bf16:
                            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                                chunk_update = Err_block @ update_slice[:, col_start:col_end]
                            chunk_update = chunk_update.float()
                        else:
                            chunk_update = Err_block @ update_slice[:, col_start:col_end]
                        W_work[:, i2 + col_start:i2 + col_end] -= chunk_update
                        del chunk_update
                
        # Move result back to original device if processed on CPU
        if Q_prime.device != original_device:
            try:
                Q_prime = Q_prime.to(original_device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"[QuIP] Cannot move result back to GPU, keeping on CPU")
        
        return Q_prime

    def _ldlq_with_iterations(self, W: Tensor, Hinv: Tensor, scale: Tensor, iterations: int = 1) -> Tensor:
        """Multi-pass LDLQ for better error compensation."""
        # Check if we should use checkpointed quantization
        M, N = W.shape
        
        # Log tensor info and memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            verbose(f"[QuIP] LDLQ start: W.shape=({M},{N}), device={W.device}, "
                    f"GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        else:
            verbose(f"[QuIP] LDLQ start: W.shape=({M},{N}), device={W.device}")
        
        # Auto-enable checkpointed LDLQ for large matrices in streaming/low_memory mode
        # This prevents OOM by processing in segments
        # BUT: Skip auto-enable if BF16 compute mode is active (BF16 already saves memory)
        from ..constants import _get_bf16_if_supported
        bf16_compute_active = _get_bf16_if_supported() == torch.bfloat16
        
        # Only auto-enable checkpointed if:
        # 1. Not already using BF16 (BF16 provides memory savings)
        # 2. Either explicitly requested, or in streaming/low_memory mode
        # 3. Matrix is large enough to benefit
        # 4. Single iteration only (checkpointed doesn't support multi-iter)
        if self.use_checkpointed_ldlq:
            # User explicitly requested checkpointed - always use it
            should_use_checkpointed = (
                N >= self.checkpointed_ldlq_threshold and
                iterations <= 1
            )
        elif bf16_compute_active:
            # BF16 compute mode is active - it already provides memory savings
            # Only use checkpointed if explicitly requested (handled above)
            should_use_checkpointed = False
            verbose(f"[QuIP] BF16 compute mode active - skipping checkpointed quantization for speed")
        else:
            # Auto-enable for streaming/low_memory mode
            should_use_checkpointed = (
                (self.streaming or self.low_memory) and 
                N >= self.checkpointed_ldlq_threshold and
                iterations <= 1
            )
        
        verbose(f"[QuIP-DEBUG] Checkpointed check: use_checkpointed={self.use_checkpointed_ldlq}, "
                f"streaming={self.streaming}, low_memory={self.low_memory}, "
                f"bf16_active={bf16_compute_active}, "
                f"N={N} >= threshold={self.checkpointed_ldlq_threshold}, "
                f"iterations={iterations} <= 1, should_use={should_use_checkpointed}")
        
        if should_use_checkpointed:
            from ..utils.checkpointed_quant import checkpointed_ldlq
            verbose(f"[QuIP] Using checkpointed LDLQ for {M}x{N} matrix "
                    f"({self.checkpoint_segments} segments)")
            try:
                return checkpointed_ldlq(
                    W, Hinv, scale,
                    block_size=self.block_size,
                    checkpoint_segments=self.checkpoint_segments,
                    use_tqdm=True
                )
            except torch.cuda.OutOfMemoryError:
                verbose(f"[QuIP-RECOVERY] OOM in checkpointed LDLQ on GPU, trying CPU")
                self._report_oom('checkpointed_ldlq_gpu', M * N)
                torch.cuda.empty_cache()
                # Move to CPU and retry with more segments
                W_cpu = W.cpu()
                Hinv_cpu = Hinv.cpu()
                scale_cpu = scale.cpu()
                result = checkpointed_ldlq(
                    W_cpu, Hinv_cpu, scale_cpu,
                    block_size=self.block_size,
                    checkpoint_segments=self.checkpoint_segments * 2,  # More segments on CPU
                    use_tqdm=True
                )
                verbose(f"[QuIP-RECOVERY] Checkpointed LDLQ completed on CPU")
                return result
        
        if iterations <= 1:
            return self._single_ldlq_pass(W, Hinv, scale)
            
        Q_accum = torch.zeros_like(W, dtype=torch.float32)
        
        for iter_idx in range(iterations):
            if iter_idx == 0:
                W_input = W
            else:
                # Use residual from previous pass
                W_input = W - Q_accum * scale
            
            Q_iter = self._single_ldlq_pass(W_input, Hinv, scale)
            Q_accum += Q_iter.float()
            Q_accum.clamp_(-127, 127)
        
        return Q_accum.to(torch.int8)

    def convert(
        self,
        weight: Tensor,
        H: Optional[Tensor] = None,
        activation_scales: Optional[Tensor] = None,
        smoothquant_alpha: float = 0.5
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, int, int]:
        """
        Quantize weight using QuIP.
        
        Returns:
            Tuple of (
                q_tensor,           # Quantized INT8 weights
                scale,              # Scale factor(s)
                dequantized_weight, # Dequantized weight for quality check
                smooth_factors,     # SmoothQuant factors (optional)
                s_u,                # Sign vector for output dimension (optional)
                s_v,                # Sign vector for input dimension (optional)
                hadamard_quip,      # Whether this is Hadamard-QuIP format
                hadamard_size_out,  # Padded output dimension for Hadamard
                hadamard_size_in,   # Padded input dimension for Hadamard
            )
        """
        # Track ORIGINAL dimensions for final slicing
        M_orig, N_orig = weight.shape
        
        # Avoid unnecessary clones if already on device and float
        W = weight.to(self.device, dtype=torch.float32)
        
        if H is None:
            H = torch.eye(N_orig, device=self.device, dtype=torch.float32)
        else:
            H = H.to(self.device, dtype=torch.float32)

        # Apply SmoothQuant if activation scales provided
        smooth_factors = None
        if activation_scales is not None:
            from .smoothquant import SmoothQuantPreprocessor
            smoother = SmoothQuantPreprocessor(alpha=smoothquant_alpha)
            smooth_factors = smoother.compute_smoothing_factors(W, activation_scales.to(self.device))
            W = smoother.apply_to_weight(W, smooth_factors)
            
            # Also smooth the Hessian
            # H is X.T @ X, so after smoothing X by 1/s: H_smooth = diag(1/s) @ H @ diag(1/s)
            inv_s = 1.0 / smooth_factors
            H = H * inv_s.unsqueeze(0) * inv_s.unsqueeze(1)

        # 2. Apply Incoherence Processing
        W, H, U, V, s_u, s_v = self._apply_incoherence(W, H)
        
        # Get actual working dimensions (may be padded)
        M, N = W.shape
        
        # Determine if LDLQ will run on CPU (to avoid unnecessary GPU transfers)
        if self._no_memory_limits:
            use_cpu_ldlq = False  # NO MEMORY LIMITS: Always use GPU
        else:
            ldlq_threshold = self._get_ldlq_threshold()
            use_cpu_ldlq = self._should_use_cpu('ldlq', N) or ((self.low_memory or self.streaming) and N > ldlq_threshold)
        
        # Ensure tensors are on the correct device for subsequent operations
        # (they may be on CPU if low_memory mode processed them there)
        # OPTIMIZATION: If LDLQ will run on CPU, keep tensors on CPU to avoid double transfer
        if not use_cpu_ldlq:
            if W.device != self.device:
                W = W.to(self.device)
            if H.device != self.device:
                H = H.to(self.device)
        else:
            # Keep on CPU for LDLQ - will be moved to GPU after quantization if needed
            verbose(f"[QuIP] Keeping tensors on CPU for LDLQ (dim={N})")
        
        # 3. Damping (same as GPTQ)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(N, device=H.device)
        H[diag, diag] += damp
        
        # 4. ActOrder (optional but recommended for QuIP)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            
            # In low_memory mode, do permutation on CPU to avoid OOM from fancy indexing
            if self.low_memory:
                H_cpu = H.cpu()
                perm_cpu = perm.cpu()
                H = H_cpu[perm_cpu][:, perm_cpu]
                W_cpu = W.cpu()
                W = W_cpu[:, perm_cpu]
            else:
                W = W[:, perm]
                H = H[perm][:, perm]
        
        # 5. Hessian Inversion (Cholesky)
        # Use CPU for inversion if N is large to save VRAM
        # ADAPTIVE STREAMING: Use dynamic threshold from adaptive manager if available
        if self._no_memory_limits:
            use_cpu_hessian = False  # NO MEMORY LIMITS: Always use GPU
            hessian_threshold = float('inf')
        else:
            hessian_threshold = self._get_hessian_threshold()
            use_cpu_hessian = self._should_use_cpu('hessian', N) or (N > hessian_threshold and self.low_memory)
        hessian_device = "cpu" if use_cpu_hessian else self.device
        H_inv_work = H.to(hessian_device)
        
        # Log Hessian streaming decision
        self._log_streaming_decision(
            operation='hessian',
            device='cpu' if use_cpu_hessian else 'gpu',
            tensor_shape=(N, N),
            reason=f"dim={N} vs threshold={hessian_threshold:.0f}"
        )
        
        # Run LDLQ on CPU for large matrices in low_memory/streaming mode to avoid OOM
        # ADAPTIVE STREAMING: Use dynamic threshold from adaptive manager if available
        if self._no_memory_limits:
            use_cpu_ldlq = False  # NO MEMORY LIMITS: Always use GPU
            ldlq_threshold = float('inf')
        else:
            ldlq_threshold = self._get_ldlq_threshold()
            use_cpu_ldlq = self._should_use_cpu('ldlq', N) or ((self.low_memory or self.streaming) and N > ldlq_threshold)
        ldlq_device = "cpu" if use_cpu_ldlq else self.device
        
        # Log LDLQ streaming decision
        self._log_streaming_decision(
            operation='ldlq',
            device='cpu' if use_cpu_ldlq else 'gpu',
            tensor_shape=(M, N),
            reason=f"dim={N} vs threshold={ldlq_threshold:.0f}"
        )
        
        try:
            H_inv_chol = torch.linalg.cholesky(H_inv_work)
            H_inv = torch.cholesky_inverse(H_inv_chol)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
            Hinv = H_inv.to(ldlq_device)  # Keep on CPU for large low_memory cases
        except RuntimeError as e:
            verbose(f"      - QuIP Hessian inversion failed on {hessian_device}: {e}. Falling back to identity.")
            Hinv = torch.eye(N, device=ldlq_device)

        # 6. LDLQ Quantization Loop (similar to GPTQ)
        # Compute per-channel (row-wise) scaling for the transformed weights
        abs_max_per_row = W.abs().amax(dim=1, keepdim=True)
        scale = (abs_max_per_row / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        
        # Move W to ldlq_device if needed (for CPU-based LDLQ in low_memory mode)
        if ldlq_device != self.device:
            W = W.to(ldlq_device)
            scale = scale.to(ldlq_device)
        
        # Keep target for learned rounding if enabled (after device transfer to save memory)
        W_target = W.clone() if self.use_learned_rounding else None
        
        # Ensure buffers are on the correct device
        self._ensure_buffers(M, N, self.block_size, ldlq_device)
        
        from ..utils.buffer_pool import get_buffer_pool_stats
        buf_stats = get_buffer_pool_stats()
        if buf_stats:
            verbose(f"[QuIP-DEBUG] Buffer pool: {buf_stats['cached_entries']} entries, "
                    f"{buf_stats['total_allocated_mb']:.1f}MB total")
        
        verbose(f"[QuIP-DEBUG] Before LDLQ: W.device={W.device}, Hinv.device={Hinv.device}, "
                f"scale.device={scale.device}, ldlq_device={ldlq_device}")
        
        # AGGRESSIVE GPU: Try LDLQ on current device, fallback to CPU on OOM
        try:
            # Run LDLQ (possibly with multiple iterations)
            Q_prime = self._ldlq_with_iterations(W, Hinv, scale, iterations=self.ldlq_iterations)
        except torch.cuda.OutOfMemoryError:
            verbose(f"[QuIP-RECOVERY] OOM during LDLQ on {ldlq_device}, forcing CPU fallback")
            self._report_oom('ldlq_main', M * N)
            torch.cuda.empty_cache()
            # Move everything to CPU and retry
            W_cpu = W.cpu()
            Hinv_cpu = Hinv.cpu()
            scale_cpu = scale.cpu()
            # Clear buffers and reallocate on CPU
            self._ensure_buffers(M, N, self.block_size, 'cpu')
            Q_prime = self._ldlq_with_iterations(W_cpu, Hinv_cpu, scale_cpu, iterations=self.ldlq_iterations)
            verbose(f"[QuIP-RECOVERY] LDLQ completed on CPU")
        
        # Move Q_prime back to main device if LDLQ ran on CPU
        if Q_prime.device != self.device:
            try:
                Q_prime = Q_prime.to(self.device)
                if scale.device != self.device:
                    scale = scale.to(self.device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"[QuIP-RECOVERY] Cannot move result to GPU, keeping on CPU")
        
        # Optional: Learned Rounding Post-Optimization
        if self.use_learned_rounding and W_target is not None:
            verbose("    Refining QuIP with learned rounding...")
            Q_prime = self._refine_with_learned_rounding(Q_prime, W_target, scale)

        # 7. Compute dequantized_weight in original space for quality reporting
        
        # 8. Restore original order if ActOrder was used
        if self.actorder and perm is not None:
            inv_perm = torch.argsort(perm).to(Q_prime.device)
            Q_prime_ordered = Q_prime[:, inv_perm].contiguous()
        else:
            Q_prime_ordered = Q_prime

        # 9. Undo Incoherence for dequantized weight (always needed for quality reporting)
        calc_device = self.device  # Always use GPU for speed in streaming mode
        
        # Memory-efficient chunked dequantization on GPU
        # Process in row chunks to avoid creating large intermediate tensors
        tensor_size = Q_prime_ordered.numel()
        chunk_size = 8192  # Process 8192 rows at a time
        M_total = Q_prime_ordered.shape[0]
        
        # Determine if tensor needs chunked processing
        needs_chunking = tensor_size > 33_000_000
        
        # Check if tensor is on CPU - if so, we may need to process on CPU to avoid OOM
        tensor_is_on_cpu = Q_prime_ordered.device.type == "cpu"
        
        if needs_chunking and tensor_is_on_cpu:
            # Tensor is on CPU and is large - process on CPU to avoid GPU OOM
            debug(f"      [HADAMARD-DEVICE] Large tensor on CPU, processing on CPU (size={tensor_size})")
            calc_device = "cpu"
            W_dequant_transformed = torch.empty_like(Q_prime_ordered, dtype=torch.float32, device="cpu")
            
            for row_start in range(0, M_total, chunk_size):
                row_end = min(row_start + chunk_size, M_total)
                # Process chunk on CPU
                chunk = Q_prime_ordered[row_start:row_end].to(torch.float32)
                scale_chunk = scale[row_start:row_end] if scale.dim() > 0 and scale.shape[0] == M_total else scale
                # Ensure device consistency
                if chunk.device != W_dequant_transformed.device:
                    chunk = chunk.to(W_dequant_transformed.device)
                if scale_chunk.device != W_dequant_transformed.device:
                    scale_chunk = scale_chunk.to(W_dequant_transformed.device)
                W_dequant_transformed[row_start:row_end] = chunk * scale_chunk
                del chunk  # Free intermediate
        elif needs_chunking and Q_prime_ordered.device.type == "cuda":
            # Large tensor on GPU: process in chunks with OOM protection
            debug(f"      [HADAMARD-DEVICE] Using chunked dequantization (size={tensor_size}, chunk_size={chunk_size})")
            
            try:
                W_dequant_transformed = torch.empty_like(Q_prime_ordered, dtype=torch.float32, device=calc_device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      [OOM-FALLBACK] GPU OOM allocating output tensor, falling back to CPU")
                self._report_oom('dequant_alloc', tensor_size)
                torch.cuda.empty_cache()
                calc_device = "cpu"
                # Move input to CPU for processing
                Q_prime_ordered = Q_prime_ordered.cpu()
                scale = scale.cpu()
                W_dequant_transformed = torch.empty_like(Q_prime_ordered, dtype=torch.float32, device="cpu")
            
            for row_start in range(0, M_total, chunk_size):
                row_end = min(row_start + chunk_size, M_total)
                try:
                    # Process chunk
                    chunk = Q_prime_ordered[row_start:row_end].to(torch.float32)
                    scale_chunk = scale[row_start:row_end] if scale.dim() > 0 and scale.shape[0] == M_total else scale
                    # Ensure device consistency
                    if chunk.device != W_dequant_transformed.device:
                        chunk = chunk.to(W_dequant_transformed.device)
                    if scale_chunk.device != W_dequant_transformed.device:
                        scale_chunk = scale_chunk.to(W_dequant_transformed.device)
                    W_dequant_transformed[row_start:row_end] = chunk * scale_chunk
                    del chunk  # Free intermediate
                    if calc_device == "cuda" and row_start % (chunk_size * 4) == 0:  # Periodic cleanup
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    verbose(f"      [OOM-RECOVERY] GPU OOM at row {row_start}, switching to CPU for remaining chunks")
                    self._report_oom('dequant_chunk', chunk_size * Q_prime_ordered.shape[1])
                    torch.cuda.empty_cache()
                    # Move remaining processing to CPU
                    if calc_device == "cuda":
                        calc_device = "cpu"
                        Q_prime_ordered = Q_prime_ordered.cpu()
                        scale = scale.cpu()
                        # Move already processed data to CPU
                        W_dequant_transformed = W_dequant_transformed.cpu()
                    # Retry this chunk on CPU
                    chunk = Q_prime_ordered[row_start:row_end].to(torch.float32)
                    scale_chunk = scale[row_start:row_end] if scale.dim() > 0 and scale.shape[0] == M_total else scale
                    W_dequant_transformed[row_start:row_end] = chunk * scale_chunk
                    del chunk
        else:
            # Small tensor: process all at once
            # Add OOM protection - fall back to CPU if GPU runs out of memory
            try:
                W_dequant_transformed = Q_prime_ordered.to(torch.float32).to(calc_device) * scale.to(calc_device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      [OOM-FALLBACK] GPU OOM during dequantization, falling back to CPU")
                # Report OOM to adaptive manager for learning
                self._report_oom('hadamard', Q_prime_ordered.numel())
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                # Process on CPU
                Q_cpu = Q_prime_ordered.cpu()
                scale_cpu = scale.cpu()
                W_dequant_transformed = Q_cpu.to(torch.float32) * scale_cpu
                calc_device = "cpu"
        
        if s_u is not None and s_v is not None:
            # Inverse transform using working version's approach:
            # 1. Apply Hadamard to rows
            # 2. Apply Hadamard to columns (via transpose)
            # 3. Apply signs AFTER Hadamard
            
            try:
                W_undo = fast_hadamard_transform(W_dequant_transformed)
                W_undo = fast_hadamard_transform(W_undo.t()).t()
                W_undo = W_undo * s_u.to(calc_device).view(-1, 1) * s_v.to(calc_device).view(1, -1)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      [OOM-FALLBACK] GPU OOM during Hadamard inverse transform, falling back to CPU")
                self._report_oom('hadamard_inverse', W_dequant_transformed.numel())
                torch.cuda.empty_cache()
                # Move to CPU and retry
                calc_device = "cpu"
                W_cpu = W_dequant_transformed.cpu()
                s_u_cpu = s_u.cpu()
                s_v_cpu = s_v.cpu()
                W_undo = fast_hadamard_transform(W_cpu)
                W_undo = fast_hadamard_transform(W_undo.t()).t()
                W_undo = W_undo * s_u_cpu.view(-1, 1) * s_v_cpu.view(1, -1)
            
            # Slice to original dimensions
            dequantized_weight = W_undo[:M_orig, :N_orig].contiguous()
        else:
            try:
                dequantized_weight = U.t().to(calc_device) @ W_dequant_transformed @ V.to(calc_device)
                dequantized_weight = dequantized_weight[:M_orig, :N_orig].contiguous()
            except torch.cuda.OutOfMemoryError:
                verbose(f"      [OOM-FALLBACK] GPU OOM during orthogonal inverse transform, falling back to CPU")
                self._report_oom('ortho_inverse', W_dequant_transformed.numel())
                torch.cuda.empty_cache()
                calc_device = "cpu"
                dequantized_weight = U.t().cpu() @ W_dequant_transformed.cpu() @ V.cpu()
                dequantized_weight = dequantized_weight[:M_orig, :N_orig].contiguous()

        # 10. Undo SmoothQuant transformation if it was applied
        if smooth_factors is not None:
            # Inverse: W = W_smooth / s (from forward: W_smooth = W * s)
            dequantized_weight = dequantized_weight / smooth_factors.to(calc_device).unsqueeze(0)

        if self.store_transformed:
            # Use per-row scale from LDLQ quantization for Hadamard-QuIP
            # The weight was quantized with per-row scales, so we need to return them
            final_scale = scale
            
            # Compute padded dimensions for Hadamard-QuIP metadata
            if self.use_hadamard:
                from ..utils.hadamard import next_power_of_two
                hadamard_size_out = next_power_of_two(M_orig)
                hadamard_size_in = next_power_of_two(N_orig)
            else:
                hadamard_size_out = 0
                hadamard_size_in = 0
            
            return (
                Q_prime_ordered.cpu().contiguous(),
                final_scale.cpu().to(SCALE_DTYPE),
                dequantized_weight.cpu().contiguous(),
                smooth_factors.cpu() if smooth_factors is not None else None,
                # Keep sign vectors at padded dimensions for Hadamard-QuIP!
                s_u.cpu() if s_u is not None else None,
                s_v.cpu() if s_v is not None else None,
                # Hadamard-QuIP metadata (only when using Hadamard)
                True if self.use_hadamard else False,  # hadamard_quip flag
                hadamard_size_out,
                hadamard_size_in,
            )
        
        # Re-quantize to original space for compatibility with standard inference kernels
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        final_abs_max = dequantized_weight.abs().max()
        final_scale = (final_abs_max / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        
        # Use chunked processing for large tensors to avoid OOM
        tensor_size = dequantized_weight.numel()
        
        # For large tensors, process in chunks to keep memory usage bounded
        # Lowered threshold to match 'balanced' tier (33M elements = ~128MB)
        if tensor_size > 33_000_000 and dequantized_weight.device.type == "cuda":
            chunk_rows = 2048  # Process 2048 rows at a time
            M_total = dequantized_weight.shape[0]
            
            # Pre-allocate output tensor
            Q_final_int8 = torch.empty_like(dequantized_weight, dtype=torch.int8, device=dequantized_weight.device)
            
            for row_start in range(0, M_total, chunk_rows):
                row_end = min(row_start + chunk_rows, M_total)
                
                # Process chunk: dequantized_weight[row_start:row_end] / final_scale
                chunk = dequantized_weight[row_start:row_end]
                scale_chunk = final_scale[row_start:row_end] if final_scale.dim() > 0 and final_scale.shape[0] == M_total else final_scale
                
                # Ensure same device
                if chunk.device != Q_final_int8.device:
                    chunk = chunk.to(Q_final_int8.device)
                if scale_chunk.device != Q_final_int8.device:
                    scale_chunk = scale_chunk.to(Q_final_int8.device)
                
                # Step-by-step with explicit memory management
                divided = chunk / scale_chunk
                rounded = torch.round(divided)
                del divided  # Free intermediate
                
                clamped = rounded.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
                del rounded  # Free intermediate
                
                Q_final_int8[row_start:row_end] = clamped.to(torch.int8)
                del clamped  # Free intermediate
                
                # Periodic cleanup every few chunks
                if (row_start // chunk_rows) % 4 == 0:
                    torch.cuda.empty_cache()
        else:
            # Small tensor: process normally but with explicit steps to avoid chained intermediates
            # Add OOM protection - fall back to CPU if GPU runs out of memory
            try:
                divided = dequantized_weight / final_scale
                rounded = torch.round(divided)
                del divided  # Free intermediate immediately
                
                clamped = rounded.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
                del rounded  # Free intermediate immediately
                
                Q_final_int8 = clamped.to(torch.int8)
                del clamped  # Free intermediate
            except torch.cuda.OutOfMemoryError:
                verbose(f"      [OOM-FALLBACK] GPU OOM during final quantization, falling back to CPU")
                # Report OOM to adaptive manager for learning
                self._report_oom('hadamard', dequantized_weight.numel())
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                # Process on CPU
                dw_cpu = dequantized_weight.cpu()
                scale_cpu = final_scale.cpu() if final_scale.dim() > 0 else final_scale
                divided = dw_cpu / scale_cpu
                rounded = torch.round(divided)
                clamped = rounded.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
                Q_final_int8 = clamped.to(torch.int8)
                # Move back to GPU if possible
                try:
                    Q_final_int8 = Q_final_int8.to(self.device)
                except torch.cuda.OutOfMemoryError:
                    pass  # Keep on CPU
        
        # Return with Hadamard-QuIP metadata (flag as False since we're not storing transformed)
        return (
            Q_final_int8.cpu().contiguous(),
            final_scale.cpu().to(SCALE_DTYPE),
            dequantized_weight.cpu().contiguous(),
            smooth_factors.cpu() if smooth_factors is not None else None,
            None, # s_u not needed for standard storage
            None, # s_v not needed for standard storage
            False, # hadamard_quip flag (False for standard storage)
            0,     # hadamard_size_out
            0,     # hadamard_size_in
        )
