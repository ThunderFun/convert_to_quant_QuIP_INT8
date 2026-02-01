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
from ..utils.memory_utils import maybe_empty_cache
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
        requant_scheme: str = "tensor",  # "tensor" or "block"
        requant_tensor_use_per_row_scale: bool = True,  # Use per-row scales for tensor-wise re-quantization
        # QuIP Paper Improvements (Section 4)
        use_diagonal_rescaling: bool = True,
        use_frobenius_scaling: bool = True,
        use_random_permutation: bool = True,
        use_greedy_coordinate_descent: bool = False,
        greedy_cd_iterations: int = 100,
        frobenius_rho: float = 1.0,
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
        self.requant_scheme = requant_scheme  # "tensor" or "block"
        self.requant_tensor_use_per_row_scale = requant_tensor_use_per_row_scale
        
        # QuIP Paper Improvements (Section 4)
        self.use_diagonal_rescaling = use_diagonal_rescaling
        self.use_frobenius_scaling = use_frobenius_scaling
        
        # store_transformed=True returns weights in a transformed domain (e.g. Hadamard-QuIP)
        # along with a limited set of metadata (currently sign vectors for Hadamard-QuIP).
        # Diagonal rescaling requires D_tilde to invert, but D_tilde is not stored/returned.
        # To keep the storage contract "stored weights can be reconstructed without extra metadata",
        # we disable diagonal rescaling when storing transformed weights.
        if store_transformed and self.use_diagonal_rescaling:
            verbose(
                "WARNING: store_transformed=True is incompatible with diagonal rescaling (D_tilde is not stored). "
                "Disabling diagonal rescaling for this converter instance."
            )
            self.use_diagonal_rescaling = False
        self.use_random_permutation = use_random_permutation
        self.use_greedy_coordinate_descent = use_greedy_coordinate_descent
        self.greedy_cd_iterations = greedy_cd_iterations
        self.frobenius_rho = frobenius_rho
        
        # Store no_memory_limits flag (required for proper threshold handling)
        self._no_memory_limits = no_memory_limits
        
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
            self._adaptive_manager = AdaptiveStreamingManager()
        
        # streaming_enabled is True for all modes except "off"
        self.streaming = streaming_mode != "off"
        
        # Cache for orthogonal matrices
        self._ortho_cache = {}
        self.low_memory = False  # Will be set by quantization.py
        
        # Initialize thresholds (will be overridden by adaptive manager if in auto mode)
        if self._no_memory_limits:
            self.hadamard_threshold = float('inf')
            self.ldlq_threshold = float('inf')
            self.hessian_threshold = float('inf')
            self.ortho_threshold = float('inf')
        else:
            # Get default thresholds from streaming config
            thresholds = StreamingConfig.get_thresholds(streaming_mode if streaming_mode != "auto" else "balanced")
            self.hadamard_threshold = thresholds.hadamard_elements
            self.ldlq_threshold = thresholds.ldlq_dim
            self.hessian_threshold = thresholds.hessian_dim
            self.ortho_threshold = thresholds.ortho_dim

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
            verbose(f"Cleaned up {cleared} buffer sets from pool")
        
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
        
        try:
            # Get buffers from pool (sliced to exact size needed)
            self._q_buffer, self._err_buffer, self._W_buffer = pool.get_buffers(
                M, N, block_size, buf_device
            )
        except torch.cuda.OutOfMemoryError:
            if buf_device != 'cpu':
                verbose(f"OOM allocating buffers on {buf_device}, falling back to CPU")
                maybe_empty_cache(force=True)
                self._q_buffer, self._err_buffer, self._W_buffer = pool.get_buffers(
                    M, N, block_size, 'cpu'
                )
            else:
                raise

    def _get_random_signs(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Get random signs (+1/-1) for QuIP. Uses local generator if provided."""
        if generator is not None:
            signs = torch.randint(0, 2, (n,), device=generator.device, generator=generator).float() * 2 - 1
            return signs
        signs = torch.randint(0, 2, (n,), device=self.device).float() * 2 - 1
        return signs
    
    def _apply_diagonal_rescaling(self, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply diagonal rescaling as per QuIP paper Algorithm 1, Lines 3-4.
        
        D_tilde ← ∜(diag(H) / diag(W^T W))  # fourth root, element-wise
        W ← W · D_tilde
        H ← D_tilde^(-1) · H · D_tilde^(-1)
        
        Args:
            W: Weight matrix (M x N)
            H: Hessian matrix (N x N)
            
        Returns:
            Tuple of (rescaled W, rescaled H, D_tilde diagonal matrix)
        """
        device = W.device
        M, N = W.shape
        
        # Compute diag(H) - Hessian diagonal
        H_diag = torch.diag(H).to(device)
        
        # Compute diag(W^T @ W) - column-wise squared norms
        WtW_diag = (W ** 2).sum(dim=0)  # Sum over rows = W^T @ W diagonal
        
        # Avoid division by zero
        WtW_diag = torch.clamp(WtW_diag, min=1e-12)
        H_diag = torch.clamp(H_diag, min=1e-12)
        
        # D_tilde = fourth_root(diag(H) / diag(W^T @ W))
        # Use double precision for diagonal rescaling to ensure consistency across devices
        ratio = H_diag.double() / WtW_diag.double()
        D_tilde = torch.pow(ratio, 0.25).to(device, dtype=torch.float32)  # Fourth root
        
        # Clamp D_tilde to reasonable range to prevent numerical instability
        D_tilde = torch.clamp(D_tilde, min=0.1, max=10.0)
        
        # Rescale W: W ← W · D_tilde (multiply each column by corresponding D_tilde element)
        W_rescaled = W * D_tilde.unsqueeze(0)
        
        # Rescale H: H ← D_tilde^(-1) · H · D_tilde^(-1)
        D_inv = 1.0 / D_tilde
        H_rescaled = H * D_inv.unsqueeze(0) * D_inv.unsqueeze(1)
        
        return W_rescaled, H_rescaled, D_tilde

    def _undo_diagonal_rescaling(self, W: Tensor, D_tilde: Tensor) -> Tensor:
        """
        Undo diagonal rescaling (inverse of _apply_diagonal_rescaling).
        
        W ← W / D_tilde (element-wise division of columns)
        
        Args:
            W: Weight matrix (M x N)
            D_tilde: Diagonal scaling factors from _apply_diagonal_rescaling
            
        Returns:
            Unscaled weight matrix
        """
        # Avoid division by zero - D_tilde should never be zero, but clamp for safety
        D_tilde_safe = torch.clamp(D_tilde, min=1e-12)
        return W / D_tilde_safe.unsqueeze(0)
    
    def _apply_random_permutation(self, W: Tensor, H: Tensor, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply random permutation to weights as per QuIP paper Section 4.3.
        
        "We also randomly permute entries at the fast matrix multiplication step
        to prevent any correlation between attention heads from worsening performance."
        
        Args:
            W: Weight matrix (M x N)
            H: Hessian matrix (N x N)
            generator: Optional generator for reproducibility
            
        Returns:
            Tuple of (permuted W, permuted H, permutation indices)
        """
        device = W.device
        M, N = W.shape
        
        # Generate random permutation on CPU for reproducibility across devices
        # Keep perm on CPU initially, only move to target device when needed
        if generator is not None:
            perm = torch.randperm(N, device='cpu', generator=generator)
        else:
            perm = torch.randperm(N, device='cpu')
        
        try:
            # Move perm to target device for GPU permutation attempt
            perm_device = perm.to(device)
            
            # Apply permutation to W columns
            W_permuted = W[:, perm_device]
            
            # Apply permutation to H (both rows and columns)
            # Fancy indexing H[perm][:, perm] creates a large intermediate copy
            H_permuted = H[perm_device][:, perm_device]
        except torch.cuda.OutOfMemoryError:
            verbose(f"OOM during GPU permutation, falling back to CPU")
            self._report_oom('perm_gpu', N * N)
            maybe_empty_cache(force=True)
            # Move to CPU and retry - perm is already on CPU
            W_cpu = W.cpu()
            H_cpu = H.cpu()
            W_permuted = W_cpu[:, perm].to(device)
            H_permuted = H_cpu[perm][:, perm].to(device)
        
        # Return perm on CPU for consistency, caller can move to device if needed
        return W_permuted, H_permuted, perm
    
    def _undo_permutation(self, W: Tensor, perm: Tensor) -> Tensor:
        """Undo the random permutation."""
        inv_perm = torch.argsort(perm)
        return W[:, inv_perm]
    
    def _apply_orthogonal(self, x: Tensor, dim: int) -> Tensor:
        """
        Apply orthogonal transformation to x along dim.
        Uses Fast Hadamard Transform if possible, otherwise full matrix.
        """
        n = x.shape[dim]
        if self.use_hadamard and is_power_of_two(n):
            # Apply random signs then Hadamard
            # Ensure signs are on same device as x to avoid device mismatch
            signs = self._get_random_signs(n).to(x.device)
            
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
            # Ensure Q is on same device as x
            if Q.device != x.device:
                Q = Q.to(x.device)
            
            if dim == 0:
                return Q @ x
            elif dim == -1 or dim == x.ndim - 1:
                return x @ Q.t()
            else:
                # For middle dimensions, transpose, apply, transpose back
                # e.g., for 3D tensor with dim=1: [B, N, C] -> [B, C, N] -> apply -> [B, C, N] -> [B, N, C]
                x = x.transpose(dim, -1)
                x = x @ Q.t()
                return x.transpose(dim, -1)

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

    def _apply_incoherence(self, W: Tensor, H: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Transform W and H to make them more "incoherent" (uniform).
        W' = U @ W @ V^T
        H' = V @ H @ V^T
        
        For non-power-of-2 dimensions with Hadamard transform:
        - Returns FULL padded tensor (M_pad x N_pad) to preserve transform information
        - Original dimensions should be tracked by caller for final slicing
        - Sign vectors are returned at full padded length
        
        QuIP Paper Improvements:
        - Diagonal rescaling (Algorithm 1, Lines 3-4)
        - Random permutation (Section 4.3)
        """
        M, N = W.shape
        
        # Track if we applied diagonal rescaling (need to undo later)
        D_tilde = None
        perm = None
        
        # QuIP Paper Improvement: Diagonal Rescaling
        if self.use_diagonal_rescaling:
            verbose("Applying diagonal rescaling (Algorithm 1, Lines 3-4)")
            W, H, D_tilde = self._apply_diagonal_rescaling(W, H)
        
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
            
            if total_elements > 50_000_000:
                verbose(f"Large tensor: using {calc_device}")
            
            # Create local generator for random signs to avoid affecting global RNG
            # ALWAYS use CPU generator for reproducibility across devices
            sign_gen = None
            if self.seed is not None:
                sign_gen = torch.Generator(device='cpu').manual_seed(self.seed)
            
            s_u = self._get_random_signs(M_pad, generator=sign_gen).to(calc_device)
            s_v = self._get_random_signs(N_pad, generator=sign_gen).to(calc_device)
            
            # Memory cleanup before large transform operations (conditional)
            if calc_device == "cpu":
                maybe_empty_cache(pressure_threshold=0.80)
            
            # Try Hadamard on calc_device, fallback to CPU on OOM
            try:
                # For W: pad -> apply signs -> 2D Hadamard (fused for efficiency)
                if M != M_pad or N != N_pad:
                    import torch.nn.functional as F
                    W_padded = F.pad(W.to(calc_device), (0, N_pad - N, 0, M_pad - M))
                    W_padded.mul_(s_u.view(-1, 1))
                    W_padded.mul_(s_v.view(1, -1))
                    # Apply H to rows and cols (matching working version order)
                    W_prime = fast_hadamard_transform(W_padded.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)
                    del W_padded
                else:
                    W_prime = W.to(calc_device) * s_u.view(-1, 1) * s_v.view(1, -1)
                    W_prime = fast_hadamard_transform(W_prime.t()).t()
                    W_prime = fast_hadamard_transform(W_prime)
            except torch.cuda.OutOfMemoryError:
                verbose(f"OOM in Hadamard W transform on {calc_device}, falling back to CPU")
                self._report_oom('hadamard_w', M_pad * N_pad)
                maybe_empty_cache(force=True)  # Force cleanup on OOM
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
                    H_padded.mul_(s_v.view(-1, 1))
                    H_padded.mul_(s_v.view(1, -1))
                    H_prime = fast_hadamard_transform(H_padded.t()).t()
                    H_prime = fast_hadamard_transform(H_prime)
                    del H_padded
                else:
                    H_prime = H.to(calc_device) * s_v.view(-1, 1) * s_v.view(1, -1)
                    H_prime = fast_hadamard_transform(H_prime.t()).t()
                    H_prime = fast_hadamard_transform(H_prime)
            except torch.cuda.OutOfMemoryError:
                verbose(f"OOM in Hadamard H transform on {calc_device}, falling back to CPU")
                self._report_oom('hadamard_h', N_pad * N_pad)
                maybe_empty_cache(force=True)  # Force cleanup on OOM
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
                    H_prime = fast_hadamard_transform(H_prime.t()).t()
                    H_prime = fast_hadamard_transform(H_prime)
            
            # QuIP Paper Improvement: Random Permutation
            # Apply AFTER Hadamard transform (Section 4.3: "at the fast matrix multiplication step")
            if self.use_random_permutation:
                debug("Applying random permutation (Section 4.3)")
                W_prime, H_prime, perm = self._apply_random_permutation(W_prime, H_prime, generator=sign_gen)
            
            # Return on calc_device to preserve memory savings (caller handles device placement)
            return W_prime, H_prime, None, None, s_u, s_v, D_tilde, perm
        else:
            U = self._get_orthogonal_matrix(M)
            V = self._get_orthogonal_matrix(N)
            
            # Apply transformation
            # W_prime = U @ W @ V.T
            W_prime = (U @ W.to(U.device) @ V.t().to(U.device)).to(self.device)
            
            # H_prime = V @ H @ V.T
            H_prime = (V @ H.to(V.device) @ V.t().to(V.device)).to(self.device)
            
                # QuIP Paper Improvement: Random Permutation
            if self.use_random_permutation:
                debug("Applying random permutation (Section 4.3)")
                # Create local generator for random permutation if seed provided
                perm_gen = None
                if self.seed is not None:
                    perm_gen = torch.Generator(device='cpu').manual_seed(self.seed)
                W_prime, H_prime, perm = self._apply_random_permutation(W_prime, H_prime, generator=perm_gen)
            
            return W_prime, H_prime, U, V, None, None, D_tilde, perm

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
            # Clamp to prevent divide-by-zero
            d_batch = torch.clamp(d_batch, min=1e-12)
            err_batch = (w_batch - q_batch * scale) / d_batch.unsqueeze(0)
            Err_block[:, j:end_j] = err_batch
            
            # Batched update for remaining columns in block
            if end_j < count:
                update_factors = Hinv_block[j:end_j, end_j:]
                W_block[:, end_j:] -= err_batch @ update_factors
            
            # Periodic cleanup during quantization (conditional, rate-limited)
            if j % (BATCH * 32) == 0 and W_block.device.type == 'cuda':
                maybe_empty_cache(pressure_threshold=0.92)

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

    def _refine_with_greedy_coordinate_descent(
        self,
        Q_prime: Tensor,  # Quantized in transformed space
        W_target: Tensor,  # Original transformed weights
        H: Tensor,  # Hessian for computing loss
        scale: Tensor,
        num_iterations: int = 100,
    ) -> Tensor:
        """
        Greedy coordinate descent as per QuIP paper Section 4.3.
        
        Updates weights in the same order as the initial pass to minimize proxy loss:
        ℓ(Ŵ) = tr(H) · ||W - Ŵ||²
        
        Args:
            Q_prime: Initial quantized weights
            W_target: Target weights to match
            H: Hessian matrix for loss computation
            scale: Quantization scale
            num_iterations: Number of coordinate descent iterations
            
        Returns:
            Refined quantized weights
        """
        verbose(f"Applying greedy coordinate descent ({num_iterations} iterations)")
        
        device = Q_prime.device
        Q_float = Q_prime.float().clone()
        W_target = W_target.to(device)
        H_diag = torch.diag(H).to(device).abs().clamp(min=1e-12)
        
        # Compute initial error
        M, N = W_target.shape
        best_loss = float('inf')
        no_improve_count = 0
        
        for iteration in range(num_iterations):
            improved = False
            
            # Process each coordinate in order (column-major to match LDLQ)
            for j in range(N):
                # Get the column
                w_col = W_target[:, j]
                q_col = Q_float[:, j]
                h_j = H_diag[j]
                
                # Compute current dequantized values
                dq_col = q_col * scale.squeeze()
                
                # Compute errors for this column
                errors = w_col - dq_col
                
                # For each row in the column, try +1 and -1 adjustments
                for i in range(M):
                    current_q = Q_float[i, j].item()
                    current_err = errors[i].item()
                    
                    # Try +1 adjustment
                    dq_plus = (current_q + 1) * scale[i].item() if scale.dim() > 0 else (current_q + 1) * scale.item()
                    err_plus = abs(w_col[i].item() - dq_plus)
                    
                    # Try -1 adjustment
                    dq_minus = (current_q - 1) * scale[i].item() if scale.dim() > 0 else (current_q - 1) * scale.item()
                    err_minus = abs(w_col[i].item() - dq_minus)
                    
                    # Current error
                    err_current = abs(current_err)
                    
                    # Choose best adjustment weighted by Hessian diagonal
                    weighted_plus = err_plus * h_j
                    weighted_minus = err_minus * h_j
                    weighted_current = err_current * h_j
                    
                    if weighted_plus < weighted_current and weighted_plus <= weighted_minus:
                        Q_float[i, j] = current_q + 1
                        improved = True
                    elif weighted_minus < weighted_current and weighted_minus < weighted_plus:
                        Q_float[i, j] = current_q - 1
                        improved = True
            
            # Compute loss for this iteration
            dq_current = Q_float * scale
            loss = torch.mean(H_diag * ((W_target - dq_current) ** 2).sum(dim=0))
            
            if loss < best_loss:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= 10:
                verbose(f"Greedy CD early stop at iteration {iteration}")
                break
        
        return Q_float.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round().to(torch.int8)

    def _single_ldlq_pass(self, W: Tensor, Hinv: Tensor, scale: Tensor) -> Tensor:
        """Single pass of LDLQ quantization."""
        M_work, N_work = W.shape
        
        # Try GPU first, fallback to CPU on OOM
        original_device = W.device
        try:
            W_work = W.clone()
            target_device = W_work.device
            # Allocate Q_prime AFTER device is determined
            Q_prime = torch.zeros_like(W_work, dtype=torch.int8)
        except torch.cuda.OutOfMemoryError:
            verbose(f"OOM on GPU W.clone(), falling back to CPU")
            self._report_oom('ldlq_clone', W.numel())
            torch.cuda.empty_cache()
            # Move inputs to CPU and retry
            W = W.cpu()
            Hinv = Hinv.cpu()
            scale = scale.cpu()
            W_work = W.clone()
            target_device = W_work.device
            # Allocate Q_prime on CPU
            Q_prime = torch.zeros_like(W_work, dtype=torch.int8)
            verbose(f"Successfully cloned on CPU, device={W_work.device}")
        
        # Ensure all inputs are on the same device
        if Hinv.device != target_device:
            Hinv = Hinv.to(target_device)
        if scale.device != target_device:
            scale = scale.to(target_device)
        
        # Ensure buffers are on the correct device
        if self._q_buffer is None or self._q_buffer.device != target_device:
            self._ensure_buffers(M_work, N_work, self.block_size, str(target_device))
        
        pbar = tqdm(range(0, N_work, self.block_size), desc="    LDLQ Pass", leave=False)
        for i1 in pbar:
            # Update progress bar with device info
            if i1 % (self.block_size * 4) == 0:
                pbar.set_postfix(device=str(W_work.device), refresh=False)
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
                remaining_cols = N_work - i2
                is_cuda = Err_block.is_cuda
                
                # Adaptive chunk size: larger for CPU, smaller for GPU
                if not is_cuda:
                    chunk_size = min(8192, remaining_cols)
                else:
                    chunk_size = min(1024, remaining_cols)
                
                from ..constants import should_use_bf16_for_op
                # Only use BF16 on CUDA; FP32 is usually faster on CPU
                use_bf16 = is_cuda and should_use_bf16_for_op(Err_block.numel() * update_slice.numel(), "ldlq")
                device_type = 'cuda' if is_cuda else 'cpu'
                
                # Optimization: If remaining part is small enough, do it in one go
                if remaining_cols <= chunk_size:
                    try:
                        if use_bf16:
                            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                                W_work[:, i2:] -= (Err_block @ update_slice).float()
                        else:
                            W_work[:, i2:] -= Err_block @ update_slice
                        if is_cuda:
                            maybe_empty_cache(pressure_threshold=0.95, device=W_work.device)
                    except torch.cuda.OutOfMemoryError:
                        # Fallback to chunked if one-shot fails
                        chunk_size = 512
                        for col_start in range(0, remaining_cols, chunk_size):
                            col_end = min(col_start + chunk_size, remaining_cols)
                            W_work[:, i2 + col_start:i2 + col_end] -= Err_block @ update_slice[:, col_start:col_end]
                else:
                    # Chunked processing
                    for col_start in range(0, remaining_cols, chunk_size):
                        col_end = min(col_start + chunk_size, remaining_cols)
                        try:
                            if use_bf16:
                                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                                    chunk_update = Err_block @ update_slice[:, col_start:col_end]
                                chunk_update = chunk_update.float()
                            else:
                                chunk_update = Err_block @ update_slice[:, col_start:col_end]
                            W_work[:, i2 + col_start:i2 + col_end] -= chunk_update
                            del chunk_update
                            
                            # Only cleanup on GPU and only occasionally
                            if is_cuda and col_start % (chunk_size * 4) == 0:
                                maybe_empty_cache(pressure_threshold=0.92, device=W_work.device)
                        except torch.cuda.OutOfMemoryError:
                            verbose(f"OOM in update chunk {col_start}, forcing CPU")
                            self._report_oom('ldlq_update_chunk', chunk_size * Err_block.shape[0])
                            maybe_empty_cache(force=True)
                            if W_work.device.type == 'cuda': W_work = W_work.cpu()
                            if Err_block.device.type == 'cuda': Err_block = Err_block.cpu()
                            if update_slice.device.type == 'cuda': update_slice = update_slice.cpu()
                            # Also move buffers to CPU to avoid mixed-device operations in next iterations
                            self._ensure_buffers(M_work, N_work, self.block_size, 'cpu')
                            device_type = 'cpu'
                            is_cuda = False
                            # Continue with CPU (FP32)
                            W_work[:, i2 + col_start:i2 + col_end] -= Err_block @ update_slice[:, col_start:col_end]
                
        # Move result back to original device if processed on CPU
        if Q_prime.device != original_device:
            try:
                Q_prime = Q_prime.to(original_device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"Cannot move result back to GPU, keeping on CPU")
        
        return Q_prime

    def _ldlq_with_iterations(self, W: Tensor, Hinv: Tensor, scale: Tensor, iterations: int = 1) -> Tensor:
        """Multi-pass LDLQ for better error compensation."""
        # Check if we should use checkpointed quantization
        M, N = W.shape
        
        verbose(f"LDLQ: processing {M}x{N} matrix on {W.device}")
        
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
            verbose(f"BF16 compute mode active - skipping checkpointed quantization for speed")
        else:
            # Auto-enable for streaming/low_memory mode
            should_use_checkpointed = (
                (self.streaming or self.low_memory) and 
                N >= self.checkpointed_ldlq_threshold and
                iterations <= 1
            )
        
        if should_use_checkpointed:
            from ..utils.checkpointed_quant import checkpointed_ldlq
            verbose(f"Using checkpointed LDLQ for {M}x{N} matrix "
                    f"({self.checkpoint_segments} segments)")
            try:
                return checkpointed_ldlq(
                    W, Hinv, scale,
                    block_size=self.block_size,
                    checkpoint_segments=self.checkpoint_segments,
                    use_tqdm=True
                )
            except torch.cuda.OutOfMemoryError:
                verbose(f"OOM in checkpointed LDLQ on GPU, trying CPU")
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
                verbose(f"Checkpointed LDLQ completed on CPU")
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

    @torch.no_grad()
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
            
            # Also smooth the Hessian (H = X.T @ X)
            # After smoothing X by 1/s: H_smooth = diag(1/s) @ H @ diag(1/s)
            inv_s = 1.0 / smooth_factors
            H = H * inv_s.unsqueeze(0) * inv_s.unsqueeze(1)

        # 2. Apply Incoherence Processing
        W, H, U, V, s_u, s_v, D_tilde, rand_perm = self._apply_incoherence(W, H)
        
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
        # If LDLQ will run on CPU, keep tensors on CPU to avoid double transfer
        if not use_cpu_ldlq:
            if W.device != self.device:
                W = W.to(self.device)
            if H.device != self.device:
                H = H.to(self.device)
        else:
            # Keep on CPU for LDLQ - will be moved to GPU after quantization if needed
            verbose(f"Keeping tensors on CPU for LDLQ (dim={N})")
        
        # 3. Damping (same as GPTQ)
        # Use double precision for damping calculation to ensure consistency
        H_diag_orig = torch.diag(H)
        dead = H_diag_orig == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        damp = self.percdamp * torch.mean(H_diag_orig.double()).float()
        diag = torch.arange(N, device=H.device)
        H[diag, diag] += damp
        
        # 4. ActOrder (optional but recommended for QuIP)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            
            # Determine if permutation should happen on CPU to avoid OOM from fancy indexing
            # Fancy indexing H[perm][:, perm] creates a large intermediate copy
            if self._no_memory_limits:
                use_cpu_perm = False
            else:
                # Use Hessian threshold as a proxy for permutation memory safety
                hessian_threshold = self._get_hessian_threshold()
                use_cpu_perm = self.low_memory or self.streaming or (N > hessian_threshold)
            
            if use_cpu_perm:
                verbose(f"Permuting tensors on CPU to avoid OOM (dim={N})")
                try:
                    H_cpu = H.cpu()
                    perm_cpu = perm.cpu()
                    H = H_cpu[perm_cpu][:, perm_cpu]
                    W_cpu = W.cpu()
                    W = W_cpu[:, perm_cpu]
                except torch.cuda.OutOfMemoryError:
                    # If we OOM just trying to move to CPU (unlikely but possible if GPU is very full)
                    self._report_oom('perm_cpu_move', N * N)
                    maybe_empty_cache(force=True)
                    # Retry with more aggressive cleanup
                    H = H.cpu()[perm.cpu()][:, perm.cpu()]
                    W = W.cpu()[:, perm.cpu()]
            else:
                try:
                    W = W[:, perm]
                    H = H[perm][:, perm]
                except torch.cuda.OutOfMemoryError:
                    verbose(f"OOM during GPU permutation, falling back to CPU")
                    self._report_oom('perm_gpu', N * N)
                    maybe_empty_cache(force=True)
                    H = H.cpu()[perm.cpu()][:, perm.cpu()]
                    W = W.cpu()[:, perm.cpu()]
        
        # 5. Hessian Inversion (Cholesky)
        # Use CPU for inversion if N is large to save VRAM
        # Use dynamic threshold from adaptive manager if available
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
        # Use dynamic threshold from adaptive manager if available
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
            # Use double precision for Hessian inversion to ensure consistency across devices
            try:
                H_inv_work_d = H_inv_work.double()
                H_inv_chol = torch.linalg.cholesky(H_inv_work_d)
                H_inv = torch.cholesky_inverse(H_inv_chol)
                H_inv = torch.linalg.cholesky(H_inv, upper=True)
                Hinv = H_inv.to(ldlq_device, dtype=torch.float32)  # Keep on CPU for large low_memory cases
                del H_inv_work_d, H_inv_chol, H_inv
            except torch.cuda.OutOfMemoryError:
                verbose(f"      - OOM during Hessian inversion on {hessian_device}, falling back to CPU")
                self._report_oom('hessian_inversion', N * N)
                maybe_empty_cache(force=True)
                # Move to CPU and retry
                H_inv_work_cpu = H_inv_work.cpu().double()
                H_inv_chol = torch.linalg.cholesky(H_inv_work_cpu)
                H_inv = torch.cholesky_inverse(H_inv_chol)
                H_inv = torch.linalg.cholesky(H_inv, upper=True)
                Hinv = H_inv.to(ldlq_device, dtype=torch.float32)
                del H_inv_work_cpu, H_inv_chol, H_inv
                verbose(f"      - Hessian inversion completed on CPU")
        except RuntimeError as e:
            verbose(f"      - QuIP Hessian inversion failed: {e}. Falling back to identity.")
            try:
                Hinv = torch.eye(N, device=ldlq_device, dtype=torch.float32)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      - OOM during identity creation on {ldlq_device}, falling back to CPU")
                Hinv = torch.eye(N, device='cpu', dtype=torch.float32)
        
        del H_inv_work
        maybe_empty_cache()

        # 6. LDLQ Quantization Loop (similar to GPTQ)
        # Compute per-channel (row-wise) scaling for the transformed weights
        
        # QuIP Paper Improvement: Frobenius Norm Scaling
        # Frobenius scaling is incompatible with Hadamard transforms.
        # Frobenius uses a single global scale, but Hadamard creates non-uniform
        # row distributions that require per-row scaling.
        if self.use_frobenius_scaling and self.use_hadamard:
            verbose("WARNING: Frobenius scaling is incompatible with Hadamard transforms. "
                    "Using max-based per-row scaling instead.")
            effective_use_frobenius = False
        else:
            effective_use_frobenius = self.use_frobenius_scaling
        
        if effective_use_frobenius:
            # Algorithm 1, Line 6: s ← ρ · ||W||_F / √(mn)
            # This uses the spectrum instead of max absolute value
            frob_norm = torch.norm(W, p='fro')  # Frobenius norm
            M_w, N_w = W.shape
            # Per-row scale: each row gets the same scale based on overall Frobenius norm
            # But we divide by sqrt(N) to get per-row contribution
            scale_value = self.frobenius_rho * frob_norm / (M_w * (N_w ** 0.5))
            scale = torch.full((M_w, 1), scale_value, device=W.device, dtype=W.dtype)
            verbose(f"Using Frobenius norm scaling: scale={scale_value:.6f}")
        else:
            # Original: max-based scaling (compatible with Hadamard)
            # Check if outlier-aware scaling is enabled
            from ..config.optimization_config import get_optimization_config
            opt_config = get_optimization_config()
            
            # Check if outlier-aware scaling is enabled and percentile is less than 1.0
            # Use a local variable for percentile to ensure we respect the config
            percentile = opt_config.outlier_percentile
            
            if opt_config.enable_outlier_aware_scaling and percentile < 1.0:
                # Use percentile to ignore extreme outliers for better quantization
                # ALWAYS use CPU for quantile to ensure consistency across devices
                # torch.quantile on CUDA can have slight numerical differences from CPU
                verbose(f"Using outlier-aware scaling: {percentile*100:.4f}th percentile (CPU-based for consistency)")
                try:
                    W_cpu = W.cpu()
                    abs_max_per_row = W_cpu.abs().quantile(percentile, dim=1, keepdim=True).to(W.device)
                    del W_cpu
                except Exception as e:
                    verbose(f"CPU quantile failed: {e}, falling back to amax scaling")
                    abs_max_per_row = W.abs().amax(dim=1, keepdim=True)
            else:
                # Standard max-based scaling
                abs_max_per_row = W.abs().amax(dim=1, keepdim=True)
            
            scale = (abs_max_per_row / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        
        # Move W to ldlq_device if needed (for CPU-based LDLQ in low_memory mode)
        if ldlq_device != self.device:
            W = W.to(ldlq_device)
            scale = scale.to(ldlq_device)
        
        # Keep target for learned rounding or greedy coordinate descent if enabled
        W_target = W.clone() if (self.use_learned_rounding or self.use_greedy_coordinate_descent) else None
        
        # Try LDLQ on current device, fallback to CPU on OOM
        try:
            # Ensure buffers are on the correct device
            self._ensure_buffers(M, N, self.block_size, ldlq_device)
            
            # If buffers fell back to CPU, move inputs to CPU too
            if self._q_buffer.device.type == 'cpu' and ldlq_device != 'cpu':
                verbose("Buffers fell back to CPU, moving inputs to CPU")
                W = W.cpu()
                Hinv = Hinv.cpu()
                scale = scale.cpu()
                ldlq_device = 'cpu'

            # Run LDLQ (possibly with multiple iterations)
            Q_prime = self._ldlq_with_iterations(W, Hinv, scale, iterations=self.ldlq_iterations)
        except torch.cuda.OutOfMemoryError:
            verbose(f"OOM during LDLQ on {ldlq_device}, forcing CPU fallback")
            self._report_oom('ldlq_main', M * N)
            maybe_empty_cache(force=True)
            # Move everything to CPU and retry
            W_cpu = W.cpu()
            Hinv_cpu = Hinv.cpu()
            scale_cpu = scale.cpu()
            # Clear buffers and reallocate on CPU
            self._ensure_buffers(M, N, self.block_size, 'cpu')
            Q_prime = self._ldlq_with_iterations(W_cpu, Hinv_cpu, scale_cpu, iterations=self.ldlq_iterations)
            verbose(f"LDLQ completed on CPU")
        
        # Free Hinv as it's no longer needed after LDLQ
        del Hinv
        maybe_empty_cache()
        
        # Move Q_prime back to main device if LDLQ ran on CPU
        if Q_prime.device != self.device:
            try:
                Q_prime = Q_prime.to(self.device)
                if scale.device != self.device:
                    scale = scale.to(self.device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"Cannot move result to GPU, keeping on CPU")
        
        # Optional: Learned Rounding Post-Optimization
        if self.use_learned_rounding and W_target is not None:
            verbose("    Refining QuIP with learned rounding...")
            Q_prime = self._refine_with_learned_rounding(Q_prime, W_target, scale)
        
        # QuIP Paper Improvement: Greedy Coordinate Descent
        if self.use_greedy_coordinate_descent and W_target is not None:
            verbose("    Refining QuIP with greedy coordinate descent...")
            # Need H on the same device as Q_prime for the coordinate descent
            H_cd = H if H.device == Q_prime.device else H.to(Q_prime.device)
            Q_prime = self._refine_with_greedy_coordinate_descent(
                Q_prime, W_target, H_cd, scale, num_iterations=self.greedy_cd_iterations
            )
            if H_cd is not H: del H_cd

        # Free refinement targets as they are no longer needed
        if W_target is not None: del W_target
        if H is not None: del H
        maybe_empty_cache()

        # 7. Compute dequantized_weight in original space for quality reporting
        
        # 8. Restore original order if ActOrder was used
        if self.actorder and perm is not None:
            try:
                inv_perm = torch.argsort(perm).to(Q_prime.device)
                Q_prime_ordered = Q_prime[:, inv_perm].contiguous()
                # Free original Q_prime immediately to save memory
                del Q_prime
            except torch.cuda.OutOfMemoryError:
                verbose(f"OOM during ActOrder restoration on {Q_prime.device}, falling back to CPU")
                self._report_oom('actorder_restore', Q_prime.numel())
                maybe_empty_cache(force=True)
                # Move to CPU and perform reordering
                Q_prime_cpu = Q_prime.cpu()
                inv_perm_cpu = torch.argsort(perm).cpu()
                Q_prime_ordered = Q_prime_cpu[:, inv_perm_cpu].contiguous()
                del Q_prime
                del Q_prime_cpu
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
            debug(f"      Hadamard: Large tensor on CPU, processing on CPU (size={tensor_size})")
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
            debug(f"      Hadamard: Using chunked dequantization (size={tensor_size}, chunk_size={chunk_size})")
            
            try:
                W_dequant_transformed = torch.empty_like(Q_prime_ordered, dtype=torch.float32, device=calc_device)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      OOM fallback: GPU OOM allocating output tensor, falling back to CPU")
                self._report_oom('dequant_alloc', tensor_size)
                maybe_empty_cache(force=True)
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
                    # Periodic cleanup (conditional, less frequent)
                    if calc_device == "cuda" and row_start % (chunk_size * 16) == 0:
                        maybe_empty_cache(pressure_threshold=0.92)
                except torch.cuda.OutOfMemoryError:
                    verbose(f"      OOM recovery: GPU OOM at row {row_start}, switching to CPU for remaining chunks")
                    self._report_oom('dequant_chunk', chunk_size * Q_prime_ordered.shape[1])
                    maybe_empty_cache(force=True)
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
                verbose(f"      OOM fallback: GPU OOM during dequantization, falling back to CPU")
                # Report OOM to adaptive manager for learning
                self._report_oom('hadamard', Q_prime_ordered.numel())
                import gc
                maybe_empty_cache(force=True)
                gc.collect()
                # Process on CPU
                Q_cpu = Q_prime_ordered.cpu()
                scale_cpu = scale.cpu()
                W_dequant_transformed = Q_cpu.to(torch.float32) * scale_cpu
                calc_device = "cpu"
        
        if s_u is not None and s_v is not None:
            # Undo permutation BEFORE inverse Hadamard (correct inverse order)
            # Forward pass: Hadamard -> Permute, so Inverse must be: Unpermute -> InvHadamard
            W_undo = W_dequant_transformed
            if rand_perm is not None:
                verbose("Undoing random permutation")
                try:
                    W_undo = self._undo_permutation(W_undo, rand_perm.to(W_undo.device))
                except torch.cuda.OutOfMemoryError:
                    verbose(f"      OOM fallback: GPU OOM during undo permutation, falling back to CPU")
                    self._report_oom('undo_permutation', W_undo.numel())
                    maybe_empty_cache(force=True)
                    # Move to CPU and retry
                    calc_device = "cpu"
                    W_undo_cpu = W_undo.cpu()
                    rand_perm_cpu = rand_perm.cpu()
                    W_undo = self._undo_permutation(W_undo_cpu, rand_perm_cpu)
            
            # Inverse transform using working version's approach:
            # 1. Apply Hadamard to rows
            # 2. Apply Hadamard to columns (via transpose)
            # 3. Apply signs AFTER Hadamard
            try:
                W_undo = fast_hadamard_transform(W_undo)
                W_undo = fast_hadamard_transform(W_undo.t()).t()
                W_undo = W_undo * s_u.to(calc_device).view(-1, 1) * s_v.to(calc_device).view(1, -1)
            except torch.cuda.OutOfMemoryError:
                verbose(f"      OOM fallback: GPU OOM during Hadamard inverse transform, falling back to CPU")
                self._report_oom('hadamard_inverse', W_undo.numel())
                maybe_empty_cache(force=True)
                # Move to CPU and retry
                calc_device = "cpu"
                W_cpu = W_undo.cpu()
                s_u_cpu = s_u.cpu()
                s_v_cpu = s_v.cpu()
                W_undo = fast_hadamard_transform(W_cpu)
                W_undo = fast_hadamard_transform(W_undo.t()).t()
                W_undo = W_undo * s_u_cpu.view(-1, 1) * s_v_cpu.view(1, -1)
            
            # Slice to original dimensions (after permutation undo, before rescaling undo)
            dequantized_weight = W_undo[:M_orig, :N_orig].contiguous()
            
            # Undo diagonal rescaling AFTER slicing
            # Rescaling was applied BEFORE padding (to original dims), so undo must apply
            # to sliced tensor (original dims)
            if D_tilde is not None:
                verbose("Undoing diagonal rescaling")
                dequantized_weight = self._undo_diagonal_rescaling(dequantized_weight, D_tilde.to(dequantized_weight.device))
        else:
            try:
                W_undo = U.t().to(calc_device) @ W_dequant_transformed @ V.to(calc_device)
                # For non-Hadamard path: undo permutation before slicing, then slice, then undo rescaling
                if rand_perm is not None:
                    verbose("Undoing random permutation")
                    try:
                        W_undo = self._undo_permutation(W_undo, rand_perm.to(W_undo.device))
                    except torch.cuda.OutOfMemoryError:
                        verbose(f"      OOM fallback: GPU OOM during undo permutation (non-Hadamard), falling back to CPU")
                        self._report_oom('undo_permutation', W_undo.numel())
                        torch.cuda.empty_cache()
                        calc_device = "cpu"
                        W_undo = W_undo.cpu()
                        rand_perm_cpu = rand_perm.cpu()
                        W_undo = self._undo_permutation(W_undo, rand_perm_cpu)
                # Slice to original dimensions
                dequantized_weight = W_undo[:M_orig, :N_orig].contiguous()
                # Undo rescaling after slicing (rescaling was applied to original dims)
                if D_tilde is not None:
                    verbose("Undoing diagonal rescaling")
                    dequantized_weight = self._undo_diagonal_rescaling(dequantized_weight, D_tilde.to(dequantized_weight.device))
            except torch.cuda.OutOfMemoryError:
                verbose(f"      OOM fallback: GPU OOM during orthogonal inverse transform, falling back to CPU")
                self._report_oom('ortho_inverse', W_dequant_transformed.numel())
                maybe_empty_cache(force=True)
                calc_device = "cpu"
                W_undo = U.t().cpu() @ W_dequant_transformed.cpu() @ V.cpu()
                # For non-Hadamard path: undo permutation before slicing, then slice, then undo rescaling
                if rand_perm is not None:
                    verbose("Undoing random permutation")
                    W_undo = self._undo_permutation(W_undo, rand_perm_cpu if 'rand_perm_cpu' in locals() else rand_perm.cpu())
                # Slice to original dimensions
                dequantized_weight = W_undo[:M_orig, :N_orig].contiguous()
                # Undo rescaling after slicing (rescaling was applied to original dims)
                if D_tilde is not None:
                    verbose("Undoing diagonal rescaling")
                    dequantized_weight = self._undo_diagonal_rescaling(dequantized_weight, D_tilde.to(dequantized_weight.device))

        # 10. Undo SmoothQuant transformation if it was applied
        if smooth_factors is not None:
            # Inverse: W = W_smooth / s (from forward: W_smooth = W * s)
            dequantized_weight = dequantized_weight / smooth_factors.to(calc_device).unsqueeze(0)

        if self.store_transformed:
            # Use per-row scale from LDLQ quantization for Hadamard-QuIP
            # The weight was quantized with per-row scales, so we need to return them
            final_scale = scale
            
            # Undo the random permutation (rand_perm) for the stored weights
            if rand_perm is not None:
                verbose(f"      [DEBUG] Undoing random permutation for stored transformed weights (dim={N})")
                # Step 8 only undid ActOrder. If we don't undo rand_perm, the stored weights
                # will be in a randomly permuted order that the inference engine doesn't know.
                
                # FIX: Unpermute at padded width (the width rand_perm was created for), then slice.
                # This ensures inv_perm indices are valid and prevents constant-color corruption.
                N_perm = rand_perm.numel()  # Padded size the permutation was created for
                if Q_prime_ordered.shape[1] < N_perm:
                    # Right-pad Q_prime_ordered back to N_perm columns (zeros are correct padding)
                    padding_cols = N_perm - Q_prime_ordered.shape[1]
                    Q_prime_ordered = torch.nn.functional.pad(
                        Q_prime_ordered, (0, padding_cols), mode='constant', value=0
                    )
                    verbose(f"      [DEBUG] Padded Q_prime_ordered from {Q_prime_ordered.shape[1] - padding_cols} to {N_perm} columns for unpermutation")
                
                Q_prime_ordered = self._undo_permutation(Q_prime_ordered, rand_perm.to(Q_prime_ordered.device))
                verbose(f"      [DEBUG] Random permutation undid. Shape: {Q_prime_ordered.shape}, First 5 elements of row 0: {Q_prime_ordered[0, :5].tolist()}")
                
                # NOTE: Do NOT slice here. Keep padded to match Hadamard metadata sizes
                # (hadamard_size_in/out computed below). The inference engine expects
                # weights at padded dimensions for correct Hadamard-QuIP processing.
            
            # Compute padded dimensions for Hadamard-QuIP metadata
            if self.use_hadamard:
                from ..utils.hadamard import next_power_of_two
                hadamard_size_out = next_power_of_two(M_orig)
                hadamard_size_in = next_power_of_two(N_orig)
            else:
                hadamard_size_out = 0
                hadamard_size_in = 0
            
            return (
                Q_prime_ordered.contiguous(),
                final_scale.to(SCALE_DTYPE),
                dequantized_weight.contiguous(),
                smooth_factors if smooth_factors is not None else None,
                # Keep sign vectors at padded dimensions for Hadamard-QuIP!
                s_u if s_u is not None else None,
                s_v if s_v is not None else None,
                # Hadamard-QuIP metadata (only when using Hadamard)
                True if self.use_hadamard else False,  # hadamard_quip flag
                hadamard_size_out,
                hadamard_size_in,
            )
        
        # Re-quantize to original space for compatibility with standard inference kernels
        if self.device == "cuda":
            maybe_empty_cache(pressure_threshold=0.90)
        
        M_total, N_total = dequantized_weight.shape
        
        # Choose re-quantization scheme: "tensor" (default) or "block"
        use_blockwise = (self.requant_scheme == "block") and (N_total % self.block_size == 0)
        
        if use_blockwise:
            # Block-wise scaling for re-quantization to preserve precision (critical for LoRA)
            verbose(f"      Re-quantizing to original space using block-wise scaling (bs={self.block_size})")
            W_reshaped = dequantized_weight.view(M_total, N_total // self.block_size, self.block_size)
            abs_max_blocks = W_reshaped.abs().amax(dim=2, keepdim=True)
            final_scale = (abs_max_blocks / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
            
            # Quantize using block-wise scales
            Q_final_int8 = torch.round(W_reshaped / final_scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8)
            Q_final_int8 = Q_final_int8.view(M_total, N_total)
            
            verbose(f"      [DEBUG] Final scale shape: {final_scale.shape}")
        else:
            # Tensor-wise scaling - can use per-row or global scale
            if self.requant_scheme == "block" and N_total % self.block_size != 0:
                verbose(f"      Re-quantizing to original space using tensor-wise scaling (N={N_total} not divisible by {self.block_size})")
            else:
                verbose(f"      Re-quantizing to original space using tensor-wise scaling")
            
            # Use per-row scales if requested (better precision, matches LDLQ quantization)
            if self.requant_tensor_use_per_row_scale:
                verbose(f"      Using per-row scales for tensor-wise requantization")
                abs_max_per_row = dequantized_weight.abs().amax(dim=1, keepdim=True)
                final_scale = (abs_max_per_row / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
            else:
                # Simple single global scale
                verbose(f"      Using global scale for tensor-wise requantization")
                final_abs_max = dequantized_weight.abs().max()
                final_scale = (final_abs_max / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
            
            # Use chunked processing for large tensors to avoid OOM
            tensor_size = dequantized_weight.numel()
            if tensor_size > 33_000_000 and dequantized_weight.device.type == "cuda":
                chunk_rows = 2048
                Q_final_int8 = torch.empty_like(dequantized_weight, dtype=torch.int8, device=dequantized_weight.device)
                for row_start in range(0, M_total, chunk_rows):
                    row_end = min(row_start + chunk_rows, M_total)
                    chunk = dequantized_weight[row_start:row_end]
                    # Handle per-row scales (shape [M, 1]) vs global scale (scalar)
                    if final_scale.dim() >= 1 and final_scale.shape[0] == M_total:
                        scale_chunk = final_scale[row_start:row_end]
                    else:
                        scale_chunk = final_scale
                    Q_final_int8[row_start:row_end] = torch.round(chunk / scale_chunk).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8)
            else:
                try:
                    Q_final_int8 = torch.round(dequantized_weight / final_scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8)
                except torch.cuda.OutOfMemoryError:
                    self._report_oom('requant', dequantized_weight.numel())
                    maybe_empty_cache(force=True)
                    Q_final_int8 = torch.round(dequantized_weight.cpu() / final_scale.cpu()).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8).to(self.device)
        
        # Return with Hadamard-QuIP metadata (flag as False since we're not storing transformed)
        return (
            Q_final_int8.contiguous(),
            final_scale.to(SCALE_DTYPE),
            dequantized_weight.contiguous(),
            smooth_factors if smooth_factors is not None else None,
            None, # s_u not needed for standard storage
            None, # s_v not needed for standard storage
            False, # hadamard_quip flag (False for standard storage)
            0,     # hadamard_size_out
            0,     # hadamard_size_in
        )

