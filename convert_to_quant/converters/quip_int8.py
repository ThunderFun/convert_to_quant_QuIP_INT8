import torch
from torch import Tensor
from typing import Tuple, Optional
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE, TARGET_INT8_DTYPE, INT8_SYMMETRIC_MAX
from ..utils.logging import verbose, debug
from ..utils.hadamard import (
    is_power_of_two,
    fast_hadamard_transform,
    random_orthogonal_matrix,
)


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
    ):
        self.block_size = block_size
        self.percdamp = percdamp
        self.device = device if torch.cuda.is_available() else "cpu"
        self.actorder = actorder
        self.use_hadamard = use_hadamard
        self.seed = seed
        self.use_triton = use_triton
        self.lazy_updates = lazy_updates
        self.use_learned_rounding = use_learned_rounding
        self.ldlq_iterations = ldlq_iterations
        self.store_transformed = store_transformed
        
        # Cache for orthogonal matrices
        self._ortho_cache = {}
        self.low_memory = False # Will be set by quantization.py

        # Pre-allocated buffers for memory optimization
        self._W_buffer = None
        self._H_buffer = None
        self._err_buffer = None
        self._q_buffer = None

    def _ensure_buffers(self, M: int, N: int, block_size: int) -> None:
        """Allocate or reuse buffers for quantization loop."""
        if self._q_buffer is None or self._q_buffer.shape[0] != M or self._q_buffer.shape[1] != block_size:
            self._q_buffer = torch.empty(M, block_size, dtype=torch.int8, device=self.device)
            self._err_buffer = torch.empty(M, block_size, dtype=torch.float32, device=self.device)
            
        if self._W_buffer is None or self._W_buffer.shape != (M, N):
            self._W_buffer = torch.empty(M, N, dtype=torch.float32, device=self.device)

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
        if self.use_hadamard and is_power_of_two(n):
            from ..utils.hadamard import hadamard_matrix
            # Use CPU for matrix generation if n is large to save VRAM
            gen_device = "cpu" if (n > 4096 and self.low_memory) else self.device
            Q = hadamard_matrix(n).to(gen_device)
        else:
            gen_device = "cpu" if (n > 4096 and self.low_memory) else self.device
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
            
            # Use CPU for large matrices if in low_memory mode
            calc_device = "cpu" if (self.low_memory or N_pad > 4096) else self.device
            
            # Set seed once before generating all random signs
            if self.seed is not None:
                torch.manual_seed(self.seed)
            
            # Compute signs (small tensors, OK to keep)
            s_u = self._get_random_signs(M_pad).to(calc_device)
            s_v = self._get_random_signs(N_pad).to(calc_device)
            
            # For W: pad -> transform (DO NOT slice - caller handles that)
            if M != M_pad or N != N_pad:
                import torch.nn.functional as F
                W_padded = F.pad(W.to(calc_device), (0, N_pad - N, 0, M_pad - M))
                W_padded.mul_(s_u.view(-1, 1))
                W_padded.mul_(s_v.view(1, -1))
                
                # Apply H to rows and cols
                W_prime = fast_hadamard_transform(W_padded.t(), inplace=True).t()
                W_prime = fast_hadamard_transform(W_prime, inplace=True)
                
                # DO NOT slice here - keep full padded tensor for correct inverse transform
                del W_padded
            else:
                W_prime = W.to(calc_device) * s_u.view(-1, 1) * s_v.view(1, -1)
                W_prime = fast_hadamard_transform(W_prime.t(), inplace=True).t()
                W_prime = fast_hadamard_transform(W_prime, inplace=True)

            # For H: pad -> transform (DO NOT slice)
            if N != N_pad:
                import torch.nn.functional as F
                H_padded = F.pad(H.to(calc_device), (0, N_pad - N, 0, N_pad - N))
                # For H, we only use s_v
                H_padded.mul_(s_v.view(-1, 1))
                H_padded.mul_(s_v.view(1, -1))
                
                H_prime = fast_hadamard_transform(H_padded.t(), inplace=True).t()
                H_prime = fast_hadamard_transform(H_prime, inplace=True)
                
                # DO NOT slice - keep full padded tensor
                del H_padded
            else:
                H_prime = H.to(calc_device) * s_v.view(-1, 1) * s_v.view(1, -1)
                H_prime = fast_hadamard_transform(H_prime.t(), inplace=True).t()
                H_prime = fast_hadamard_transform(H_prime, inplace=True)
            
            # Return full sign vectors (not truncated) for correct inverse transform
            return W_prime.to(self.device), H_prime.to(self.device), None, None, s_u.to(self.device), s_v.to(self.device)
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
        BATCH = 64
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
        best_delta = delta.detach().clone()
        
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
                best_delta = delta.detach().clone()
        
        return (Q_float + best_delta).round().clamp(-127, 127).to(torch.int8)

    def _single_ldlq_pass(self, W: Tensor, Hinv: Tensor, scale: Tensor) -> Tensor:
        """Single pass of LDLQ quantization."""
        M_work, N_work = W.shape
        Q_prime = torch.zeros_like(W, dtype=torch.int8)
        
        # We must clone W because it's modified in-place
        W_work = W.clone()
        
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
                W_work[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]
                
        return Q_prime

    def _ldlq_with_iterations(self, W: Tensor, Hinv: Tensor, scale: Tensor, iterations: int = 1) -> Tensor:
        """Multi-pass LDLQ for better error compensation."""
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
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Quantize weight using QuIP.
        """
        # Track ORIGINAL dimensions for final slicing
        M_orig, N_orig = weight.shape
        
        # Avoid unnecessary clones if already on device and float
        W = weight.to(self.device, dtype=torch.float32)
        
        # DIAGNOSTIC: Track Hessian source
        h_is_identity = H is None
        if H is None:
            H = torch.eye(N_orig, device=self.device, dtype=torch.float32)
            debug(f"      [DIAG] QuIP Hessian: IDENTITY (H=None passed)")
        else:
            H = H.to(self.device, dtype=torch.float32)
            # Compute diagnostics on the incoming H
            h_diag = torch.diag(H)
            h_trace = h_diag.sum().item()
            h_max_diag = h_diag.max().item()
            h_min_diag = h_diag[h_diag > 0].min().item() if (h_diag > 0).any() else 0
            h_cond_estimate = h_max_diag / max(h_min_diag, 1e-12)
            debug(f"      [DIAG] QuIP Hessian: COMPUTED (trace={h_trace:.2f}, diag_max={h_max_diag:.4f}, diag_min={h_min_diag:.6f}, cond~={h_cond_estimate:.1f})")
        
        debug(f"      [DIAG] QuIP seed={self.seed}, actorder={self.actorder}, hadamard={self.use_hadamard}")

        # NEW: Apply SmoothQuant if activation scales provided
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
        # NOTE: For non-power-of-2 with Hadamard, W and H are now PADDED dimensions!
        W, H, U, V, s_u, s_v = self._apply_incoherence(W, H)
        
        # Get actual working dimensions (may be padded)
        M, N = W.shape
        
        # Ensure buffers are ready with actual (potentially padded) dimensions
        self._ensure_buffers(M, N, self.block_size)
        
        # 3. Damping (same as GPTQ)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(N, device=self.device)
        H[diag, diag] += damp
        
        # 4. ActOrder (optional but recommended for QuIP)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
        
        # 5. Hessian Inversion (Cholesky)
        # Use CPU for inversion if N is large to save VRAM
        hessian_device = "cpu" if (N > 4096 or self.low_memory) else self.device
        H_inv_work = H.to(hessian_device)
        
        try:
            H_inv_chol = torch.linalg.cholesky(H_inv_work)
            H_inv = torch.cholesky_inverse(H_inv_chol)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
            Hinv = H_inv.to(self.device)
        except RuntimeError as e:
            verbose(f"      - QuIP Hessian inversion failed on {hessian_device}: {e}. Falling back to identity.")
            Hinv = torch.eye(N, device=self.device)

        # 6. LDLQ Quantization Loop (similar to GPTQ)
        # Keep target for learned rounding if enabled
        W_target = W.clone() if self.use_learned_rounding else None
        
        # Compute per-channel (row-wise) scaling for the transformed weights
        abs_max_per_row = W.abs().amax(dim=1, keepdim=True)
        scale = (abs_max_per_row / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        
        # Run LDLQ (possibly with multiple iterations)
        Q_prime = self._ldlq_with_iterations(W, Hinv, scale, iterations=self.ldlq_iterations)
        
        # Optional: Learned Rounding Post-Optimization
        if self.use_learned_rounding and W_target is not None:
            verbose("    Refining QuIP with learned rounding...")
            Q_prime = self._refine_with_learned_rounding(Q_prime, W_target, scale)

        # 7. Handle store_transformed option
        # We'll compute dequantized_weight in original space for quality reporting
        # but return Q_prime if store_transformed is True.
        
        # 8. Restore original order if ActOrder was used
        if self.actorder and perm is not None:
            inv_perm = torch.argsort(perm)
            Q_prime_ordered = Q_prime[:, inv_perm].contiguous()
        else:
            Q_prime_ordered = Q_prime

        # 9. Undo Incoherence for dequantized weight (always needed for quality reporting)
        calc_device = "cpu" if self.low_memory else self.device
        W_dequant_transformed = Q_prime_ordered.to(torch.float32).to(calc_device) * scale.to(calc_device)
        
        if s_u is not None and s_v is not None:
            # Inverse transform: reverse order of forward (H rows → H cols → signs)
            # W_dequant_transformed is already at padded dimensions (M, N) = (M_pad, N_pad)
            # NO re-padding needed - we kept the full tensor throughout
            
            W_undo = fast_hadamard_transform(W_dequant_transformed, inplace=True)
            W_undo = fast_hadamard_transform(W_undo.t(), inplace=True).t()
            W_undo = W_undo * s_u.to(calc_device).view(-1, 1) * s_v.to(calc_device).view(1, -1)
            
            # NOW slice to original dimensions (this is the only place we slice)
            dequantized_weight = W_undo[:M_orig, :N_orig].contiguous()
        else:
            dequantized_weight = U.t().to(calc_device) @ W_dequant_transformed @ V.to(calc_device)
            dequantized_weight = dequantized_weight[:M_orig, :N_orig].contiguous()

        # 10. Undo SmoothQuant transformation if it was applied
        if smooth_factors is not None:
            # Inverse: W = W_smooth / s (from forward: W_smooth = W * s)
            dequantized_weight = dequantized_weight / smooth_factors.to(calc_device).unsqueeze(0)

        if self.store_transformed:
            return (
                Q_prime[:M_orig, :N_orig].cpu().contiguous(), 
                scale[:M_orig].cpu().to(SCALE_DTYPE), 
                dequantized_weight.cpu().contiguous(),
                smooth_factors.cpu() if smooth_factors is not None else None,
                s_u[:M_orig].cpu() if s_u is not None else None,
                s_v[:N_orig].cpu() if s_v is not None else None
            )
        
        # Re-quantize to original space for compatibility with standard inference kernels
        final_abs_max = dequantized_weight.abs().max()
        final_scale = (final_abs_max / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        Q_final_int8 = torch.round(dequantized_weight / final_scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8)
        
        return (
            Q_final_int8.cpu().contiguous(), 
            final_scale.cpu().to(SCALE_DTYPE), 
            dequantized_weight.cpu().contiguous(), 
            smooth_factors.cpu() if smooth_factors is not None else None,
            None, # s_u not needed for standard storage
            None  # s_v not needed for standard storage
        )
