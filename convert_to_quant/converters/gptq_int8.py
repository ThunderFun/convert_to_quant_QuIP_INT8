import torch
from torch import Tensor
from typing import Tuple, Optional
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE, SCALE_DTYPE, TARGET_INT8_DTYPE, INT8_SYMMETRIC_MAX
from ..utils.logging import verbose, debug
from ..comfy.gptq_kernels import triton_gptq_quant_block, HAS_TRITON
 
class GPTQInt8Converter:
    """
    GPTQ-style sequential INT8 quantization with error compensation.
    
    Reference: https://arxiv.org/abs/2210.17323
    """
    def __init__(
        self,
        block_size: Optional[int] = 128,
        percdamp: float = 0.01,
        device: str = "cuda",
        actorder: bool = False,
        use_triton: bool = False,
        lazy_updates: bool = True,
        low_memory: bool = False,
        streaming: bool = False,
    ):
        self.block_size = block_size if block_size is not None else 128
        self.percdamp = percdamp  # Hessian damping factor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.actorder = actorder
        self.use_triton = use_triton
        self.lazy_updates = lazy_updates
        self.low_memory = low_memory
        self.streaming = streaming
        
        # Pre-allocated buffers for memory optimization
        self._q_buffer = None
        self._err_buffer = None

    def _ensure_buffers(self, M, block_size):
        """Ensure reusable buffers are allocated and have correct size."""
        if self._q_buffer is None or self._q_buffer.shape != (M, block_size):
            self._q_buffer = torch.empty(M, block_size, dtype=torch.int8, device=self.device)
            self._err_buffer = torch.empty(M, block_size, dtype=torch.float32, device=self.device)

    def convert(self, weight: Tensor, H: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize weight using GPTQ-style sequential optimization.
        
        Args:
            weight: Weight tensor (out_features, in_features)
            H: Hessian approximation (X.T @ X), shape (in_features, in_features)
            
        Returns:
            Tuple of (quantized_weight, scale, dequantized_weight)
        """
        M, N = weight.shape
        
        # Use CPU for Hessian inversion if it's large to save GPU memory
        hessian_device = "cpu" if N > 4096 else self.device
        
        W = weight.clone().float().to(self.device)
        
        if H is None:
            # Use identity approximated Hessian if not provided
            H = torch.eye(N, device=hessian_device)
        else:
            # Clone to avoid modifying caller's tensor (damping is applied in-place below)
            H = H.to(hessian_device).float().clone()
            
        # Damping
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(N, device=hessian_device)
        H[diag, diag] += damp
        
        # ActOrder: Sort columns by Hessian diagonal descending
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
        
        # Cholesky decomposition for inverse
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H.to(self.device) # Move back to GPU for the main loop
        except RuntimeError as e:
            verbose(f"      - Hessian inversion failed on {hessian_device}: {e}. Falling back to identity.")
            Hinv = torch.eye(N, device=self.device)
        
        Q = torch.zeros_like(W, dtype=torch.int8)
        
        # We'll use a single scale for the whole tensor for now, 
        # or we could do per-column or per-block. 
        # Standard GPTQ often uses per-group (block) scaling.
        # For simplicity and compatibility with our INT8 format, 
        # let's compute a global scale first or per-block.
        
        # Let's do per-tensor scale for now to match the sketch, 
        # but we can adapt it to block-wise.
        abs_max = W.abs().max()
        scale = (abs_max / INT8_SYMMETRIC_MAX).clamp(min=1e-12)
        
        pbar = tqdm(range(0, N, self.block_size), desc="    GPTQ Quantizing", leave=False)
        for i1 in pbar:
            i2 = min(i1 + self.block_size, N)
            count = i2 - i1
            
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros((M, count), dtype=torch.int8, device=self.device)
            Err_block = torch.zeros((M, count), dtype=torch.float32, device=self.device)
            Hinv_block = Hinv[i1:i2, i1:i2]
            
            if self.use_triton and HAS_TRITON:
                # Triton-accelerated block processing
                Q_block, Err_block = triton_gptq_quant_block(W_block, Hinv_block, scale)
            elif self.lazy_updates:
                # Vectorized column processing with lazy updates
                # Increased batch size for better GPU utilization
                BATCH = 64
                for j in range(0, count, BATCH):
                    end_j = min(j + BATCH, count)
                    
                    # Quantize batch
                    w_batch = W_block[:, j:end_j]
                    q_batch = torch.round(w_batch / scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
                    Q_block[:, j:end_j] = q_batch.to(torch.int8)
                    
                    # Compute errors
                    d_batch = torch.diag(Hinv_block[j:end_j, j:end_j])
                    err_batch = (w_batch - q_batch * scale) / d_batch.unsqueeze(0)
                    Err_block[:, j:end_j] = err_batch
                    
                    # Batched update for remaining columns in block
                    if end_j < count:
                        update_factors = Hinv_block[j:end_j, end_j:]
                        # BF16 optimization for error updates
                        from ..constants import should_use_bf16_for_op
                        use_bf16 = should_use_bf16_for_op(err_batch.numel() * update_factors.numel(), "matmul")
                        if use_bf16:
                            device_type = 'cuda' if err_batch.is_cuda else 'cpu'
                            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                                update = err_batch @ update_factors
                            W_block[:, end_j:] -= update.float()
                        else:
                            W_block[:, end_j:] -= err_batch @ update_factors
            else:
                # Original sequential processing
                for j in range(count):
                    w = W_block[:, j]
                    d = Hinv_block[j, j]
                    
                    # Quantize
                    q = torch.round(w / scale).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
                    Q_block[:, j] = q.to(torch.int8)
                    
                    # Error
                    err = (w - q * scale) / d
                    
                    # Compensate using rank-1 update
                    if j < count - 1:
                        W_block[:, j+1:].addr_(err, Hinv_block[j, j+1:], alpha=-1)
                    
                    Err_block[:, j] = err
                
            Q[:, i1:i2] = Q_block
            if i2 < N:
                # BF16 optimization for block error updates
                from ..constants import should_use_bf16_for_op
                use_bf16 = should_use_bf16_for_op(Err_block.numel() * Hinv[i1:i2, i2:].numel(), "matmul")
                if use_bf16:
                    device_type = 'cuda' if Err_block.is_cuda else 'cpu'
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        update = Err_block @ Hinv[i1:i2, i2:]
                    W[:, i2:] -= update.float()
                else:
                    W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]
            
            # Periodic cleanup (only in low_memory mode to save time)
            if self.low_memory and i1 % (self.block_size * 5) == 0:
                torch.cuda.empty_cache()
        
        # Restore original order if ActOrder was used
        if self.actorder and perm is not None:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm].contiguous()
            
        dequantized_weight = Q.to(COMPUTE_DTYPE) * scale
        
        return Q.cpu(), scale.cpu().to(SCALE_DTYPE), dequantized_weight.cpu()
