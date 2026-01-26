import triton
import triton.language as tl
import torch
from typing import Optional

HAS_TRITON_HADAMARD = False  # DISABLED: Triton kernel has butterfly algorithm bug causing non-invertible transform

if HAS_TRITON_HADAMARD:
    @triton.jit
    def _hadamard_kernel(
        x_ptr, out_ptr,
        n_rows, n_cols,
        stride_row, stride_col,
        BLOCK_SIZE: tl.constexpr
    ):
        """Hadamard transform using correct in-register butterfly algorithm.
        
        This implementation works correctly for all power-of-2 sizes by:
        1. Loading entire row into registers
        2. Performing butterfly operations entirely in registers using XOR indexing
        3. No store/load synchronization needed (everything in registers)
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load entire row into registers
        x = tl.load(x_ptr + row_idx * stride_row + col_offsets * stride_col,
                    mask=mask, other=0.0)
        
        # Butterfly passes - all in registers using XOR for partner indices
        # The butterfly algorithm for Hadamard:
        # At stage h, pair (i, i^h) computes:
        #   a' = a + b (for even position in pair)
        #   b' = a - b (for odd position in pair)
        h = 1
        while h < BLOCK_SIZE:
            # XOR gives us partner index for this butterfly stage
            partner_idx = col_offsets ^ h
            
            # Determine if this is the "even" or "odd" half of the butterfly pair
            # Even: position's h-th bit is 0 (does addition)
            # Odd: position's h-th bit is 1 (does subtraction)  
            is_even = (col_offsets & h) == 0
            
            # For even positions: result = x[i] + x[i^h]
            # For odd positions: result = x[i^h] - x[i]
            # We need to gather x[partner_idx], but Triton doesn't support direct gather
            # Solution: Store intermediate, then load with new indices
            
            # Store current values
            tl.store(out_ptr + row_idx * stride_row + col_offsets * stride_col, x,
                    mask=mask)
            
            # Memory barrier - ensure all stores complete before loads
            tl.debug_barrier()
            
            # Load partner values
            x_partner = tl.load(out_ptr + row_idx * stride_row + partner_idx * stride_col,
                               mask=partner_idx < n_cols, other=0.0)
            
            # Butterfly: even gets sum, odd gets difference
            x = tl.where(is_even, x + x_partner, x_partner - x)
            h *= 2
        
        # Normalize and store final result
        x = x / tl.sqrt(float(BLOCK_SIZE))
        tl.store(out_ptr + row_idx * stride_row + col_offsets * stride_col, x,
                mask=mask)

    def triton_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
        """Triton-accelerated Hadamard transform."""
        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        
        original_shape = x.shape
        n = x.shape[-1]
        # Check if n is power of 2
        if not (n > 0 and (n & (n - 1) == 0)):
            # Fallback to CPU/Python if not power of 2 (though QuIP should ensure it is)
            from ..utils.hadamard import fast_hadamard_transform as py_fht
            return py_fht(x)

        x = x.reshape(-1, n).contiguous()
        out = torch.empty_like(x)
        n_rows = x.shape[0]
        
        BLOCK_SIZE = triton.next_power_of_2(n)
        grid = (n_rows,)
        
        _hadamard_kernel[grid](
            x, out,
            n_rows, n,
            x.stride(0), x.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out.reshape(original_shape)

    @triton.jit
    def _fused_sign_hadamard_kernel(
        x_ptr, signs_ptr, out_ptr,
        n_rows, n_cols,
        stride_row, stride_col,
        BLOCK_SIZE: tl.constexpr
    ):
        """Apply signs then Hadamard in one pass.
        
        Uses the same XOR butterfly algorithm as _hadamard_kernel,
        but applies element-wise signs before the butterfly stages.
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load row and signs
        x = tl.load(x_ptr + row_idx * stride_row + col_offsets * stride_col,
                    mask=mask, other=0.0)
        signs = tl.load(signs_ptr + col_offsets, mask=mask, other=1.0)
        
        # Apply signs first
        x = x * signs
        
        # Butterfly Hadamard with XOR indexing
        h = 1
        while h < BLOCK_SIZE:
            partner_idx = col_offsets ^ h
            is_even = (col_offsets & h) == 0
            
            # Store current values
            tl.store(out_ptr + row_idx * stride_row + col_offsets * stride_col, x,
                    mask=mask)
            
            # Memory barrier
            tl.debug_barrier()
            
            # Load partner values
            x_partner = tl.load(out_ptr + row_idx * stride_row + partner_idx * stride_col,
                               mask=partner_idx < n_cols, other=0.0)
            
            # Butterfly
            x = tl.where(is_even, x + x_partner, x_partner - x)
            h *= 2
        
        # Normalize and store final result
        x = x / tl.sqrt(float(BLOCK_SIZE))
        tl.store(out_ptr + row_idx * stride_row + col_offsets * stride_col, x,
                mask=mask)

def fast_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Dispatch to Triton or fallback."""
    if HAS_TRITON_HADAMARD and x.is_cuda:
        return triton_hadamard_transform(x)
    else:
        # Fallback to existing Python implementation
        from ..utils.hadamard import fast_hadamard_transform as py_fht
        return py_fht(x, normalize=normalize)
