import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def gptq_quant_block_kernel(
        W_ptr,          # Input weights [M, block_size]
        Q_ptr,          # Output quantized [M, block_size]  
        Err_ptr,        # Output errors [M, block_size]
        Hinv_ptr,       # Inverse Hessian block [block_size, block_size]
        scale,          # Quantization scale (scalar)
        M,
        BLOCK_SIZE: tl.constexpr,
        stride_wm, stride_wn,
        stride_qm, stride_qn,
        stride_em, stride_en,
        stride_hm, stride_hn,
    ):
        # Each program handles one row
        pid_m = tl.program_id(0)
        
        if pid_m >= M:
            return

        # Sequential processing within kernel for error compensation
        for j in range(BLOCK_SIZE):
            # Load weight for column j
            w_ptr = W_ptr + pid_m * stride_wm + j * stride_wn
            w = tl.load(w_ptr)
            
            # Quantize using banker's rounding (round-half-to-even) to match PyTorch
            q_raw = w / scale
            # Banker's rounding: round 0.5 to nearest even number
            q_floor = tl.floor(q_raw)
            frac = q_raw - q_floor
            # If fractional part is 0.5, round to nearest even
            # Otherwise use standard rounding (< 0.5 → floor, > 0.5 → ceil)
            floor_is_even = (q_floor % 2.0 == 0.0)
            use_floor = (frac < 0.5) | ((frac == 0.5) & floor_is_even)
            q = tl.where(use_floor, q_floor, q_floor + 1.0)
            
            # Use INT8_SYMMETRIC_MAX (127) for symmetric quantization range
            # Matching the main implementation in gptq_int8.py
            q = tl.maximum(tl.minimum(q, 127.0), -127.0)
            
            # Store quantized value
            tl.store(Q_ptr + pid_m * stride_qm + j * stride_qn, q.to(tl.int8))
            
            # Load Hinv diagonal
            d = tl.load(Hinv_ptr + j * stride_hm + j * stride_hn)
            err = (w - q * scale) / d
            
            # Store error
            tl.store(Err_ptr + pid_m * stride_em + j * stride_en, err)
            
            # Update remaining columns in this row
            # This is the sequential part that makes GPTQ hard to parallelize across columns
            for k in range(j + 1, BLOCK_SIZE):
                h_jk = tl.load(Hinv_ptr + j * stride_hm + k * stride_hn)
                w_k_ptr = W_ptr + pid_m * stride_wm + k * stride_wn
                w_k = tl.load(w_k_ptr)
                w_k -= err * h_jk
                tl.store(w_k_ptr, w_k)

    def triton_gptq_quant_block(W, Hinv, scale):
        """
        Launch Triton kernel for GPTQ block quantization.
        W: [M, block_size] (modified in-place)
        Hinv: [block_size, block_size]
        scale: float
        Returns: (Q, Err)
        """
        M, BLOCK_SIZE = W.shape
        Q = torch.empty((M, BLOCK_SIZE), device=W.device, dtype=torch.int8)
        Err = torch.empty((M, BLOCK_SIZE), device=W.device, dtype=torch.float32)
        
        # Ensure W is contiguous for better memory access patterns
        if not W.is_contiguous():
            W = W.contiguous()
        
        grid = (M,)
        gptq_quant_block_kernel[grid](
            W, Q, Err, Hinv, float(scale.data.item()),
            M, BLOCK_SIZE,
            W.stride(0), W.stride(1),
            Q.stride(0), Q.stride(1),
            Err.stride(0), Err.stride(1),
            Hinv.stride(0), Hinv.stride(1),
        )
        
        return Q, Err
else:
    def triton_gptq_quant_block(W, Hinv, scale):
        raise ImportError("Triton is not installed. Cannot use gptq_turbo.")
