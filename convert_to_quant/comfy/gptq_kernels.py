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
        scale_ptr,      # Quantization scale (scalar or vector [M, 1])
        M,
        BLOCK_SIZE: tl.constexpr,
        stride_wm, stride_wn,
        stride_qm, stride_qn,
        stride_em, stride_en,
        stride_hm, stride_hn,
        stride_sm,      # Stride for scale (0 for scalar)
        IS_SCALAR: tl.constexpr,
        DAMPING: tl.constexpr = 1e-12
    ):
        # Each program handles one row
        pid_m = tl.program_id(0)
        
        if pid_m >= M:
            return

        # Load scale for this row
        if IS_SCALAR:
            scale = tl.load(scale_ptr)
        else:
            scale = tl.load(scale_ptr + pid_m * stride_sm)
        
        # Ensure scale is not zero to avoid NaN
        scale = tl.maximum(tl.abs(scale), DAMPING)

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
            q = tl.maximum(tl.minimum(q, 127.0), -127.0)
            
            # Store quantized value
            tl.store(Q_ptr + pid_m * stride_qm + j * stride_qn, q.to(tl.int8))
            
            # Load Hinv diagonal with damping for stability
            d = tl.load(Hinv_ptr + j * stride_hm + j * stride_hn)
            d = tl.maximum(tl.abs(d), DAMPING)
            
            # Use explicit FP32 for error calculation to prevent drift
            err = (w.to(tl.float32) - q.to(tl.float32) * scale.to(tl.float32)) / d.to(tl.float32)
            
            # Store error
            tl.store(Err_ptr + pid_m * stride_em + j * stride_en, err)
            
            # Update remaining columns in this row
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
        scale: float or Tensor [M, 1]
        Returns: (Q, Err)
        """
        M, BLOCK_SIZE = W.shape
        Q = torch.empty((M, BLOCK_SIZE), device=W.device, dtype=torch.int8)
        Err = torch.empty((M, BLOCK_SIZE), device=W.device, dtype=torch.float32)
        
        # Ensure W is contiguous for better memory access patterns
        if not W.is_contiguous():
            W = W.contiguous()
        
        # Handle scale (scalar or vector)
        is_scalar = scale.numel() == 1
        if is_scalar:
            # If it's a tensor with 1 element, get the value
            if isinstance(scale, torch.Tensor):
                scale_val = scale.data
            else:
                scale_val = torch.tensor([float(scale)], device=W.device)
            stride_sm = 0
        else:
            scale_val = scale
            stride_sm = scale.stride(0)
            
        grid = (M,)
        gptq_quant_block_kernel[grid](
            W, Q, Err, Hinv, scale_val,
            M, BLOCK_SIZE,
            W.stride(0), W.stride(1),
            Q.stride(0), Q.stride(1),
            Err.stride(0), Err.stride(1),
            Hinv.stride(0), Hinv.stride(1),
            stride_sm,
            IS_SCALAR=is_scalar,
        )
        
        return Q, Err
else:
    def triton_gptq_quant_block(W, Hinv, scale):
        raise ImportError("Triton is not installed. Cannot use gptq_turbo.")
