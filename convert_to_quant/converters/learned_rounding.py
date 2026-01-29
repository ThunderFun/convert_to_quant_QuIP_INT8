"""
Learned rounding converter for INT8 quantization.

This module implements advanced quantization using learned adaptive rounding
with SVD-based optimization. Inherits from BaseLearnedConverter.
"""
import gc
import math
import torch
from typing import Tuple, Optional
from tqdm import tqdm
from torch.optim import AdamW, RAdam

from ..constants import (
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    INT8_SYMMETRIC_MAX,
)
from ..comfy.quant_ops import BlockWiseINT8Layout
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import verbose, debug, minimal
from .base_converter import BaseLearnedConverter
from .smoothquant import SmoothQuantPreprocessor

class LearnedRoundingConverter(BaseLearnedConverter):
    """
    Learned rounding converter for INT8 quantization.
    """

    def __init__(
        self,
        scaling_mode: str = "block",
        block_size: int = 128,
        target_format: str = "int8",
        smoothquant: bool = False,
        smoothquant_alpha: float = 0.5,
        low_memory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.target_format = target_format
        self.smoothquant = smoothquant
        self.smoothquant_alpha = smoothquant_alpha
        self.low_memory = low_memory

        if target_format == "int8" and scaling_mode not in ("tensor", "axis"):
            scaling_mode = "block"
        self.scaling_mode = scaling_mode
        self.target_dtype = TARGET_INT8_DTYPE if target_format == "int8" else torch.float16

        verbose(f"LearnedRoundingConverter initialized on device: {self.device}")
        verbose(f"  - Target format: {self.target_format}")
        verbose(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode == "block":
            verbose(f"    - Block size: {self.block_size}")

    def convert(
        self, W_orig: torch.Tensor, activation_scales: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if self.smoothquant and activation_scales is not None:
            preprocessor = SmoothQuantPreprocessor(alpha=self.smoothquant_alpha)
            s = preprocessor.compute_smoothing_factors(W_float32, activation_scales.to(self.device))
            W_float32 = preprocessor.apply_to_weight(W_float32, s)

        if torch.all(W_float32 == 0):
            qdata = torch.zeros_like(W_float32, dtype=self.target_dtype)
            if self.target_format == "int8" and self.scaling_mode == "block":
                scale = torch.ones(W_float32.shape[0] // self.block_size, W_float32.shape[1] // self.block_size, device=self.device, dtype=SCALE_DTYPE)
            elif self.scaling_mode == "axis":
                scale = torch.ones(W_float32.shape[0], device=self.device, dtype=SCALE_DTYPE)
            else:
                scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            return qdata, scale, torch.zeros_like(W_float32)

        if self.target_format == "fp16":
            qdata = W_float32.to(torch.float16)
            return qdata, torch.tensor(1.0, device=self.device, dtype=SCALE_DTYPE), qdata.to(COMPUTE_DTYPE)

        if self.scaling_mode == "axis":
            return self._convert_int8_axiswise(W_float32)
        elif self.scaling_mode == "tensor":
            return self._convert_int8_tensorwise(W_float32)
        else:
            return self._convert_int8(W_float32)

    def _convert_int8(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, N = W_float32.shape
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(f"INT8 block-wise requires dimensions divisible by {self.block_size}, got {M}x{N}")

        qdata, layout_params = BlockWiseINT8Layout.quantize(W_float32, block_size=self.block_size, is_weight=True)
        scale = layout_params["scale"]

        if not self.no_learned_rounding and self.num_iter > 0:
            qdata = self._optimize_int8_learned_rounding(W_float32, qdata, scale)

        dequantized = BlockWiseINT8Layout.dequantize(qdata, scale, self.block_size, is_weight=True, orig_dtype=COMPUTE_DTYPE)
        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized

    def _convert_int8_tensorwise(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (W_float32.abs().max().float() / 127.0).clamp(min=1e-30)
        if self.no_learned_rounding:
            qdata = W_float32.mul(1.0 / scale).round_().clamp_(-127.0, 127.0).to(torch.int8)
            return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), qdata.to(COMPUTE_DTYPE) * scale

        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        qdata_float = W_float32.mul(1.0 / scale).round_().clamp_(-127.0, 127.0)
        final_qdata_float = self._optimize_int8_tensorwise_original(W_float32, qdata_float, scale, U_k, Vh_k)
        final_qdata = final_qdata_float.round().clamp_(-127.0, 127.0).to(torch.int8)
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), final_qdata.to(COMPUTE_DTYPE) * scale

    def _convert_int8_axiswise(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (W_float32.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-30)
        if self.no_learned_rounding:
            qdata = W_float32.mul(1.0 / scale).round_().clamp_(-127.0, 127.0).to(torch.int8)
            return qdata, scale.squeeze(1).to(device=self.device, dtype=SCALE_DTYPE), qdata.to(COMPUTE_DTYPE) * scale

        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        qdata_float = W_float32.mul(1.0 / scale).round_().clamp_(-127.0, 127.0)
        final_qdata_float = self._optimize_int8_tensorwise_original(W_float32, qdata_float, scale, U_k, Vh_k)
        final_qdata = final_qdata_float.round().clamp_(-127.0, 127.0).to(torch.int8)
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata, scale.squeeze(1).to(device=self.device, dtype=SCALE_DTYPE), final_qdata.to(COMPUTE_DTYPE) * scale

    def _optimize_int8_tensorwise_original(self, W_float32, qdata_float, scale, U_k, Vh_k):
        q_refined = qdata_float.clone()
        best_loss, worse_loss_counter = float("inf"), 0
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 ({self.scaling_mode})", leave=False, dynamic_ncols=True)
        
        # BF16 optimization check for SVD-based loss calculation
        from ..constants import should_use_bf16_for_op
        use_bf16 = should_use_bf16_for_op(q_refined.numel(), "svd")
        device_type = 'cuda' if q_refined.is_cuda else 'cpu'
        
        for i in pbar:
            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss = torch.linalg.norm(U_k.T @ (q_refined * scale - W_float32) @ Vh_k.T)
                else:
                    loss = torch.linalg.norm(U_k.T @ (q_refined * scale - W_float32) @ Vh_k.T)
            if loss.item() < best_loss:
                best_loss, worse_loss_counter = loss.item(), 0
            else:
                worse_loss_counter += 1
            if worse_loss_counter > self.early_stop_stall: break
            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        grad = U_k @ ((U_k.T @ (q_refined * scale - W_float32) @ Vh_k.T) / loss.clamp_min(1e-20)) @ Vh_k
                else:
                    grad = U_k @ ((U_k.T @ (q_refined * scale - W_float32) @ Vh_k.T) / loss.clamp_min(1e-20)) @ Vh_k
                q_refined -= curr_lr * (grad * scale)
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})
        pbar.close()
        return q_refined

    def _optimize_int8_learned_rounding(self, W_float32, qdata, scale):
        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        if self.optimizer_choice == "adamw": final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam": final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k)
        else: final_qdata = self._optimize_int8_original(W_float32, qdata, scale, U_k, Vh_k)
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata

    def _optimize_int8_adamw(self, W_float32, qdata, scale, U_k, Vh_k):
        M, N = W_float32.shape
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = AdamW([delta], lr=curr_lr)
        best_loss, best_delta, worse_loss_counter = float("inf"), delta.detach().clone(), 0
        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (AdamW)", leave=False, dynamic_ncols=True)
        
        # BF16 optimization check for AdamW
        from ..constants import should_use_bf16_for_op
        use_bf16 = should_use_bf16_for_op(qdata_float.numel(), "svd")
        device_type = 'cuda' if qdata_float.is_cuda else 'cpu'
        
        for i in pbar:
            optimizer.zero_grad()
            dq = self._int8_dequantize_blockwise(qdata_float + delta, scale, M, N, self.block_size)
            # BF16 for forward pass, FP32 for optimizer state
            if use_bf16:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
            else:
                loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss, best_delta, worse_loss_counter = loss.item(), delta.detach().clone(), 0
            else:
                worse_loss_counter += 1
            if worse_loss_counter > self.early_stop_stall: break
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})
        pbar.close()
        return (qdata_float + best_delta).clamp(-127.0, 127.0).round().to(TARGET_INT8_DTYPE)

    def _optimize_int8_radam(self, W_float32, qdata, scale, U_k, Vh_k):
        M, N = W_float32.shape
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = RAdam([delta], lr=curr_lr)
        best_loss, best_delta, worse_loss_counter = float("inf"), delta.detach().clone(), 0
        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (RAdam)", leave=False, dynamic_ncols=True)
        
        # BF16 optimization check for RAdam
        from ..constants import should_use_bf16_for_op
        use_bf16 = should_use_bf16_for_op(qdata_float.numel(), "svd")
        device_type = 'cuda' if qdata_float.is_cuda else 'cpu'
        
        for i in pbar:
            optimizer.zero_grad()
            dq = self._int8_dequantize_blockwise(qdata_float + delta, scale, M, N, self.block_size)
            if use_bf16:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
            else:
                loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss, best_delta, worse_loss_counter = loss.item(), delta.detach().clone(), 0
            else:
                worse_loss_counter += 1
            if worse_loss_counter > self.early_stop_stall: break
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})
        pbar.close()
        return (qdata_float + best_delta).clamp(-127.0, 127.0).round().to(TARGET_INT8_DTYPE)

    def _optimize_int8_original(self, W_float32, qdata, scale, U_k, Vh_k):
        M, N = W_float32.shape
        q_refined = qdata.to(COMPUTE_DTYPE)
        best_loss, worse_loss_counter = float("inf"), 0
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (Original)", leave=False, dynamic_ncols=True)
        
        # BF16 optimization check
        from ..constants import should_use_bf16_for_op
        use_bf16 = should_use_bf16_for_op(q_refined.numel(), "svd")
        device_type = 'cuda' if q_refined.is_cuda else 'cpu'
        
        for i in pbar:
            with torch.no_grad():
                dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, self.block_size)
                if use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
                else:
                    loss = torch.linalg.norm(U_k.T @ (dq - W_float32) @ Vh_k.T)
            if loss.item() < best_loss:
                best_loss, worse_loss_counter = loss.item(), 0
            else:
                worse_loss_counter += 1
            if worse_loss_counter > self.early_stop_stall: break
            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        grad = U_k @ ((U_k.T @ (dq - W_float32) @ Vh_k.T) / loss.clamp_min(1e-20)) @ Vh_k
                else:
                    grad = U_k @ ((U_k.T @ (dq - W_float32) @ Vh_k.T) / loss.clamp_min(1e-20)) @ Vh_k
                grad_blocked = grad.reshape(M // self.block_size, self.block_size, N // self.block_size, self.block_size).permute(0, 2, 1, 3)
                grad_scaled = (grad_blocked * scale.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(M, N)
                q_refined -= curr_lr * grad_scaled
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})
        pbar.close()
        return q_refined.clamp(-127.0, 127.0).round().to(TARGET_INT8_DTYPE)

    def _int8_dequantize_blockwise(self, qdata, scale, M, N, block_size):
        q_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size).permute(0, 2, 1, 3)
        return (q_blocked * scale.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(M, N)
