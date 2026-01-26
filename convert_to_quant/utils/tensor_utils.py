"""
Tensor utility functions for convert_to_quant.

Provides serialization helpers for dictionary/tensor conversion and scale normalization.
"""
import json
import torch
from typing import Dict, Tuple

def dict_to_tensor(data_dict: dict) -> torch.Tensor:
    """Convert a dictionary to a torch.uint8 tensor containing JSON bytes."""
    json_str = json.dumps(data_dict)
    json_bytes = json_str.encode("utf-8")
    return torch.tensor(list(json_bytes), dtype=torch.uint8)

def tensor_to_dict(tensor: torch.Tensor) -> dict:
    """Convert a torch.uint8 tensor containing JSON bytes back to a dictionary."""
    if tensor.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {tensor.dtype}")
    json_bytes = bytes(tensor.tolist())
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)

def normalize_tensorwise_scales(tensors: Dict[str, torch.Tensor], enabled: bool = True) -> Tuple[Dict[str, torch.Tensor], int]:
    """Normalize 1-element scale tensors to scalars and ensure all tensors are contiguous."""
    from .logging import debug
    normalized_count = 0
    new_tensors = {}
    for key, tensor in tensors.items():
        # 1. Handle scale normalization if enabled
        if enabled and (key.endswith(".weight_scale") or key.endswith(".input_scale") or key.endswith(".scale_weight") or key.endswith(".scale_input")):
            if tensor.numel() == 1 and tensor.ndim > 0:
                new_tensors[key] = torch.tensor(tensor.item(), dtype=tensor.dtype)
                normalized_count += 1
                continue
        
        # 2. Ensure all tensors are contiguous for safetensors compatibility
        if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
            debug(f"Fixing non-contiguous tensor: {key}")
            new_tensors[key] = tensor.contiguous()
        else:
            new_tensors[key] = tensor
            
    return new_tensors, normalized_count

def generate_calibration_data(samples: int, in_features: int, device: str, seed: int) -> torch.Tensor:
    """Generate random calibration data."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(samples, in_features, device=device, generator=generator)

def adaptive_lr_update(lr: float, improved: bool, counter: int, worse_counter: int, small_mult: float) -> float:
    """Centralized adaptive LR update logic."""
    from ..constants import ADAPTIVE_LR_TIERS_IMPROVE, ADAPTIVE_LR_TIERS_DECAY
    if improved:
        for threshold, mult, max_lr in reversed(ADAPTIVE_LR_TIERS_IMPROVE):
            if counter >= threshold:
                return min(lr * mult * small_mult, max_lr)
    else:
        for threshold, mult, min_lr in reversed(ADAPTIVE_LR_TIERS_DECAY):
            if worse_counter >= threshold:
                return max(lr * mult, min_lr)
    return lr

def compute_bias_correction(X: torch.Tensor, W_orig: torch.Tensor, W_dq: torch.Tensor, b_orig: torch.Tensor) -> torch.Tensor:
    """Compute bias correction to minimize output error."""
    with torch.no_grad():
        error = (X @ W_orig.T) - (X @ W_dq.T)
        correction = error.mean(dim=0)
        return b_orig - correction

def load_lora_tensors(lora_path: str) -> dict:
    """Load LoRA tensors and group them by base layer name."""
    from safetensors import safe_open
    lora_data = {}
    if not lora_path:
        return lora_data
        
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "lora_A" in key or "lora_down" in key:
                # Normalize key to match base model
                base = key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
                # Strip common prefixes
                base = base.replace("model.diffusion_model.", "").replace("transformer.", "")
                lora_data[base] = f.get_tensor(key)
    return lora_data
