"""
Tensor utility functions for convert_to_quant.

Provides serialization helpers for dictionary/tensor conversion and scale normalization.
"""
import json
import os
import torch
from typing import Dict, Tuple, List, Optional, Any

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


def load_multiple_loras_for_merging(lora_paths: List[str]) -> Dict[str, List[Dict[str, any]]]:
    """
    Load multiple LoRA files and group them by base layer name.
    
    Returns a dict mapping base layer names to a list of LoRA configurations.
    Format: {base_name: [{"lora_A": tensor, "lora_B": tensor, "alpha": float, "rank": int}, ...]}
    
    Args:
        lora_paths: List of paths to LoRA files
        
    Returns:
        Dict mapping base names to list of LoRA configs for that layer
    """
    from safetensors import safe_open
    combined_loras: Dict[str, List[Dict]] = {}
    
    if not lora_paths:
        return combined_loras
    
    for lora_path in lora_paths:
        if not lora_path or not os.path.exists(lora_path):
            continue
            
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            
            # Collect all lora_A keys
            lora_a_keys = [k for k in f.keys() if "lora_A" in k or "lora_down" in k]
            
            for a_key in lora_a_keys:
                # Determine corresponding B key
                if "lora_A" in a_key:
                    b_key = a_key.replace("lora_A", "lora_B")
                    base_name = a_key.replace(".lora_A.weight", "")
                else:  # lora_down
                    b_key = a_key.replace("lora_down", "lora_up")
                    base_name = a_key.replace(".lora_down.weight", "")
                
                # Normalize base name
                base_name = base_name.replace("model.diffusion_model.", "").replace("transformer.", "")
                
                if b_key in f.keys():
                    lora_a = f.get_tensor(a_key)
                    lora_b = f.get_tensor(b_key)
                    
                    # Get alpha from metadata if available, default to rank
                    rank = lora_a.shape[0]
                    alpha_key = base_name.replace(".", "_") + "_alpha"
                    alpha = float(metadata.get(alpha_key, rank)) if metadata else float(rank)
                    
                    if base_name not in combined_loras:
                        combined_loras[base_name] = []
                    
                    combined_loras[base_name].append({
                        "lora_A": lora_a,
                        "lora_B": lora_b,
                        "alpha": alpha,
                        "rank": rank,
                        "source": os.path.basename(lora_path)
                    })
    
    return combined_loras


def load_lora_for_merging(lora_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load LoRA weights for merging into base model.
    
    Returns a dict mapping base layer names to their LoRA A and B matrices.
    Format: {base_name: {"lora_A": tensor, "lora_B": tensor, "alpha": float}}
    """
    from safetensors import safe_open
    lora_weights = {}
    if not lora_path:
        return lora_weights
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        
        # First pass: collect all lora_A keys
        lora_a_keys = [k for k in f.keys() if "lora_A" in k or "lora_down" in k]
        
        for a_key in lora_a_keys:
            # Determine corresponding B key
            if "lora_A" in a_key:
                b_key = a_key.replace("lora_A", "lora_B")
                base_name = a_key.replace(".lora_A.weight", "")
            else:  # lora_down
                b_key = a_key.replace("lora_down", "lora_up")
                base_name = a_key.replace(".lora_down.weight", "")
            
            # Normalize base name
            base_name = base_name.replace("model.diffusion_model.", "").replace("transformer.", "")
            
            if b_key in f.keys():
                lora_a = f.get_tensor(a_key)
                lora_b = f.get_tensor(b_key)
                
                # Get alpha from metadata if available, default to rank
                rank = lora_a.shape[0]
                alpha_key = base_name.replace(".", "_") + "_alpha"
                alpha = float(metadata.get(alpha_key, rank)) if metadata else float(rank)
                
                lora_weights[base_name] = {
                    "lora_A": lora_a,
                    "lora_B": lora_b,
                    "alpha": alpha,
                    "rank": rank
                }
    
    return lora_weights


def merge_lora_into_weight(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    rank: int,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Merge LoRA weights into a base weight tensor.
    
    Formula: W_merged = W_base + (lora_B @ lora_A) * (alpha / rank) * scale
    
    Args:
        weight: Base weight tensor [out_features, in_features]
        lora_a: LoRA A matrix [rank, in_features]
        lora_b: LoRA B matrix [out_features, rank]
        alpha: LoRA alpha parameter
        rank: LoRA rank
        scale: Additional scaling factor (default 1.0)
        
    Returns:
        Merged weight tensor
    """
    with torch.no_grad():
        # Ensure compatible dtypes
        base_dtype = weight.dtype
        device = weight.device
        
        lora_a = lora_a.to(device=device, dtype=base_dtype)
        lora_b = lora_b.to(device=device, dtype=base_dtype)
        
        # Compute LoRA delta: B @ A
        delta = lora_b @ lora_a
        
        # Apply scaling
        scaling = (alpha / rank) * scale
        delta = delta * scaling
        
        # Merge with base weight
        merged = weight + delta
        
        return merged


def merge_multiple_loras(
    weight: torch.Tensor,
    lora_configs: list,
) -> torch.Tensor:
    """
    Merge multiple LoRAs into a base weight tensor with dampening.
    
    Args:
        weight: Base weight tensor
        lora_configs: List of dicts with keys: lora_A, lora_B, alpha, rank, scale
        
    Returns:
        Merged weight tensor
    """
    merged = weight
    
    for i, config in enumerate(lora_configs):
        # Apply dampening for multiple LoRAs: scale *= 0.9^i
        dampening = 0.9 ** i
        scale = config.get("scale", 1.0) * dampening
        
        merged = merge_lora_into_weight(
            merged,
            config["lora_A"],
            config["lora_B"],
            config["alpha"],
            config["rank"],
            scale
        )
    
    return merged
