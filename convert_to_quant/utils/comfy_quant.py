"""
ComfyUI quantization metadata utilities for convert_to_quant.

Handles .comfy_quant tensor creation, parsing, editing, and performance heuristics.
"""
import json
import os
import re
import torch
from typing import Dict, Tuple, Optional, Any, List
from safetensors import safe_open
from safetensors.torch import save_file

from .tensor_utils import dict_to_tensor, tensor_to_dict, normalize_tensorwise_scales
from ..constants import NORMALIZE_SCALES_ENABLED
from .logging import info, verbose, warning, error, minimal


# Block-based formats that require group_size
BLOCK_BASED_FORMATS = (
    "int8_blockwise",
)

def create_comfy_quant_tensor(
    format_type: str,
    block_size: Optional[int] = None,
    full_precision_matrix_mult: Optional[bool] = None,
) -> torch.Tensor:
    """
    Create a .comfy_quant layer configuration tensor for ComfyUI.

    Args:
        format_type: One of "int8_blockwise", "int8_tensorwise", "int8_axiswise"
        block_size: Block/group size for quantization (for block-based formats)
        full_precision_matrix_mult: If True, adds "full_precision_matrix_mult": True.

    Returns:
        torch.uint8 tensor containing JSON-encoded layer configuration
    """
    comfy_quant = {"format": format_type}

    if block_size is not None and format_type in BLOCK_BASED_FORMATS:
        comfy_quant["group_size"] = block_size

    if full_precision_matrix_mult is True:
        comfy_quant["full_precision_matrix_mult"] = True

    return dict_to_tensor(comfy_quant)

def fix_comfy_quant_params_structure(
    comfy_quant_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, bool]:
    """Check and fix comfy_quant config with incorrect nested params structure."""
    try:
        config = tensor_to_dict(comfy_quant_tensor)
    except Exception:
        return comfy_quant_tensor, False

    if "params" not in config:
        return comfy_quant_tensor, False

    params = config.pop("params")
    if isinstance(params, dict):
        if "group_size" in params:
            config["group_size"] = params["group_size"]
        for key, value in params.items():
            if key != "group_size" and key not in config:
                config[key] = value

    return dict_to_tensor(config), True

def parse_add_keys_string(add_keys_str: str) -> Dict[str, Any]:
    """Parse a Python-like key:value string into a dictionary."""
    result = {}
    if not add_keys_str or not add_keys_str.strip():
        return result

    pattern = r"'([^']+)':\s*([^,]+?)(?:,|$)"
    matches = re.findall(pattern, add_keys_str.strip())

    for key, value in matches:
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            result[key] = value[1:-1]
        elif value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        else:
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value

    return result

def edit_comfy_quant(
    input_file: str,
    output_file: str,
    remove_keys: Optional[List[str]] = None,
    add_keys_str: Optional[str] = None,
    layer_filter: Optional[str] = None,
    save_quant_metadata: bool = False,
):
    """Edit comfy_quant layer configurations and _quantization_metadata in a model."""
    info("ComfyQuant Layer & Metadata Editor")
    info("=" * 60)
    info(f"Input:  {input_file}")
    info(f"Output: {output_file}")

    add_keys = parse_add_keys_string(add_keys_str) if add_keys_str else {}

    if remove_keys: info(f"Keys to remove: {remove_keys}")
    if add_keys: info(f"Keys to add: {add_keys}")
    if layer_filter:
        info(f"Layer filter: {layer_filter}")
        try:
            layer_regex = re.compile(layer_filter)
        except re.error as e:
            error(f"FATAL: Invalid regex pattern '{layer_filter}': {e}")
            return
    else:
        layer_regex = None

    info("-" * 60)

    tensors = {}
    existing_metadata: Optional[Dict[str, str]] = None
    with safe_open(input_file, framework="pt", device="cpu") as f:
        existing_metadata = f.metadata()
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    quant_metadata: Optional[Dict[str, Any]] = None
    quant_metadata_modified = False
    if existing_metadata and "_quantization_metadata" in existing_metadata:
        try:
            quant_metadata = json.loads(existing_metadata["_quantization_metadata"])
            info(f"Found _quantization_metadata header with {len(quant_metadata.get('layers', {}))} layer entries")
        except json.JSONDecodeError as e:
            warning(f"  WARNING: Failed to parse _quantization_metadata: {e}")
            quant_metadata = None

    edited_count = 0
    skipped_filter = 0
    skipped_no_change = 0
    total_comfy_quant = 0
    keys_removed: Dict[str, int] = {}
    keys_added: Dict[str, int] = {}

    for key in list(tensors.keys()):
        if not key.endswith(".comfy_quant"): continue
        total_comfy_quant += 1
        base_name = key[:-12]
        if layer_regex and not layer_regex.search(base_name):
            skipped_filter += 1
            continue
        try:
            config = tensor_to_dict(tensors[key])
        except Exception as e:
            warning(f"  WARNING: Failed to decode {key}: {e}")
            continue
        original_config = config.copy()
        if remove_keys:
            for k in remove_keys:
                if k in config:
                    del config[k]
                    keys_removed[k] = keys_removed.get(k, 0) + 1
        if add_keys:
            for k, v in add_keys.items():
                if k not in config or config[k] != v:
                    config[k] = v
                    keys_added[k] = keys_added.get(k, 0) + 1
        if config != original_config:
            tensors[key] = dict_to_tensor(config)
            edited_count += 1
            if quant_metadata and "layers" in quant_metadata and base_name in quant_metadata["layers"]:
                quant_metadata["layers"][base_name] = config
                quant_metadata_modified = True
        else:
            skipped_no_change += 1

    if save_quant_metadata:
        info("-" * 60)
        info("Generating _quantization_metadata from .comfy_quant tensors...")
        generated_layers = {}
        for key in tensors.keys():
            if not key.endswith(".comfy_quant"): continue
            base_name = key[:-12]
            try:
                config = tensor_to_dict(tensors[key])
                generated_layers[base_name] = config
            except Exception as e:
                warning(f"  WARNING: Failed to parse {key}: {e}")
        if generated_layers:
            quant_metadata = {"format_version": "1.0", "layers": generated_layers}
            quant_metadata_modified = True
            info(f"  Generated metadata for {len(generated_layers)} layers")

    save_kwargs: Dict[str, Any] = {}
    if quant_metadata and quant_metadata_modified:
        output_metadata = dict(existing_metadata) if existing_metadata else {}
        output_metadata["_quantization_metadata"] = json.dumps(quant_metadata)
        save_kwargs["metadata"] = output_metadata
    elif existing_metadata:
        save_kwargs["metadata"] = existing_metadata

    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        tensors, normalized_count = normalize_tensorwise_scales(tensors, NORMALIZE_SCALES_ENABLED)
        save_file(tensors, output_file, **save_kwargs)
        info("Edit complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")

def should_skip_layer_for_performance(tensor: torch.Tensor, block_size: int) -> Tuple[bool, str]:
    """Check if a layer should be skipped based on performance heuristics."""
    if tensor.ndim != 2: return True, "not 2D"
    rows, cols = tensor.shape
    if rows < block_size or cols < block_size: return True, f"dimension smaller than block_size ({block_size})"
    if rows % block_size != 0 or cols % block_size != 0: return True, f"dimensions not divisible by block_size ({block_size})"
    return False, ""
