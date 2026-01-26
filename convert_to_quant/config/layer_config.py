"""
Layer configuration loading and matching for convert_to_quant.

Provides regex-based layer pattern matching for per-layer quantization settings.
"""
import json
import os
import re
import torch
from typing import Dict, Any, Optional, Tuple
from safetensors import safe_open
from tqdm import tqdm

from ..constants import VALID_QUANT_FORMATS

def pattern_specificity(pattern: str) -> tuple:
    """Calculate specificity score for a regex pattern."""
    if pattern.startswith("_"): return (999, 0)
    literal_pattern = re.sub(r"\\.|\\[.*?\\]|\\(.*?\\)|[.*+?^${}|\\\\]", "", pattern)
    literal_len = len(literal_pattern)
    has_number = bool(re.search(r"\\d|\\\\d", pattern))
    tier = 0 if (has_number and literal_len >= 8) else 1
    return (tier, literal_len)

def load_layer_config(config_path: str) -> Dict[str, Any]:
    """Load and validate layer configuration from JSON file."""
    if not os.path.exists(config_path): raise FileNotFoundError(f"Layer config file not found: {config_path}")
    with open(config_path, "r") as f: config = json.load(f)
    if not isinstance(config, dict): raise ValueError(f"Layer config must be a JSON object, got {type(config).__name__}")

    compiled_patterns = {}
    for key, settings in config.items():
        if key.startswith("_"):
            if key == "_default" and isinstance(settings, dict) and "format" in settings:
                fmt = settings["format"]
                if not fmt: raise ValueError("_default has empty 'format' field.")
                if fmt not in VALID_QUANT_FORMATS: raise ValueError(f"_default has invalid format '{fmt}'. Valid: {sorted(VALID_QUANT_FORMATS)}")
            continue

        if not isinstance(settings, dict): raise ValueError(f"Layer config entry '{key}' must be an object")
        try: compiled_patterns[key] = re.compile(key)
        except re.error as e: raise ValueError(f"Layer config entry '{key}' has invalid regex: {e}")

        if settings.get("skip"): continue
        if "format" not in settings: raise ValueError(f"Layer config entry '{key}' missing 'format'")
        fmt = settings["format"]
        if not fmt: raise ValueError(f"Layer config entry '{key}' has empty 'format'")
        if fmt not in VALID_QUANT_FORMATS: raise ValueError(f"Layer config entry '{key}' has invalid format '{fmt}'")

    config["_compiled_patterns"] = compiled_patterns
    print(f"Loaded layer config with {len([k for k in config if not k.startswith('_')])} layer patterns")
    return config

def get_layer_settings(layer_key: str, config: Dict[str, Any], fullmatch: bool = False) -> Optional[Dict[str, Any]]:
    """Find the most specific matching config entry for a layer using regex."""
    base_key = layer_key[:-7] if layer_key.endswith(".weight") else layer_key
    compiled_patterns = config.get("_compiled_patterns", {})
    matches = []
    for pattern, settings in config.items():
        if pattern.startswith("_"): continue
        regex = compiled_patterns.get(pattern) or re.compile(pattern)
        if (regex.fullmatch(base_key) if fullmatch else regex.search(base_key)):
            matches.append((pattern_specificity(pattern), pattern, settings))

    if matches:
        matches.sort(key=lambda x: (x[0][0], -x[0][1]))
        return matches[0][2]
    return config.get("_default")

def generate_config_template(input_file: str, output_path: str, block_size: int = 128):
    """Generate a JSON config template from model."""
    print(f"Generating layer config template from: {input_file}")
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            weight_keys = [k for k in f.keys() if k.endswith(".weight")]
    except Exception as e: raise RuntimeError(f"Error reading model file: {e}")

    config = {"_default": {"format": ""}, "_exclusions": []}
    viable_count, skipped_count = 0, 0
    with safe_open(input_file, framework="pt", device="cpu") as f:
        for key in tqdm(weight_keys, desc="Analyzing layers"):
            tensor = f.get_tensor(key)
            if tensor.numel() == 0 or tensor.ndim != 2 or tensor.shape[0] < 16 or tensor.shape[1] < 16:
                skipped_count += 1
                continue
            config[key[:-7]] = {"format": "", "_shape": list(tensor.shape)}
            viable_count += 1

    with open(output_path, "w") as f: json.dump(config, f, indent=2)
    print(f"Template written to: {output_path} ({viable_count} viable layers)")
    return viable_count, skipped_count
