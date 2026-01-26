"""
INT8 quantization functions for convert_to_quant.

Main quantization function that processes safetensors files and applies
INT8 quantization with learned rounding optimization.
"""
import gc
import json
import os
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any, Optional
from tqdm import tqdm

from .constants import (
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    INT8_SYMMETRIC_MAX,
    AVOID_KEY_NAMES,
    T5XXL_REMOVE_KEY_NAMES,
    MODEL_FILTERS,
    VALID_QUANT_FORMATS,
    NORMALIZE_SCALES_ENABLED,
)
from .converters.learned_rounding import LearnedRoundingConverter
from .converters.gptq_int8 import GPTQInt8Converter
from .converters.quip_int8 import QuIPInt8Converter
from .config.layer_config import get_layer_settings
from .utils.tensor_utils import normalize_tensorwise_scales, load_lora_tensors
from .utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from .utils.memory_efficient_loader import MemoryEfficientSafeOpen
from .pinned_transfer import get_pinned_transfer_stats
from .utils.logging import info, verbose, debug, minimal, warning, error, log_debug
from .utils.quality_metrics import QualityReporter

@log_debug
def convert_to_int8(
    input_file: str,
    output_file: str,
    comfy_quant: bool,
    filter_flags: Dict[str, bool],
    calib_samples: int,
    seed: int,
    fp16: bool = False,
    fallback: Optional[str] = None,
    custom_layers: Optional[str] = None,
    exclude_layers: Optional[str] = None,
    custom_type: Optional[str] = None,
    custom_block_size: Optional[int] = None,
    custom_scaling_mode: Optional[str] = None,
    custom_simple: bool = False,
    custom_heur: bool = False,
    fallback_block_size: Optional[int] = None,
    fallback_simple: bool = False,
    full_precision_matrix_mult: bool = False,
    skip_inefficient_layers: bool = False,
    include_input_scale: bool = False,
    no_learned_rounding: bool = False,
    save_quant_metadata: bool = False,
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    low_memory: bool = False,
    report_quality: bool = False,
    quality_threshold: float = 30.0,
    smoothquant: bool = False,
    smoothquant_alpha: float = 0.5,
    calibration_data_path: Optional[str] = None,
    calibration_lora_path: Optional[str] = None,
    gptq_actorder: bool = False,
    gptq_fast: bool = True,
    gptq_turbo: bool = False,
    quip_actorder: bool = True,
    quip_hadamard: bool = True,
    quip_seed: Optional[int] = None,
    **converter_kwargs,
):
    # Ensure filter_flags is a dict
    filter_flags = filter_flags or {}

    # Determine target format
    if fp16:
        target_format = "fp16"
        format_name = "FP16"
    else:
        target_format = "int8"
        format_name = "INT8"

    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    if target_format == "int8":
        info("Target format: INT8 (block-wise quantization)")
        info(f"INT8 Range: [{-INT8_SYMMETRIC_MAX}, {INT8_SYMMETRIC_MAX}]")
    else:
        info("Target format: FP16")
    info("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    # Use unified loader (handles both standard and low-memory modes)
    try:
        loader = MemoryEfficientSafeOpen(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    all_keys = loader.keys()
    original_metadata = loader.metadata()
    quant_metadata_layers = {} if save_quant_metadata else None
    quality_reporter = QualityReporter(threshold=quality_threshold) if report_quality else None

    # Add target_format and no_learned_rounding to converter kwargs
    converter_kwargs["target_format"] = target_format
    converter_kwargs["no_learned_rounding"] = no_learned_rounding

    # Get format-aware block_size default
    format_block_sizes = {"int8": 128, "fp16": None}
    block_size = converter_kwargs.get("block_size") or format_block_sizes.get(target_format, 128)

    # Helper function to create converter for a specific format type
    def create_converter_for_format(fmt: str, overrides: dict = None, is_primary: bool = True):
        kwargs = converter_kwargs.copy()
        kwargs["target_format"] = fmt

        if not is_primary:
            kwargs["no_learned_rounding"] = False
        
        kwargs["smoothquant"] = smoothquant
        kwargs["smoothquant_alpha"] = smoothquant_alpha

        if overrides:
            kwargs.update(overrides)

        return LearnedRoundingConverter(**kwargs)

    # Create converters
    converters = {"primary": create_converter_for_format(target_format)}

    if fallback:
        fallback_overrides = {}
        if fallback_block_size is not None:
            fallback_overrides["block_size"] = fallback_block_size
        if fallback_simple:
            fallback_overrides["no_learned_rounding"] = True
        converters["fallback"] = create_converter_for_format(
            fallback, fallback_overrides if fallback_overrides else None, is_primary=False
        )
        info(f"Fallback quantization enabled: {fallback.upper()} for excluded layers")

    if custom_layers and custom_type:
        custom_overrides = {}
        if custom_block_size is not None:
            custom_overrides["block_size"] = custom_block_size
        if custom_scaling_mode is not None:
            custom_overrides["scaling_mode"] = custom_scaling_mode
        if custom_simple:
            custom_overrides["no_learned_rounding"] = True
        converters["custom"] = create_converter_for_format(
            custom_type, custom_overrides if custom_overrides else None, is_primary=False
        )
        info(f"Custom layer quantization enabled: {custom_type.upper()} for pattern '{custom_layers}'")

    # Compile regex patterns
    custom_pattern = re.compile(custom_layers) if custom_layers else None
    exclude_pattern = re.compile(exclude_layers) if exclude_layers else None

    minimal("Scanning model and generating calibration data...")
    calibration_data_cache = {}
    real_calibration_data = {}
    if calibration_data_path:
        info(f"Loading calibration data from: {calibration_data_path}")
        try:
            if calibration_data_path.endswith(".json"):
                with open(calibration_data_path, "r") as f:
                    json_data = json.load(f)
                for layer_name, stats in json_data.items():
                    if isinstance(stats, dict):
                        if "channel_max" in stats:
                            real_calibration_data[f"{layer_name}.channel_max"] = torch.tensor(stats["channel_max"])
                        if "channel_mean" in stats:
                            real_calibration_data[f"{layer_name}.channel_mean"] = torch.tensor(stats["channel_mean"])
                        if "input_scale" in stats:
                            real_calibration_data[f"{layer_name}.input_scale"] = torch.tensor(stats["input_scale"])
                    else:
                        real_calibration_data[f"{layer_name}.input_scale"] = torch.tensor(stats)
            else:
                with safe_open(calibration_data_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        real_calibration_data[key] = f.get_tensor(key)
            info(f"  Loaded {len(real_calibration_data)} calibration entries.")
        except Exception as e:
            error(f"ERROR: Failed to load calibration data: {e}")

    lora_tensors = load_lora_tensors(calibration_lora_path) if calibration_lora_path else {}
    
    for key in all_keys:
        if key.endswith(".weight"):
            shape = loader.get_shape(key)
            if len(shape) == 2:
                in_features = shape[1]
                if in_features not in calibration_data_cache:
                    # Find matching LoRA if available
                    lora_match = None
                    if lora_tensors:
                        base_name = key.rsplit(".weight", 1)[0]
                        norm_base = base_name.replace("model.diffusion_model.", "").replace("transformer.", "")
                        if norm_base in lora_tensors:
                            lora_match = lora_tensors[norm_base]
                    
                    if lora_match is not None:
                        # Use LoRA directions + some noise
                        rank = lora_match.shape[0]
                        if rank >= calib_samples:
                            x = lora_match[:calib_samples].to(seed_device).to(COMPUTE_DTYPE)
                        else:
                            x_lora = lora_match.to(seed_device).to(COMPUTE_DTYPE)
                            x_rand = torch.randn(
                                calib_samples - rank,
                                in_features,
                                dtype=COMPUTE_DTYPE,
                                generator=seed_generator,
                                device=seed_device,
                            )
                            x = torch.cat([x_lora, x_rand], dim=0)
                        calibration_data_cache[in_features] = x
                    else:
                        calibration_data_cache[in_features] = torch.randn(
                            calib_samples,
                            in_features,
                            dtype=COMPUTE_DTYPE,
                            generator=seed_generator,
                            device=seed_device,
                        )
    if lora_tensors:
        info(f"Simulated calibration data generated (using {len(lora_tensors)} LoRA matches).\n")
    else:
        info("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted(
        [
            key
            for key in all_keys
            if (key.endswith(".weight") or "lora_A" in key or "lora_B" in key)
            and loader.get_ndim(key) == 2
        ]
    )
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0
    custom_count = 0
    fallback_count = 0

    info(f"Found {total_weights} weight tensors to potentially process.")
    info("-" * 60)

    for i, key in enumerate(weight_keys):
        exclusion_reason = ""
        use_custom = False
        use_fallback = False
        use_layer_config = False
        layer_format = target_format
        layer_settings = None

        text_encoder_filter = (
            filter_flags.get("t5xxl") or
            filter_flags.get("mistral") or
            filter_flags.get("visual")
        )

        if filter_flags.get("t5xxl") and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            info(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            skipped_count += 1
            continue

        if layer_config:
            layer_settings = get_layer_settings(key, layer_config, fullmatch=layer_config_fullmatch)
            if layer_settings:
                if layer_settings.get("skip", False):
                    info(f"({i+1}/{total_weights}) Skipping (layer-config): {key}")
                    original_tensor = loader.get_tensor(key)
                    new_tensors[key] = original_tensor.to(device="cpu", dtype=original_tensor.dtype)
                    loader.mark_processed(key)
                    skipped_count += 1
                    continue
                use_layer_config = True
                layer_format = "int8" if layer_settings["format"].startswith("int8") else "fp16"

        if not use_layer_config and custom_pattern and custom_pattern.search(key):
            use_custom = True
            layer_format = custom_type

        if not use_custom and not use_layer_config and exclude_pattern and exclude_pattern.search(key):
            exclusion_reason = "regex exclusion (--exclude-layers)"

        if not use_custom and not use_layer_config:
            for filter_name, is_active in filter_flags.items():
                if not is_active: continue
                cfg = MODEL_FILTERS[filter_name]
                if cfg.get("exclude") and any(n in key for n in cfg["exclude"]):
                    exclusion_reason = f"{filter_name} exclusion"
                    break
                if cfg.get("highprec") and any(n in key for n in cfg["highprec"]):
                    exclusion_reason = f"{filter_name} keep in high precision"
                    break

        if exclusion_reason and not use_custom and not use_layer_config:
            if fallback:
                use_fallback = True
                layer_format = fallback
                info(f"({i+1}/{total_weights}) Processing (fallback {fallback.upper()}): {key} (was: {exclusion_reason})")
            else:
                info(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})")
                original_tensor = loader.get_tensor(key)
                new_tensors[key] = original_tensor.to(device="cpu", dtype=original_tensor.dtype)
                loader.mark_processed(key)
                skipped_count += 1
                continue

        if use_layer_config:
            info(f"({i+1}/{total_weights}) Processing (config {layer_settings['format']}): {key}")
            custom_count += 1
        elif use_custom:
            info(f"({i+1}/{total_weights}) Processing (custom {custom_type.upper()}): {key}")
            custom_count += 1
        elif use_fallback:
            fallback_count += 1
        else:
            info(f"({i+1}/{total_weights}) Processing ({format_name}): {key}")

        processed_count += 1
        original_tensor = loader.get_tensor(key)

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            new_tensors[key] = original_tensor.to(device="cpu", dtype=original_tensor.dtype)
            continue

        is_lora_layer = "lora_" in key
        apply_heur = (custom_heur if use_custom else skip_inefficient_layers) and not is_lora_layer
        
        if apply_heur:
            should_skip, skip_perf_reason = should_skip_layer_for_performance(original_tensor, block_size)
            if should_skip:
                new_tensors[key] = original_tensor.to(device="cpu", dtype=original_tensor.dtype)
                loader.mark_processed(key)
                skipped_count += 1
                continue

        if use_layer_config:
            cfg_overrides = {}
            if layer_settings.get("block_size") is not None: cfg_overrides["block_size"] = layer_settings["block_size"]
            if layer_settings.get("scaling_mode") is not None: cfg_overrides["scaling_mode"] = layer_settings["scaling_mode"]
            if layer_settings.get("simple"): cfg_overrides["no_learned_rounding"] = True
            converter = create_converter_for_format(layer_format, cfg_overrides)
        elif use_custom:
            converter = converters["custom"]
        elif use_fallback:
            converter = converters["fallback"]
        else:
            converter = converters["primary"]

        is_int8 = layer_format == "int8"
        is_fp16 = layer_format == "fp16"

        act_scales = None
        base_name = key[: key.rfind(".weight")] if key.endswith(".weight") else key
        channel_max_key = f"{base_name}.channel_max"
        
        if channel_max_key in real_calibration_data:
            act_scales = real_calibration_data[channel_max_key]
        elif smoothquant:
            in_features = original_tensor.shape[1]
            if in_features in calibration_data_cache:
                act_scales = calibration_data_cache[in_features].abs().amax(dim=0)

        optimizer_type = converter_kwargs.get("optimizer")
        if optimizer_type == "gptq" and is_int8:
            gptq_converter = GPTQInt8Converter(
                block_size=converter.block_size,
                device=device,
                actorder=gptq_actorder,
                lazy_updates=gptq_fast,
                use_triton=gptq_turbo,
                low_memory=low_memory
            )
            H = None
            in_features = original_tensor.shape[1]
            if in_features in calibration_data_cache:
                # Use CPU for Hessian calculation if in low_memory mode to save VRAM
                calc_device = "cpu" if (low_memory or in_features > 4096) else device
                X = calibration_data_cache[in_features].to(calc_device).float()
                H = X.T @ X
            q_tensor, dequant_s, dequant_w = gptq_converter.convert(original_tensor, H=H)
        elif optimizer_type == "quip" and is_int8:
            quip_converter = QuIPInt8Converter(
                block_size=converter.block_size,
                device=device,
                actorder=quip_actorder,
                use_hadamard=quip_hadamard,
                seed=quip_seed,
                use_triton=gptq_turbo,
                lazy_updates=gptq_fast
            )
            quip_converter.low_memory = low_memory
            H = None
            in_features = original_tensor.shape[1]
            # CRITICAL FIX: Only use Hessian from real calibration data
            # Random synthetic data degrades quality by ~2.7 dB (see tests/diagnose_hessian_impact.py)
            # Without real calibration, H=None (identity) gives better results
            if calibration_data_path and in_features in calibration_data_cache:
                # Use CPU for Hessian calculation if in low_memory mode to save VRAM
                calc_device = "cpu" if (low_memory or in_features > 4096) else device
                X = calibration_data_cache[in_features].to(calc_device).float()
                H = X.T @ X
                debug(f"      QuIP using computed Hessian from real calibration data")
            else:
                debug(f"      QuIP using identity Hessian (no real calibration data)")
            # QuIP now returns up to 6 values
            res = quip_converter.convert(
                original_tensor, 
                H=H, 
                activation_scales=act_scales,
                smoothquant_alpha=smoothquant_alpha
            )
            q_tensor, dequant_s, dequant_w = res[0], res[1], res[2]
            smooth_factors = res[3] if len(res) > 3 else None
            s_u = res[4] if len(res) > 4 else None
            s_v = res[5] if len(res) > 5 else None
        else:
            res = converter.convert(original_tensor, activation_scales=act_scales)
            q_tensor, dequant_s, dequant_w = res[0], res[1], res[2]
            smooth_factors = res[3] if len(res) > 3 else None
            s_u, s_v = None, None

        new_tensors[key] = q_tensor.to(device="cpu")
        if smooth_factors is not None:
            new_tensors[f"{base_name}.smooth_factors"] = smooth_factors.to(device="cpu", dtype=SCALE_DTYPE)
        if s_u is not None:
            new_tensors[f"{base_name}.quip_s_u"] = s_u.to(device="cpu", dtype=SCALE_DTYPE)
        if s_v is not None:
            new_tensors[f"{base_name}.quip_s_v"] = s_v.to(device="cpu", dtype=SCALE_DTYPE)
            
        if quality_reporter:
            quality_reporter.add_layer(key, original_tensor, dequant_w)

        bias_key = f"{base_name}.bias"

        if comfy_quant is True:
            layer_block_size = converter.block_size
            layer_full_precision_mm = full_precision_matrix_mult
            if use_layer_config and "full_precision_matrix_mult" in layer_settings:
                layer_full_precision_mm = layer_settings["full_precision_matrix_mult"]

            if is_int8:
                new_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                
                if converter.scaling_mode == "axis":
                    comfy_quant_format = "int8_axiswise"
                    block_size_for_meta = None
                elif converter.scaling_mode == "tensor":
                    comfy_quant_format = "int8_tensorwise"
                    block_size_for_meta = None
                else:
                    comfy_quant_format = "int8_blockwise"
                    block_size_for_meta = layer_block_size

                comfy_quant_tensor = create_comfy_quant_tensor(
                    comfy_quant_format,
                    block_size=block_size_for_meta,
                    full_precision_matrix_mult=layer_full_precision_mm or None,
                )
                # Only write input_scale if real calibration data is available
                input_scale_key = f"{base_name}.input_scale"
                if input_scale_key in real_calibration_data:
                    new_tensors[input_scale_key] = real_calibration_data[input_scale_key].to(device="cpu", dtype=torch.float32)
                # Otherwise, omit input_scale entirely - inference will use dynamic quantization

                new_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor.to(device="cpu")

                if save_quant_metadata:
                    meta_entry = {"format": comfy_quant_format}
                    if block_size_for_meta is not None: meta_entry["group_size"] = block_size_for_meta
                    if layer_full_precision_mm: meta_entry["full_precision_matrix_mult"] = True
                    quant_metadata_layers[base_name] = meta_entry
        else:
            if is_int8:
                new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                if include_input_scale or text_encoder_filter:
                    if text_encoder_filter:
                        new_tensors[f"{base_name}.scale_input"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                    else:
                        new_tensors[f"{base_name}.scale_input"] = torch.ones_like(dequant_s, dtype=SCALE_DTYPE, device="cpu")

        layer_uses_simple = custom_simple if use_custom else (fallback_simple if use_fallback else no_learned_rounding)

        if bias_key in all_keys:
            if layer_uses_simple:
                new_tensors[bias_key] = loader.get_tensor(bias_key)
            else:
                with torch.no_grad():
                    original_bias = loader.get_tensor(bias_key)
                    in_features = original_tensor.shape[1]
                    if in_features not in calibration_data_cache:
                        new_tensors[bias_key] = original_bias
                    else:
                        X_calib_dev = calibration_data_cache[in_features].to(device=device)
                        W_orig_dev = original_tensor.to(device=device, dtype=COMPUTE_DTYPE)
                        W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                        b_orig_dev = original_bias.to(device=device, dtype=COMPUTE_DTYPE)
                        weight_error = W_orig_dev - W_dequant_dev
                        output_error = X_calib_dev @ weight_error.T
                        bias_correction = output_error.mean(dim=0)
                        b_new = b_orig_dev - bias_correction
                        new_tensors[bias_key] = b_new.to(device="cpu", dtype=original_bias.dtype)
                        if device == "cuda": torch.cuda.empty_cache()

    for key in all_keys:
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and filter_flags.get("t5xxl"): continue
        if key not in new_tensors:
            new_tensors[key] = loader.get_tensor(key)

    loader.close()
    calibration_data_cache.clear()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        output_metadata = dict(original_metadata)
        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)
        save_kwargs = {"metadata": output_metadata} if output_metadata else {}

        new_tensors, normalized_count = normalize_tensorwise_scales(new_tensors, NORMALIZE_SCALES_ENABLED)
        save_file(new_tensors, output_file, **save_kwargs)
        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return

    if quality_reporter:
        info("\n" + quality_reporter.get_report_string())
