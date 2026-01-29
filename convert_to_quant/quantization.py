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
from typing import Dict, Any, Optional, List, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ParallelProcessor:
    """
    Parallel processing utility for layer-wise quantization.
    
    Uses thread pool for I/O-bound operations and can be extended
    for multi-GPU processing.
    """
    
    def __init__(self, max_workers: int = 4, use_parallel: bool = False):
        self.max_workers = max_workers
        self.use_parallel = use_parallel and max_workers > 1
        self._lock = threading.Lock()
        
    def map_parallel(self, func: Callable, items: List, desc: str = "Processing"):
        """
        Map function over items in parallel using thread pool.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            desc: Description for progress bar
            
        Returns:
            List of results in same order as items
        """
        if not self.use_parallel or len(items) <= 1:
            # Sequential processing
            results = []
            for item in tqdm(items, desc=desc):
                results.append(func(item))
            return results
        
        # Parallel processing with thread pool
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(func, item): idx 
                for idx, item in enumerate(items)
            }
            
            # Collect results with progress bar
            with tqdm(total=len(items), desc=f"{desc} (parallel)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error processing item {idx}: {e}")
                        raise
                    pbar.update(1)
        
        return results

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
from .utils.tensor_utils import normalize_tensorwise_scales, load_lora_tensors, load_lora_for_merging, load_multiple_loras_for_merging, merge_lora_into_weight, merge_multiple_loras
from .utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from .utils.memory_efficient_loader import MemoryEfficientSafeOpen
from .utils.hadamard import set_low_memory_mode
from .utils.tensor_prefetch import AsyncTensorPrefetcher
from .config.optimization_config import get_optimization_config
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
    streaming_mode: str = "balanced",
    streaming_thresholds: Optional[Dict[str, Optional[int]]] = None,
    no_memory_limits: bool = False,
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
    quip_store_transformed: bool = False,
    quip_checkpointed: bool = False,
    quip_checkpoint_threshold: int = 8192,
    quip_checkpoint_segments: int = 4,
    merge_lora_path: Optional[str] = None,
    merge_lora_paths: Optional[List[str]] = None,
    merge_lora_scale: float = 1.0,
    merge_lora_dampen: bool = True,
    **converter_kwargs,
):
    # Ensure filter_flags is a dict
    filter_flags = filter_flags or {}
    
    # Set low-memory mode for mixed precision Hadamard transform
    # streaming_mode != "off" means streaming is enabled
    streaming_enabled = streaming_mode != "off"
    if low_memory or streaming_enabled:
        set_low_memory_mode(True)
        # Get the actual mixed precision dtype
        from convert_to_quant.utils.hadamard import _get_mixed_precision_dtype
        mixed_dtype = _get_mixed_precision_dtype()
        dtype_name = "BF16" if mixed_dtype == torch.bfloat16 else "FP16"
        if streaming_enabled:
            info(f"Streaming mode enabled ({streaming_mode}) - using {dtype_name} Hadamard transform")
        else:
            info(f"Memory-efficient mode enabled - using {dtype_name} Hadamard transform")

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

    # Use unified loader (handles both standard, low-memory, and streaming modes)
    # streaming_mode != "off" or low_memory=True both enable streaming loader
    use_streaming_loader = low_memory or streaming_enabled
    try:
        loader = MemoryEfficientSafeOpen(input_file, low_memory=use_streaming_loader)
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
    converter_kwargs["low_memory"] = low_memory
    converter_kwargs["streaming_mode"] = streaming_mode
    converter_kwargs["streaming_thresholds"] = streaming_thresholds or {}
    converter_kwargs["no_memory_limits"] = no_memory_limits

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
    
    # Load LoRA for merging (if specified)
    lora_for_merge = {}
    all_lora_paths = []
    
    # Collect all LoRA paths
    if merge_lora_paths:
        all_lora_paths.extend(merge_lora_paths)
    if merge_lora_path:
        all_lora_paths.append(merge_lora_path)
    
    if all_lora_paths:
        if len(all_lora_paths) == 1:
            info(f"Loading LoRA for merging: {all_lora_paths[0]}")
            lora_for_merge = load_lora_for_merging(all_lora_paths[0])
        else:
            info(f"Loading {len(all_lora_paths)} LoRAs for merging:")
            for path in all_lora_paths:
                info(f"  - {path}")
            lora_for_merge = load_multiple_loras_for_merging(all_lora_paths)
        
        info(f"  Found {len(lora_for_merge)} unique layer names")
        if merge_lora_scale != 1.0:
            info(f"  LoRA merge scale: {merge_lora_scale}")
        if len(all_lora_paths) > 1 and merge_lora_dampen:
            info(f"  Dampening enabled for multiple LoRAs")
    
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

    # Check if async prefetching is enabled
    opt_config = get_optimization_config()
    use_async_prefetch = (
        opt_config.enable_async_prefetch and 
        device == "cuda" and 
        not use_streaming_loader  # Don't combine with streaming loader (conflicts)
    )
    
    if use_async_prefetch:
        info(f"Async tensor prefetching enabled ({opt_config.async_prefetch_workers} workers)")
        tensor_prefetcher = AsyncTensorPrefetcher(
            loader, 
            weight_keys, 
            device=device,
            max_workers=opt_config.async_prefetch_workers,
            enable_pin_memory=opt_config.enable_pinned_pool
        )
        tensor_iterator = tensor_prefetcher
    else:
        tensor_prefetcher = None
        tensor_iterator = enumerate(weight_keys)

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
        
        # Merge LoRA weights if specified and this layer has a matching LoRA
        if lora_for_merge and key.endswith(".weight") and original_tensor.ndim == 2:
            base_name = key.rsplit(".weight", 1)[0]
            norm_base = base_name.replace("model.diffusion_model.", "").replace("transformer.", "")
            
            if norm_base in lora_for_merge:
                lora_data_list = lora_for_merge[norm_base]
                
                # Check if it's a list (multiple LoRAs) or single dict
                if isinstance(lora_data_list, list):
                    # Multiple LoRAs - merge with dampening
                    info(f"      Merging {len(lora_data_list)} LoRAs for layer: {key}")
                    lora_configs = []
                    for i, lora_data in enumerate(lora_data_list):
                        scale = merge_lora_scale * (0.9 ** i) if merge_lora_dampen and i > 0 else merge_lora_scale
                        lora_configs.append({
                            "lora_A": lora_data["lora_A"],
                            "lora_B": lora_data["lora_B"],
                            "alpha": lora_data["alpha"],
                            "rank": lora_data["rank"],
                            "scale": scale
                        })
                    
                    original_tensor = merge_multiple_loras(
                        original_tensor.to(device=device, dtype=COMPUTE_DTYPE),
                        lora_configs
                    )
                else:
                    # Single LoRA
                    info(f"      Merging LoRA for layer: {key}")
                    original_tensor = merge_lora_into_weight(
                        original_tensor.to(device=device, dtype=COMPUTE_DTYPE),
                        lora_data_list["lora_A"],
                        lora_data_list["lora_B"],
                        lora_data_list["alpha"],
                        lora_data_list["rank"],
                        merge_lora_scale
                    )

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
                low_memory=low_memory,
                streaming=streaming_enabled
            )
            H = None
            in_features = original_tensor.shape[1]
            if in_features in calibration_data_cache:
                # Use CPU for Hessian calculation if in low_memory mode to save VRAM
                calc_device = "cpu" if (low_memory or in_features > 4096) else device
                X = calibration_data_cache[in_features].to(calc_device)
                
                # BF16 optimization: Use BF16 for matmul on large tensors, convert back to FP32
                from .constants import should_use_bf16_for_op
                if should_use_bf16_for_op(X.numel(), "hessian"):
                    with torch.autocast(device_type='cuda' if X.is_cuda else 'cpu', dtype=torch.bfloat16):
                        H = X.T @ X
                    H = H.float()  # Back to FP32 for Cholesky
                else:
                    H = (X.T @ X).float()
            q_tensor, dequant_s, dequant_w = gptq_converter.convert(original_tensor, H=H)
        elif optimizer_type == "quip" and is_int8:
            # Z-Image specific QuIP settings for Tensorwise INT8 compatibility
            quip_converter = QuIPInt8Converter(
                block_size=converter.block_size,
                device=device,
                actorder=quip_actorder,
                use_hadamard=quip_hadamard,
                seed=quip_seed,
                use_triton=gptq_turbo,
                lazy_updates=gptq_fast,
                store_transformed=quip_store_transformed,
                streaming_mode=streaming_mode,
                streaming_thresholds=streaming_thresholds or {},
                no_memory_limits=no_memory_limits,
                use_checkpointed_ldlq=quip_checkpointed,
                checkpointed_ldlq_threshold=quip_checkpoint_threshold,
                checkpoint_segments=quip_checkpoint_segments
            )
            quip_converter.low_memory = low_memory
            H = None
            in_features = original_tensor.shape[1]
            # Only use Hessian from real calibration data
            # Random synthetic data degrades quality, so without real calibration H=None (identity) is better
            if calibration_data_path and in_features in calibration_data_cache:
                # Use CPU for Hessian calculation if in low_memory mode to save VRAM
                calc_device = "cpu" if (low_memory or in_features > 4096) else device
                X = calibration_data_cache[in_features].to(calc_device)
                
                # BF16 optimization: Use BF16 for matmul on large tensors, convert back to FP32
                from .constants import should_use_bf16_for_op
                if should_use_bf16_for_op(X.numel(), "hessian"):
                    with torch.autocast(device_type='cuda' if X.is_cuda else 'cpu', dtype=torch.bfloat16):
                        H = X.T @ X
                    H = H.float()  # Back to FP32 for Cholesky
                else:
                    H = (X.T @ X).float()
            # QuIP now returns up to 9 values (including Hadamard-QuIP metadata)
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
            hadamard_quip = res[6] if len(res) > 6 else False
            hadamard_size_out = res[7] if len(res) > 7 else 0
            hadamard_size_in = res[8] if len(res) > 8 else 0
            if low_memory or streaming_enabled:
                quip_converter.cleanup()
        else:
            res = converter.convert(original_tensor, activation_scales=act_scales)
            q_tensor, dequant_s, dequant_w = res[0], res[1], res[2]
            smooth_factors = res[3] if len(res) > 3 else None
            s_u, s_v = None, None
            hadamard_quip = False
            hadamard_size_out = 0
            hadamard_size_in = 0

        new_tensors[key] = q_tensor.to(device="cpu")
        if smooth_factors is not None:
            new_tensors[f"{base_name}.smooth_factors"] = smooth_factors.to(device="cpu", dtype=SCALE_DTYPE)
        
        # Save QuIP/Hadamard-QuIP metadata
        if hadamard_quip and s_u is not None and s_v is not None:
            # Hadamard-QuIP format: save sign vectors with proper keys for INT8 inference
            new_tensors[f"{base_name}.hadamard_quip"] = torch.tensor(1, dtype=torch.int8)  # Flag as Hadamard-QuIP
            new_tensors[f"{base_name}.hadamard_size_out"] = torch.tensor(hadamard_size_out, dtype=torch.int32)
            new_tensors[f"{base_name}.hadamard_size_in"] = torch.tensor(hadamard_size_in, dtype=torch.int32)
            new_tensors[f"{base_name}.sign_row"] = s_u.to(device="cpu", dtype=SCALE_DTYPE)  # s_u is output signs (row)
            new_tensors[f"{base_name}.sign_col"] = s_v.to(device="cpu", dtype=SCALE_DTYPE)  # s_v is input signs (col)
        elif s_u is not None and s_v is not None:
            # Legacy QuIP# format (dense rotation matrices) - keep for backward compatibility
            new_tensors[f"{base_name}.quip_s_u"] = s_u.to(device="cpu", dtype=SCALE_DTYPE)
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
                # Determine scaling mode and format
                if optimizer_type == "quip":
                    # QuIP always uses tensor-wise scaling for the transformed weights
                    comfy_quant_format = "int8_tensorwise"
                    block_size_for_meta = None
                    # QuIP uses .weight_scale for Tensorwise INT8 compatibility
                    scale_key = f"{base_name}.weight_scale"
                elif converter.scaling_mode == "axis":
                    comfy_quant_format = "int8_axiswise"
                    block_size_for_meta = None
                    scale_key = f"{base_name}.weight_scale"
                elif converter.scaling_mode == "tensor":
                    comfy_quant_format = "int8_tensorwise"
                    block_size_for_meta = None
                    scale_key = f"{base_name}.weight_scale"
                else:
                    comfy_quant_format = "int8_blockwise"
                    block_size_for_meta = layer_block_size
                    scale_key = f"{base_name}.weight_scale"

                new_tensors[scale_key] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()

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

        # Aggressive cleanup between layers in streaming mode to prevent OOM accumulation
        if streaming_enabled and device == "cuda":
            torch.cuda.empty_cache()

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
