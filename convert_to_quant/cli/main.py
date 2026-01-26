"""
CLI main function for convert_to_quant.

Entry point that handles argument parsing and dispatches to appropriate conversion functions.
"""
import argparse
import os
import sys
import torch
from safetensors.torch import load_file, save_file

from .argument_parser import (
    MultiHelpArgumentParser,
    EXPERIMENTAL_ARGS,
    FILTER_ARGS,
    ADVANCED_ARGS,
)
from ..constants import (
    NORMALIZE_SCALES_ENABLED,
    MODEL_FILTERS,
)
from ..config.layer_config import load_layer_config, generate_config_template
from ..quantization import convert_to_int8
from ..utils.comfy_quant import edit_comfy_quant
from ..pinned_transfer import set_verbose as set_pinned_verbose
import json
from safetensors import safe_open

def load_input_scales(path: str) -> dict:
    """Load input scales from JSON or safetensors file.

    Args:
        path: Path to JSON file or safetensors model with .input_scale tensors

    Returns:
        Dict mapping layer base names to input_scale values (float)
    """
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif path.endswith('.safetensors'):
        scales = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith('.input_scale'):
                    base = key.rsplit('.input_scale', 1)[0]
                    scales[base] = f.get_tensor(key).item()
        return scales
    else:
        raise ValueError(f"Unsupported input scales format: {path}. Use .json or .safetensors")

def extract_filter_flags(args) -> dict:
    """Extract model filter flags from parsed args with validation."""
    flags = {}
    for name in MODEL_FILTERS.keys():
        if not hasattr(args, name):
            raise RuntimeError(
                f"BUG: Filter '{name}' in MODEL_FILTERS but not in argparse. "
                f"Add --{name} to argument_parser.py"
            )
        if getattr(args, name):
            flags[name] = True
    return flags

from ..utils.logging import setup_logging, info, minimal, warning

def main():
    parser = MultiHelpArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert safetensors weights to INT8 format.\n\n"
        "Default behavior: INT8 block-wise quantization.\n"
        "For experimental options, see --help-experimental.\n"
        "For model-specific layer exclusions, see --help-filters.\n"
        "For advanced LR tuning and early stopping, see --help-advanced.",
        experimental_args=EXPERIMENTAL_ARGS,
        filter_args=FILTER_ARGS,
        advanced_args=ADVANCED_ARGS,
    )

    parser.add_argument("-i", "--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("-o", "--output", type=str, help="Output safetensors file path.")
    parser.add_argument("--comfy_quant", action="store_true", help="Use Comfy quantization method.")
    parser.add_argument("--int8", action="store_true", default=True, help="Use INT8 block-wise quantization (default).")
    parser.add_argument("--smoothquant", action="store_true", help="Enable SmoothQuant preprocessing.")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5, help="SmoothQuant migration strength.")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16.")
    parser.add_argument("--fallback", type=str, default=None, choices=["int8", "fp16"], help="Fallback quantization type.")
    parser.add_argument("--custom-layers", type=str, default=None, help="Regex pattern for custom quantization.")
    parser.add_argument("--exclude-layers", type=str, default=None, help="Regex pattern for layer exclusion.")
    parser.add_argument("--custom-type", type=str, default=None, choices=["int8", "fp16"], help="Quantization type for custom layers.")
    parser.add_argument("--custom-block-size", type=int, default=None, help="Block size for custom layers.")
    parser.add_argument("--custom-scaling-mode", type=str, default=None, choices=["tensor", "block", "axis"], help="Scaling mode for custom layers.")
    parser.add_argument("--custom-simple", action="store_true", help="Use simple quantization for custom layers.")
    parser.add_argument("--custom-heur", action="store_true", help="Apply heuristics to custom layers.")
    parser.add_argument("--fallback-block-size", type=int, default=None, help="Block size for fallback layers.")
    parser.add_argument("--fallback-simple", action="store_true", help="Use simple quantization for fallback layers.")
    parser.add_argument("--simple", action="store_true", help="Skip SVD optimization.")
    parser.add_argument("--full_precision_matrix_mult", action="store_true", help="Set full_precision_matrix_mult=True.")
    parser.add_argument("--heur", action="store_true", help="Skip inefficient layers.")
    parser.add_argument("--input_scale", action="store_true", help="Include input_scale tensor.")
    parser.add_argument("--static-activations", action="store_true", help="Enable static activation quantization. Requires --calibration-data with input_scale values.")
    parser.add_argument("--verbose", type=str, default="NORMAL", choices=["DEBUG", "VERBOSE", "NORMAL", "MINIMAL"], help="Set verbosity.")

    for filter_name, filter_cfg in MODEL_FILTERS.items():
        parser.add_argument(f"--{filter_name}", action="store_true", help=filter_cfg.get("help"))

    parser.add_argument("--full_matrix", action="store_true", help="Use full matrices for SVD.")
    parser.add_argument("--scaling_mode", type=str, default="block", choices=["tensor", "block", "axis"], help="Scaling mode.")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for INT8.")
    parser.add_argument("--calib_samples", type=int, default=6144, help="Number of random samples.")
    parser.add_argument("--manual_seed", type=int, default=-1, help="Manual seed.")
    parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "radam", "gptq", "quip"], help="Optimization algorithm.")
    parser.add_argument("--num_iter", type=int, default=1000, help="Optimization iterations.")
    parser.add_argument("--lr", type=float, default=8.077300000003e-3, help="Initial learning rate.")
    parser.add_argument("--lr_schedule", type=str, default="adaptive", choices=["adaptive", "exponential", "plateau"], help="LR schedule.")
    parser.add_argument("--lr_gamma", type=float, default=0.99, help="Decay factor.")
    parser.add_argument("--lr_patience", type=int, default=9, help="Steps before decay.")
    parser.add_argument("--lr_factor", type=float, default=0.92, help="LR reduction factor.")
    parser.add_argument("--lr_min", type=float, default=1e-10, help="Minimum LR.")
    parser.add_argument("--lr_cooldown", type=int, default=6, help="Steps to wait after reduction.")
    parser.add_argument("--lr_threshold", type=float, default=0.0, help="Min improvement.")
    parser.add_argument("--lr_adaptive_mode", type=str, default="simple-reset", choices=["simple-reset", "no-reset"], help="Adaptive mode.")
    parser.add_argument("--lr-shape-influence", type=float, default=1.0, help="Aspect ratio influence.")
    parser.add_argument("--lr-threshold-mode", type=str, default="rel", choices=["rel", "abs"], help="Threshold mode.")
    parser.add_argument("--early-stop-loss", type=float, default=1e-8, help="Early stop loss.")
    parser.add_argument("--early-stop-lr", type=float, default=1e-10, help="Early stop LR.")
    parser.add_argument("--early-stop-stall", type=int, default=1000, help="Early stop stall.")
    parser.add_argument("--top_p", type=float, default=0.2, help="SVD top_p.")
    parser.add_argument("--min_k", type=int, default=64, help="SVD min_k.")
    parser.add_argument("--max_k", type=int, default=1024, help="SVD max_k.")
    parser.add_argument("--save-quant-metadata", action="store_true", help="Save metadata in header.")
    parser.add_argument("--no-normalize-scales", action="store_true", help="Disable scale normalization.")
    parser.add_argument("--report-quality", action="store_true", help="Output quality metrics.")
    parser.add_argument("--quality-threshold", type=float, default=30.0, help="SQNR threshold.")
    parser.add_argument("--calibration-data", type=str, help="Path to calibration data.")
    parser.add_argument("--calibration-lora", type=str, help="Path to LoRA file for informed calibration.")
    parser.add_argument("--edit-quant", action="store_true", help="Edit .comfy_quant tensors.")
    parser.add_argument("--remove-keys", type=str, help="Keys to remove.")
    parser.add_argument("--add-keys", type=str, help="Keys to add.")
    parser.add_argument("--quant-filter", type=str, help="Filter for editing.")
    parser.add_argument("--layer-config", type=str, help="Path to layer config JSON.")
    parser.add_argument("--fullmatch", action="store_true", help="Use fullmatch for layer config.")
    parser.add_argument("--dry-run", type=str, nargs="?", const="analyze", choices=["analyze", "create-template"], help="Dry run mode.")
    parser.add_argument("--verbose-pinned", action="store_true", help="Verbose pinned memory.")
    parser.add_argument("--low-memory", action="store_true", help="Low memory mode.")
    parser.add_argument("--gptq-actorder", action="store_true", help="Enable GPTQ activation ordering.")
    parser.add_argument("--gptq-fast", action="store_true", default=True, help="Enable GPTQ vectorized processing.")
    parser.add_argument("--gptq-turbo", action="store_true", help="Enable GPTQ Triton kernel.")
 
    parser.add_argument("--quip-actorder", action="store_true", default=True, help="Enable activation ordering for QuIP.")
    parser.add_argument("--quip-hadamard", action="store_true", default=True, help="Use Hadamard transform for QuIP.")
    parser.add_argument("--quip-seed", type=int, default=None, help="Seed for QuIP random orthogonal matrices.")
 
    args = parser.parse_args()
    setup_logging(args.verbose)
    global NORMALIZE_SCALES_ENABLED
    NORMALIZE_SCALES_ENABLED = not args.no_normalize_scales
    set_pinned_verbose(args.verbose_pinned)

    if args.static_activations and not args.calibration_data:
        warning("WARNING: --static-activations requires calibration data with input_scale values.")
        warning("         Run calibrate_activation_scales first, or omit this flag for dynamic quantization.")

    if args.dry_run == "create-template":
        template_path = os.path.splitext(args.input)[0] + "_layer_config_template.json"
        generate_config_template(args.input, template_path, block_size=args.block_size or 128)
        return

    if args.edit_quant:
        remove_keys_list = [k.strip() for k in args.remove_keys.split(",") if k.strip()] if args.remove_keys else None
        edit_comfy_quant(args.input, args.output or f"{os.path.splitext(args.input)[0]}_edited.safetensors",
                         remove_keys=remove_keys_list, add_keys_str=args.add_keys,
                         layer_filter=args.quant_filter, save_quant_metadata=args.save_quant_metadata)
        return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        prefix = "simple_" if args.simple else "learned_"
        fmt = "int8" if not args.fp16 else "fp16"
        scaling = f"_bs{args.block_size}" if fmt == "int8" else ""
        filter_flags = extract_filter_flags(args)
        mixed = "mixed" if (any(filter_flags.values()) or args.custom_layers) else ""
        args.output = f"{base}_{prefix}{fmt}{mixed}{scaling}.safetensors"

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("Error: Output file cannot be same as input.")
        return

    seed = int(torch.randint(0, 2**32 - 1, ()).item()) if args.manual_seed == -1 else args.manual_seed
    layer_config_data = load_layer_config(args.layer_config) if args.layer_config else None
    filter_flags = extract_filter_flags(args)

    convert_to_int8(
        args.input, args.output, args.comfy_quant, filter_flags, args.calib_samples, seed,
        fp16=args.fp16, fallback=args.fallback, custom_layers=args.custom_layers,
        exclude_layers=args.exclude_layers, custom_type=args.custom_type,
        custom_block_size=args.custom_block_size, custom_scaling_mode=args.custom_scaling_mode,
        custom_simple=args.custom_simple, custom_heur=args.custom_heur,
        fallback_block_size=args.fallback_block_size, fallback_simple=args.fallback_simple,
        full_precision_matrix_mult=args.full_precision_matrix_mult,
        skip_inefficient_layers=args.heur, include_input_scale=args.input_scale,
        no_learned_rounding=args.simple, save_quant_metadata=args.save_quant_metadata,
        layer_config=layer_config_data, layer_config_fullmatch=args.fullmatch,
        low_memory=args.low_memory, report_quality=args.report_quality,
        quality_threshold=args.quality_threshold, smoothquant=args.smoothquant,
        smoothquant_alpha=args.smoothquant_alpha, calibration_data_path=args.calibration_data,
        calibration_lora_path=args.calibration_lora,
        gptq_actorder=args.gptq_actorder, gptq_fast=args.gptq_fast, gptq_turbo=args.gptq_turbo,
        optimizer=args.optimizer, num_iter=args.num_iter, lr=args.lr, lr_schedule=args.lr_schedule,
        quip_actorder=args.quip_actorder, quip_hadamard=args.quip_hadamard, quip_seed=args.quip_seed,
        top_p=args.top_p, min_k=args.min_k, max_k=args.max_k, full_matrix=args.full_matrix,
        scaling_mode=args.scaling_mode, block_size=args.block_size, lr_gamma=args.lr_gamma,
        lr_patience=args.lr_patience, lr_factor=args.lr_factor, lr_min=args.lr_min,
        lr_cooldown=args.lr_cooldown, lr_threshold=args.lr_threshold,
        lr_adaptive_mode=args.lr_adaptive_mode, lr_shape_influence=args.lr_shape_influence,
        lr_threshold_mode=args.lr_threshold_mode, early_stop_loss=args.early_stop_loss,
        early_stop_lr=args.early_stop_lr, early_stop_stall=args.early_stop_stall
    )

if __name__ == "__main__":
    main()
