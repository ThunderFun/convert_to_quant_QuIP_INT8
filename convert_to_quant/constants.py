"""
Constants and configuration values for convert_to_quant.

Contains model-specific key name filters and dtype settings.
"""
import torch

# --- Model-specific exclusion lists (layers to skip quantization) ---
AVOID_KEY_NAMES = [
    "norm",
    "bias",
    "embed_tokens",
    "lm_head",
    "shared",
    "patch_embedding",
    "audio_model.patch_embedding",
    "ref_conv",
    "control_adapter",
    "motion_encoder.enc.net_app",
    "face_encoder.conv",
    "pose_patch_embedding",
    "motion_encoder.enc.fc",
    "img_emb.proj",
    "k_norm",
    "q_norm",
    "motion_encoder.dec",
    "head.modulation",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "norm_k",
    "norm_q",
    "tekken_model",
    "multi_modal_projector",
    "patch_conv",
    "ln_pre",
    "input_layernorm",
    "attention_norm",
    "post_attention_layernorm",
]
LORA_AVOID_KEY_NAMES = ["alpha", "scale"]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
VISUAL_AVOID_KEY_NAMES = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = [
    "layernorm",
    "img_attn_k_norm",
    "img_attn_q_norm",
    "txt_attn_k_norm",
    "txt_attn_q_norm",
    "norm1",
    "norm2",
    "vision_in.proj.0",
    "vision_in.proj.4",
    "img_in.proj",
    "cond_type_embedding",
]
ZIMAGE_AVOID_KEY_NAMES = [
    "cap_embedder.0",
    "cap_pad_token",
    "attention_norm1",
    "attention_norm2",
    "ffn_norm1",
    "ffn_norm2",
    "k_norm",
    "q_norm",
    "x_pad_token",
    "norm", # General norm exclusion
    "bias", # Biases should stay in high precision
]

# --- Layer key names for specific models (layers to include as high-precision) ---
FLUX2_LAYER_KEYNAMES = [
    "stream_modulation",
    "guidance_in",
    "time_in",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
    "txt_in",
]
NERF_LAYER_KEYNAMES_SMALL = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv", "__x0__"]
WAN_LAYER_KEYNAMES = [
    "text_embedding",
    "time_embedding",
    "audio_model.text_embedding",
    "casual_audio_encoder",
    "frame_packer",
    "trainable_cond_mask",
    "cond_encoder",
    "audio_model.time_embedding",
    "time_projection",
    "video_model.time_projection",
    "head.head",
    "face_encoder.out_proj",
    "face_adapter",
    "audio_injector",
]
QWEN_LAYER_KEYNAMES = [
    "time_text_embed",
    "img_in",
    "norm_out",
    "proj_out",
    "transformer_blocks.0.img_mod.1",
    "txt_in",
]
ZIMAGE_LAYER_KEYNAMES = [
    "x_embedder",
    "clip_text_pooled_proj",
    "final_layer",
    "cap_embedder.1",
    "adaLN_modulation",
    "t_embedder",
    "time_text_embed",
    "pos_embed", # Embeddings
    "proj_out", # Final output layers
]
ZIMAGE_REFINER_LAYER_KEYNAMES = ["context_refiner", "noise_refiner"]

# --- Model Filter Registry ---
MODEL_FILTERS = {
    # Text Encoders
    "t5xxl": {
        "help": "T5-XXL text encoder: skip norms/biases, remove decoder layers",
        "category": "text",
        "exclude": AVOID_KEY_NAMES,
        "remove": T5XXL_REMOVE_KEY_NAMES,
    },
    "mistral": {
        "help": "Mistral text encoder exclusions",
        "category": "text",
        "exclude": AVOID_KEY_NAMES,
    },
    "visual": {
        "help": "Visual encoder: skip MLP layers (down/up/gate proj)",
        "category": "text",
        "exclude": VISUAL_AVOID_KEY_NAMES,
    },
    # Diffusion Models (Flux-style)
    "flux2": {
        "help": "Flux.2: keep modulation/guidance/time/final layers high-precision",
        "category": "diffusion",
        "highprec": FLUX2_LAYER_KEYNAMES,
    },
    "distillation_large": {
        "help": "Chroma/distilled (large): keep distilled_guidance, final, img/txt_in high-precision",
        "category": "diffusion",
        "highprec": DISTILL_LAYER_KEYNAMES_LARGE,
    },
    "distillation_small": {
        "help": "Chroma/distilled (small): keep only distilled_guidance high-precision",
        "category": "diffusion",
        "highprec": DISTILL_LAYER_KEYNAMES_SMALL,
    },
    "nerf_large": {
        "help": "NeRF (large): keep nerf_blocks, distilled_guidance, txt_in high-precision",
        "category": "diffusion",
        "highprec": NERF_LAYER_KEYNAMES_LARGE,
    },
    "nerf_small": {
        "help": "NeRF (small): keep nerf_blocks, distilled_guidance high-precision",
        "category": "diffusion",
        "highprec": NERF_LAYER_KEYNAMES_SMALL,
    },
    "radiance": {
        "help": "Radiance model: keep img_in_patch, nerf_final_layer high-precision",
        "category": "diffusion",
        "highprec": RADIANCE_LAYER_KEYNAMES,
    },
    # Video Models
    "wan": {
        "help": "WAN video model: skip embeddings, encoders, head",
        "category": "video",
        "exclude": AVOID_KEY_NAMES,
        "highprec": WAN_LAYER_KEYNAMES,
    },
    "hunyuan": {
        "help": "Hunyuan Video 1.5: skip layernorm, attn norms, vision_in",
        "category": "video",
        "exclude": HUNYUAN_AVOID_KEY_NAMES,
    },
    # Image Models
    "qwen": {
        "help": "Qwen Image: skip added norms, keep time_text_embed high-precision",
        "category": "image",
        "exclude": QWEN_AVOID_KEY_NAMES,
        "highprec": QWEN_LAYER_KEYNAMES,
    },
    "zimage": {
        "help": "Z-Image: skip cap_embedder/norms, keep x_embedder/final high-precision",
        "category": "image",
        "exclude": ZIMAGE_AVOID_KEY_NAMES,
        "highprec": ZIMAGE_LAYER_KEYNAMES,
    },
    "zimage_refiner": {
        "help": "Z-Image Refiner: keep context/noise refiner high-precision",
        "category": "image",
        "exclude": ZIMAGE_AVOID_KEY_NAMES,
        "highprec": ZIMAGE_REFINER_LAYER_KEYNAMES,
    },
    "lora": {
        "help": "LoRA model: skip alpha/scale, quantize lora_up/down",
        "category": "diffusion",
        "exclude": LORA_AVOID_KEY_NAMES,
    },
}


def build_exclusion_patterns(active_filters: dict) -> tuple:
    """
    Build layer exclusion patterns from active filter flags.

    Args:
        active_filters: Dict of filter_name -> bool (e.g., {"radiance": True, "t5xxl": False})

    Returns:
        Tuple of (exclude_patterns, highprec_patterns, remove_patterns)
    """
    exclude = []
    highprec = []
    remove = []

    for name, cfg in MODEL_FILTERS.items():
        if active_filters.get(name, False):
            exclude.extend(cfg.get("exclude", []))
            highprec.extend(cfg.get("highprec", []))
            remove.extend(cfg.get("remove", []))

    return exclude, highprec, remove

# --- Dtype settings ---
TARGET_INT8_DTYPE = torch.int8
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# INT8 constants (using symmetric range [-127, 127] for symmetric quantization)
INT8_MIN = int(torch.iinfo(TARGET_INT8_DTYPE).min)  # -128
INT8_MAX = int(torch.iinfo(TARGET_INT8_DTYPE).max)  # 127
INT8_SYMMETRIC_MAX = min(abs(INT8_MIN), INT8_MAX)  # 127 (symmetric range)

# --- Adaptive LR Tier Configuration ---
# Used by 'original' optimizer in LearnedRoundingConverter.
ADAPTIVE_LR_TIERS_IMPROVE = [
    (0, 1.25, 100.0),
    (50, 1.375, 100.0),
    (75, 1.5, 100.0),
    (100, 1.75, 100.0),
    (125, 2.0, 100.0),
    (150, 2.25, 100.0),
    (200, 2.5, 100.0),
    (250, 2.75, 100.0),
    (300, 3.0, 100.0),
]

ADAPTIVE_LR_TIERS_DECAY = [
    (0, 0.95, 9e-8),
    (26, 0.97, 8e-8),
    (51, 0.985, 7e-8),
    (76, 0.9875, 6e-8),
    (101, 0.98875, 5e-8),
    (151, 0.99, 4e-8),
    (201, 0.99125, 3e-8),
    (251, 0.9925, 2e-8),
    (301, 0.995, 5e-9),
]

# Valid quantization formats (maps to QUANT_ALGOS in quant_ops.py)
VALID_QUANT_FORMATS = {
    "int8_blockwise",
    "int8_tensorwise",
    "int8_axiswise",
}

# Global config: normalize 1-element scale arrays to scalars (set from CLI)
NORMALIZE_SCALES_ENABLED = True


# =============================================================================
# BF16 Compute Mode Configuration
# =============================================================================

import os


def get_compute_dtype() -> torch.dtype:
    """
    Determine compute dtype based on environment and hardware.
    
    Modes:
    - "off"/"0"/"false": Force FP32
    - "on"/"1"/"true"/"force": Force BF16 (if supported)
    - "auto": Use BF16 on Ampere+ for eligible operations
    
    Returns:
        torch.dtype: Either torch.float32 or torch.bfloat16
    """
    bf16_mode = os.environ.get("BF16_COMPUTE_MODE", "auto")
    
    if bf16_mode in ("0", "off", "false"):
        return torch.float32
    
    if bf16_mode in ("1", "on", "true", "force"):
        return _get_bf16_if_supported()
    
    # Auto mode - will be decided per-operation
    return torch.float32  # Default, ops can override


def _get_bf16_if_supported() -> torch.dtype:
    """Check if BF16 is supported on current GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    
    major, minor = torch.cuda.get_device_capability()
    # Ampere (SM80+) and newer support BF16
    if major >= 8:
        return torch.bfloat16
    return torch.float32


def should_use_bf16_for_op(tensor_size: int, operation: str) -> bool:
    """
    Determine if BF16 should be used for a specific operation.
    
    Args:
        tensor_size: Number of elements in the tensor
        operation: Type of operation ("matmul", "hadamard", "hessian", etc.)
    
    Returns:
        True if BF16 is recommended for this operation
    """
    bf16_mode = os.environ.get("BF16_COMPUTE_MODE", "auto")
    
    if bf16_mode in ("0", "off", "false"):
        return False
    
    if bf16_mode in ("1", "on", "true", "force"):
        return _get_bf16_if_supported() == torch.bfloat16
    
    # Auto mode: Use BF16 for large tensors on supported hardware
    if _get_bf16_if_supported() != torch.bfloat16:
        return False
    
    # Read thresholds from environment with defaults
    thresholds = {
        "matmul": int(os.environ.get("BF16_MATMUL_THRESHOLD", "1000000")),
        "hadamard": int(os.environ.get("BF16_HADAMARD_THRESHOLD", "500000")),
        "hessian": int(os.environ.get("BF16_HESSIAN_THRESHOLD", "1000000")),
        "svd": int(os.environ.get("BF16_SVD_THRESHOLD", "2000000")),
        "ldlq": int(os.environ.get("BF16_LDLQ_THRESHOLD", "500000")),
    }
    
    return tensor_size >= thresholds.get(operation, 1000000)


def get_bf16_compute_mode() -> str:
    """Get the current BF16 compute mode setting."""
    return os.environ.get("BF16_COMPUTE_MODE", "auto")
