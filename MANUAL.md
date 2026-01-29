# Learned Rounding Quantization Converter - Manual

A tool for converting safetensors model weights to INT8 quantized format with optional SVD-based learned rounding optimization.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Command-Line Arguments](#command-line-arguments)
  - [Required Arguments](#required-arguments)
  - [Output Format Options](#output-format-options)
  - [Quantization Options](#quantization-options)
  - [Model-Specific Filters](#model-specific-filters)
  - [Optimization Options](#optimization-options)
  - [Streaming & Memory Options](#streaming--memory-options)
  - [BF16 Compute Options](#bf16-compute-options)
  - [Advanced Options](#advanced-options)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Model-Specific Guidance](#model-specific-guidance)
- [Troubleshooting](#troubleshooting)

---

## Overview

This script converts neural network weights from full precision (FP16/FP32/BF16) to INT8 quantized format:

| Format | Description | Use Case |
|--------|-------------|----------|
| **INT8** (block-wise) | 8-bit integer with per-block scaling | Broader hardware support, good compression, fast inference |

### Key Features

- **Unified Safetensors Loader**: Memory-efficient streaming loader with standard and low-memory modes
- **Learned rounding optimization**: Uses SVD-based optimization to minimize quantization error
- **GPTQ Optimizer**: Sequential layer-wise optimization with error compensation
- **QuIP (Quantization with Incoherence Processing)**: Near-lossless INT8 quantization using random orthogonal transformations. Defaults to standard `int8_tensorwise` format for maximum compatibility and inference performance
- **SmoothQuant**: Preprocessing to migrate quantization difficulty from activations to weights
- **Multiple optimizer choices**: Original, AdamW, RAdam, GPTQ, QuIP
- **Bias correction**: Automatically adjusts biases to compensate for quantization error
- **Model-specific filters**: Keep sensitive layers in high precision for various architectures
- **ComfyUI compatible**: Generates `.comfy_quant` metadata for seamless integration
- **BF16 Compute Mode**: Half-precision computation on Ampere+ GPUs for 2× memory savings
- **Checkpointed Quantization**: Extreme memory savings (75-90%) for large layers
- **Streaming Modes**: Configurable CPU/GPU offloading with auto-detection based on VRAM

---

## Requirements

Install dependencies:

```bash
pip install torch safetensors tqdm
```

For INT8, any CUDA GPU or CPU works. Triton is recommended for GPU acceleration.

---

## Quick Start

### Basic INT8 conversion (ComfyUI compatible)

```bash
convert_to_quant -i model.safetensors --comfy_quant
```

### Low VRAM / Memory-efficient mode

```bash
convert_to_quant -i model.safetensors --comfy_quant --low-memory
```

### Fast conversion (no optimization)

```bash
convert_to_quant -i model.safetensors --comfy_quant --simple
```

### INT8 with SmoothQuant and Quality Reporting

```bash
convert_to_quant -i model.safetensors --smoothquant --report-quality --comfy_quant
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i`, `--input` | Path to input safetensors file |

### Output Format Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-o`, `--output` | Auto-generated | Output file path. If not specified, generates a descriptive filename |
| `--comfy_quant` | False | Enable ComfyUI-compatible quantization format (adds `.comfy_quant` metadata) |
| `--full_precision_matrix_mult` | False | Add `full_precision_matrix_mult=True` to `.comfy_quant` metadata |
| `--save-quant-metadata` | False | Save metadata in header |

### Quantization Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--scaling_mode` | `block` | Scale computation mode: `block` (2D tiles), `axis` (per-row), or `tensor` (per-tensor) |
| `--block_size` | `128` | Block size for block-wise quantization. Not required for `tensor` or `axis` modes. |
| `--simple` | False | Skip SVD optimization, use simple round-to-nearest |
| `--heur` | False | Skip layers with poor quantization characteristics |
| `--smoothquant` | False | Enable SmoothQuant preprocessing to migrate quantization difficulty from activations to weights |
| `--smoothquant-alpha` | `0.5` | SmoothQuant migration strength (0.0=all to weight, 1.0=all to activation) |
| `--calibration-data` | None | Path to calibration statistics file (.safetensors) containing per-channel activation stats |
| `--report-quality` | False | Output quality metrics (MSE, SQNR) after conversion |
| `--quality-threshold` | `30.0` | SQNR threshold for warnings in dB |
| `--fp16` | False | Convert to FP16 instead of INT8 |
| `--low-memory` | False | Use streaming loader to reduce RAM usage by ~50% (slower but memory-efficient, forces CPU) |
| `--static-activations` | False | Enable static activation quantization (requires `--calibration-data` with input_scale values) |

### Model-Specific Filters

These flags keep certain layers in high precision (not quantized):

#### Text Encoders

| Argument | Description |
|----------|-------------|
| `--t5xxl` | T5-XXL text encoder (removes decoder, keeps norm/bias layers high precision) |
| `--mistral` | Mistral text encoder exclusions |
| `--visual` | Visual encoder: skip MLP layers (down/up/gate proj) |

#### Diffusion Models (Flux-style)

| Argument | Description |
|----------|-------------|
| `--flux2` | Flux.2: keep modulation/guidance/time/final layers high-precision |
| `--lora` | LoRA: skip alpha/scale, quantize lora_up/down |
| `--distillation_large` | Chroma/distilled (large): keep distilled_guidance, final, img/txt_in high-precision |
| `--distillation_small` | Chroma/distilled (small): keep only distilled_guidance high-precision |
| `--nerf_large` | NeRF (large): keep nerf_blocks, distilled_guidance, txt_in high-precision |
| `--nerf_small` | NeRF (small): keep nerf_blocks, distilled_guidance high-precision |
| `--radiance` | Radiance model: keep img_in_patch, nerf_final_layer high-precision |

#### Video Models

| Argument | Description |
|----------|-------------|
| `--wan` | WAN video model: skip embeddings, encoders, head |
| `--hunyuan` | Hunyuan Video 1.5: skip layernorm, attn norms, vision_in |

#### Image Models

| Argument | Description |
|----------|-------------|
| `--qwen` | Qwen Image: skip added norms, keep time_text_embed high-precision |
| `--zimage` | Z-Image: skip cap_embedder/norms, keep x_embedder/final high-precision |
| `--zimage_refiner` | Z-Image refiner: keep context/noise refiner high-precision |

### Optimization Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `original` | Optimization algorithm: `original`, `adamw`, `radam`, `gptq`, `quip` |
| `--num_iter` | `1000` | Maximum optimization iterations per tensor |
| `--lr` | `0.008077` | Learning rate for optimizers |
| `--top_p` | `0.2` | Proportion of SVD principal components to use |
| `--min_k` | `64` | Minimum number of SVD components |
| `--max_k` | `1024` | Maximum number of SVD components |
| `--full_matrix` | False | Use full SVD instead of low-rank approximation |
| `--quip-actorder` | True | Enable activation ordering for QuIP |
| `--quip-hadamard` | True | Use Hadamard transform for QuIP |
| `--quip-seed` | None | Seed for QuIP random orthogonal matrices |
| `--quip-store-transformed` | False | Store QuIP weights in transformed space (experimental, requires custom loader) |
| `--quip-checkpointed` | False | Enable checkpointed LDLQ quantization for 75-90% memory reduction on large layers |
| `--quip-checkpoint-threshold` | `8192` | Dimension threshold to use checkpointed quantization |
| `--quip-checkpoint-segments` | `4` | Number of segments for checkpointed quantization (higher = more memory savings but slower) |

**Note:** By default, QuIP stores weights in original space with a single global scale (`store_transformed=False`), ensuring full compatibility with standard inference pipelines including Z-Image and ComfyUI.

### Learning Rate Schedule Options

These options control the learning rate schedule for the `--optimizer original` algorithm:

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr_schedule` | `adaptive` | Schedule type: `adaptive`, `exponential`, or `plateau` |
| `--lr_gamma` | `0.99` | [exponential] Multiplicative decay factor per step |
| `--lr_patience` | `9` | [plateau] Steps without improvement before LR reduction |
| `--lr_factor` | `0.92` | [plateau] Factor to multiply LR by when reducing |
| `--lr_min` | `1e-10` | [plateau] Lower bound on learning rate |
| `--lr_cooldown` | `6` | [plateau] Steps to wait after LR reduction before resuming monitoring |
| `--lr_threshold` | `0.0` | [plateau] Minimum improvement to count as "significant" |
| `--lr_adaptive_mode` | `simple-reset` | [adaptive] Counter reset behavior: `simple-reset` or `no-reset` |

### LoRA Merging Options

Merge LoRA weights into the base model before quantization:

| Argument | Default | Description |
|----------|---------|-------------|
| `--merge-lora` | None | Path to LoRA file to merge into model weights before quantization |
| `--merge-loras` | None | Multiple LoRA files to merge (space-separated paths). Automatically applies dampening. |
| `--merge-lora-scale` | `1.0` | Scale factor applied to LoRA weights during merging |
| `--merge-lora-dampen` | True | Apply dampening when merging multiple LoRAs (reduces each subsequent LoRA's contribution) |

**Note:** When merging multiple LoRAs, the first LoRA uses full scale, and subsequent LoRAs are dampened by 0.9^(index) to prevent over-saturation.

### Streaming & Memory Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--streaming-mode` | `balanced` | Streaming mode for memory-efficient processing: `off`, `minimal`, `balanced`, `aggressive`, `auto` |
| `--stream-hadamard-threshold` | None | Override: Hadamard transform element threshold |
| `--stream-hessian-threshold` | None | Override: Hessian inversion dimension threshold |
| `--stream-ldlq-threshold` | None | Override: LDLQ quantization dimension threshold |
| `--stream-ortho-threshold` | None | Override: Orthogonal matrix generation dimension threshold |
| `--no-memory-limits` | False | Disable all memory limits and OOM prevention. Forces GPU processing with no safety checks. Use with caution. |

**Streaming Mode Comparison:**

| Mode | Hadamard Threshold | Behavior | Use Case |
|------|-------------------|----------|----------|
| `off` | ∞ (infinity) | Never offload to CPU | Workstations with 24GB+ VRAM |
| `minimal` | 100M elements (~400MB) | Conservative offloading | 12-16GB VRAM |
| `balanced` | 50M elements (~200MB) | Moderate offloading | 8-12GB VRAM |
| `aggressive` | 25M elements (~100MB) | Aggressive offloading | <8GB VRAM or maximum safety |
| `auto` | Adaptive | Detects based on VRAM | Recommended default |

### BF16 Compute Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--bf16-compute` | `auto` | Enable BF16 compute mode for faster quantization on Ampere+ GPUs: `auto`, `on`, `off` |
| `--bf16-threshold` | `1000000` | Minimum tensor size (elements) to use BF16 in auto mode |
| `--bf16-hadamard-threshold` | `500000` | Minimum tensor size (elements) for BF16 Hadamard transform |
| `--bf16-hessian-threshold` | `1000000` | Minimum tensor size (elements) for BF16 Hessian calculation |

**Note:** BF16 requires Ampere (SM80+) or newer GPU (RTX 30 series, A100, etc.)

### Advanced Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr-shape-influence` | `1.0` | Aspect ratio influence on learning rate (plateau schedule) |
| `--lr-threshold-mode` | `rel` | Threshold mode: `rel` (relative) or `abs` (absolute) |
| `--early-stop-loss` | `1e-8` | Early stopping loss threshold |
| `--early-stop-lr` | `1e-10` | Early stopping learning rate threshold |
| `--early-stop-stall` | `1000` | Early stopping stall threshold (iterations without improvement) |

---

## Usage Examples

### Example 1: Flux model with distillation layers preserved

```bash
convert_to_quant -i flux1-dev.safetensors \
    --comfy_quant \
    --distillation_large \
    --optimizer original \
    --num_iter 1000
```

### Example 2: T5-XXL text encoder

```bash
convert_to_quant -i t5xxl.safetensors \
    --comfy_quant \
    --t5xxl \
    --scaling_mode block \
    --block_size 128
```

### Example 3: INT8 with AdamW optimizer

```bash
convert_to_quant -i model.safetensors \
    --comfy_quant \
    --block_size 128 \
    --optimizer adamw \
    --num_iter 300
```

### Example 4: Axis-wise INT8

```bash
convert_to_quant -i model.safetensors \
    --scaling_mode axis \
    --comfy_quant
```

### Example 5: GPTQ-style INT8 with Automatic Calibration

```bash
# Basic GPTQ with internal synthetic calibration
convert_to_quant -i model.safetensors --optimizer gptq --comfy_quant

# GPTQ with LoRA-informed internal calibration (best results)
convert_to_quant -i model.safetensors --optimizer gptq --calibration-lora my_lora.safetensors --comfy_quant

# Using pre-computed calibration data from a file
convert_to_quant -i model.safetensors --optimizer gptq --calibration-data calib.safetensors --comfy_quant
```

### Example 6: QuIP-style INT8 (Near-Lossless)

```bash
# Basic QuIP conversion (default: stores in original space for compatibility)
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant

# QuIP with SmoothQuant for maximum accuracy
convert_to_quant -i model.safetensors --optimizer quip --smoothquant --comfy_quant

# Experimental: Store in transformed space (requires custom loader)
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant --quip-store-transformed

# Checkpointed quantization for extreme memory savings
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant --quip-checkpointed
```

**QuIP Storage Modes:**

| Mode | Flag | Format | Compatibility | Use Case |
|------|------|--------|---------------|----------|
| **Original Space** | *(default)* | `int8_tensorwise` | Full (Z-Image, ComfyUI, etc.) | **Recommended** for all standard inference |
| **Transformed Space** | `--quip-store-transformed` | Custom | Requires custom loader | Experimental research only |

### Example 7: LoRA Merging

```bash
# Merge single LoRA into base model before quantization
convert_to_quant -i base_model.safetensors \
    --merge-lora my_style_lora.safetensors \
    --optimizer quip \
    --comfy_quant

# Merge multiple LoRAs with automatic dampening
convert_to_quant -i base_model.safetensors \
    --merge-loras style_lora.safetensors character_lora.safetensors \
    --optimizer quip \
    --comfy_quant

# Adjust merge scale for subtler effect
convert_to_quant -i base_model.safetensors \
    --merge-lora my_lora.safetensors \
    --merge-lora-scale 0.7 \
    --comfy_quant
```

### Example 8: Streaming Modes

```bash
# Auto-detect based on VRAM (recommended)
convert_to_quant -i model.safetensors --optimizer quip --streaming-mode=auto --comfy_quant

# Balanced streaming (balanced speed/memory)
convert_to_quant -i model.safetensors --optimizer quip --streaming-mode=balanced --comfy_quant

# Aggressive streaming (faster, requires 12GB+ VRAM)
convert_to_quant -i model.safetensors \
    --optimizer quip \
    --streaming-mode=aggressive \
    --comfy_quant

# Custom thresholds with aggressive mode
convert_to_quant -i model.safetensors \
    --optimizer quip \
    --streaming-mode=aggressive \
    --stream-hadamard-threshold 50000000 \
    --comfy_quant
```

**Streaming Mode Comparison:**

| Mode | VRAM Usage | Speed | Recommended For |
|------|------------|-------|-----------------|
| `--streaming-mode=off` | High | Fastest | 24GB+ VRAM |
| `--streaming-mode=minimal` | Medium-High | Fast | 12-16GB VRAM |
| `--streaming-mode=balanced` | Medium | Medium | 8-12GB VRAM |
| `--streaming-mode=aggressive` | Low | Medium-Slow | <8GB VRAM |
| `--streaming-mode=auto` | Adaptive | Adaptive | All (recommended) |

### Example 9: BF16 Compute Mode

```bash
# Auto-enable BF16 for large tensors (default)
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=auto

# Force BF16 for all operations
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=on

# BF16 with custom thresholds
convert_to_quant -i model.safetensors --comfy_quant \
    --bf16-compute=auto \
    --bf16-threshold 1000000 \
    --bf16-hadamard-threshold 500000
```

### Example 10: Maximum Memory Safety

```bash
# For 8GB GPUs - maximum memory safety
convert_to_quant -i model.safetensors --comfy_quant \
    --optimizer quip \
    --streaming-mode=aggressive \
    --bf16-compute=on \
    --quip-checkpointed
```

### Example 11: Maximum Performance (No Safety)

```bash
# For 24GB+ workstations - maximum speed, no safety checks
convert_to_quant -i model.safetensors --comfy_quant \
    --optimizer quip \
    --streaming-mode=off \
    --no-memory-limits \
    --bf16-compute=on
```

---

## Unified Safetensors Loader

The `UnifiedSafetensorsLoader` class provides a consistent interface for loading safetensors files with two distinct modes of operation.

### Loading Modes

| Mode | Flag | Behavior | Memory Profile | Calculation Device |
|------|------|----------|----------------|-------------------|
| **Standard** | *(default)* | Preloads all tensors on initialization | ~2x model size in RAM | GPU |
| **Low-Memory** | `--low-memory` | Streams tensors on-demand | ~1x model size + 1 tensor | CPU (saves VRAM) |
| **Streaming** | `--streaming-mode` | Streams tensors on-demand with adaptive GPU/CPU selection | ~1x model size + 1 tensor | GPU/CPU (adaptive) |

### How It Works

**Standard Mode:**
- Loads all tensors from the safetensors file at initialization time
- Stores tensors in an internal dictionary for fast access
- Tensors persist in memory until the loader is closed
- Best for: Systems with sufficient RAM, fastest processing

**Low-Memory Mode:**
- Reads only the file header at initialization (contains tensor metadata)
- Keeps the file handle open for streaming access
- Loads individual tensors on-demand via `get_tensor()`
- Forces CPU for calculations to save VRAM (slower)
- Tensors should be deleted after processing to free memory
- Best for: Very limited RAM and VRAM environments

**Streaming Mode:**
- Same memory-efficient loading as Low-Memory mode
- Uses adaptive thresholds to decide CPU vs GPU for each operation
- Keeps calculations on GPU when possible for faster processing
- Recommended for: QuIP quantization where speed matters
- Best balance of memory efficiency and speed

### Programmatic Usage

```python
from convert_to_quant.utils.memory_efficient_loader import UnifiedSafetensorsLoader

# Low-memory streaming mode
with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    # Get metadata without loading tensors
    metadata = loader.metadata()
    
    # Iterate through all tensor keys
    for key in loader.keys():
        # Get tensor shape without loading data
        shape = loader.get_shape(key)
        ndim = loader.get_ndim(key)
        
        # Load tensor on-demand
        tensor = loader.get_tensor(key)
        
        # ... process tensor ...
        
        # Free memory when done (optional in low_memory mode)
        loader.mark_processed(key)

# Standard preloading mode
with UnifiedSafetensorsLoader("model.safetensors", low_memory=False) as loader:
    # All tensors already loaded
    tensor = loader.get_tensor("some_key")
    # Fast access, but uses more memory
```

### API Reference

| Method | Description |
|--------|-------------|
| `keys()` | Returns list of all tensor keys in the file |
| `metadata()` | Returns file metadata dictionary |
| `get_shape(key)` | Gets tensor shape without loading data |
| `get_ndim(key)` | Gets tensor dimensionality without loading data |
| `get_tensor(key)` | Loads and returns the tensor |
| `mark_processed(key)` | Marks tensor for cleanup (frees memory in standard mode) |
| `close()` | Closes file handles and releases resources |

### Supported Data Types

The loader supports all safetensors data types:
- Float: `F64`, `F32`, `F16`, `BF16`, `F8_E5M2`, `F8_E4M3`
- Integer: `I64`, `I32`, `I16`, `I8`, `U8`
- Boolean: `BOOL`

---

### Static Activation Quantization

Static activation quantization pre-computes input scales during calibration, eliminating the need for dynamic scale computation at inference time. This can improve performance but requires calibration data.

**Prerequisites:**
1. Generate calibration data using the calibration script
2. Calibration data must contain `input_scale` values for layers

**Usage:**

```bash
# First, generate calibration data
calibrate_activation_scales -i model.safetensors -o calibration.safetensors

# Then quantize with static activations
convert_to_quant -i model.safetensors \
    --calibration-data calibration.safetensors \
    --static-activations \
    --comfy_quant
```

**Benefits:**
- Faster inference - No dynamic scale computation
- More consistent quantization behavior
- Better performance for deployment

**Note:** If `--static-activations` is specified without calibration data, a warning is issued and dynamic quantization is used as fallback.

---

## How It Works

### Quantization Process

1. **Load weights**: Uses `UnifiedSafetensorsLoader` to read tensors (standard or streaming mode)
2. **Filter layers**: Apply model-specific exclusions (norm, bias, embeddings, etc.)
3. **For each weight tensor**:
   - Compute scaling factors (per-block, per-row, or per-tensor)
   - Initial quantization (round to nearest)
   - **Learned rounding optimization** (if enabled):
     - Compute low-rank SVD of the weight matrix
     - Optimize quantized values to minimize error in SVD subspace
   - Apply bias correction using synthetic calibration data
4. **Save**: Write quantized tensors with scale factors and metadata

### INT8 Block-wise Quantization Math

INT8 uses symmetric range `[-127, 127]` with **2D block-wise scaling**.

For a weight matrix `W` with shape `(M, N)` and block size `bs`:

**Block structure:**

```
W_blocked = W.reshape(M//bs, bs, N//bs, bs).permute(0, 2, 1, 3)
# Shape: (M//bs, N//bs, bs, bs)
```

**Per-block scale computation:**

```
amax = max(|W_blocked|, dim=(-2, -1))  # Shape: (M//bs, N//bs)
scale = amax / 127
```

**Quantization:**

```
Q_blocked = round(clamp(W_blocked / scale, -127, 127))
Q = Q_blocked.permute(0, 2, 1, 3).reshape(M, N)
```

### QuIP Quantization

QuIP (Quantization with Incoherence Processing) uses random orthogonal transformations to make weights more quantization-friendly:

1. **Incoherence Processing**: Apply random Hadamard transform to weights and Hessian
2. **LDLQ Quantization**: Sequential quantization with error compensation in transformed space
3. **Storage** (default `store_transformed=False`):
   - Transform weights back to original space
   - Re-quantize with a single global scale
   - Store as standard `int8_tensorwise` format

The incoherence processing makes weights more uniform during optimization, reducing quantization error while the final output uses standard INT8 format for maximum compatibility and inference performance.

**Storage Mode Comparison:**

| Mode | Process | Output Format | Inference |
|------|---------|---------------|-----------|
| **Original Space** (default) | Transform → Quantize → Inverse Transform → Re-quantize | `int8_tensorwise` | Standard kernels, maximum speed |
| **Transformed Space** (`--quip-store-transformed`) | Transform → Quantize | Custom with metadata | Requires custom loader with inverse transform |

### BF16 Compute Mode

BF16 (BFloat16) compute mode uses half-precision for eligible operations on Ampere+ GPUs:

1. **Memory Savings**: 2× reduction in memory usage for large tensors
2. **Speed**: Faster computation on Tensor Cores
3. **Compatibility**: Automatic fallback to FP32 on older GPUs

**Eligible Operations:**
- Matrix multiplication (matmul)
- Hadamard transforms
- Hessian calculations
- SVD operations
- LDLQ quantization

**Auto Mode Logic:**
```python
if tensor_size >= threshold and bf16_supported:
    use_bf16()
else:
    use_fp32()
```

### Checkpointed Quantization

Checkpointed quantization reduces memory usage by 75-90% on large layers through gradient checkpointing:

1. **Segment Division**: Large layers are divided into segments
2. **Sequential Processing**: Each segment is processed independently
3. **Memory Reuse**: Intermediate results are recomputed rather than stored

**Trade-offs:**
- More memory savings at the cost of slightly slower processing
- Recommended for very large models or limited VRAM

---

## Troubleshooting

### "Dimensions not divisible by block_size"

INT8 block-wise quantization requires tensor dimensions to be divisible by the block size.

**Solutions:**

- Use `--heur` to automatically skip incompatible layers
- Try a different `--block_size` (e.g., 64 instead of 128)
- Use `--scaling_mode axis` or `--scaling_mode tensor` instead

### Out of memory

Large models may exceed GPU memory during SVD computation.

**Solutions:**

- Use `--simple` for simple quantization (no learned rounding)
- Reduce `--max_k` (e.g., 256 instead of 1024)
- Process on CPU (slower but works)
- Use `--streaming-mode=aggressive` for maximum memory safety
- Use `--quip-checkpointed` for extreme memory savings
- Enable BF16: `--bf16-compute=on`
- Close other applications to free up system memory

### Poor quality results

Output model produces artifacts or degraded results.

**Solutions:**

- Keep sensitive layers in high precision (use model-specific flags)
- Use `--heur` to avoid problematic layer shapes
- Increase optimization effort: `--num_iter 2000 --max_k 2048`
- Use `--smoothquant` and `--optimizer gptq` with real calibration data
- Try `--optimizer quip` for near-lossless results

---

## Performance Tuning

This section provides guidance on optimizing quantization speed and memory usage.

### Speed Optimization

#### 1. Use Triton Kernels (GPU only)
Install Triton for significantly faster quantization operations:
```bash
# Linux
pip install -U triton

# Windows (torch>=2.6)
pip install -U "triton-windows<3.3"
```

Speedup: **2-5x** faster than PyTorch fallback for large matrices.

#### 2. Reduce Iterations for Faster Results
For quick quantization without maximum quality:
```bash
# Fast mode (minimal optimization)
convert_to_quant -i model.safetensors --comfy_quant --simple

# Reduced iterations
convert_to_quant -i model.safetensors --comfy_quant --num_iter 100
```

#### 3. Skip Learned Rounding
Use `--simple` to skip SVD-based optimization entirely:
```bash
convert_to_quant -i model.safetensors --comfy_quant --simple
```

Trade-off: **~10x faster** but slightly lower quality.

#### 4. Use QuIP for Speed + Quality
QuIP provides excellent quality with minimal iterations:
```bash
convert_to_quant -i model.safetensors --comfy_quant --optimizer quip
```

QuIP converges faster than learned rounding because it uses Hessian-weighted quantization.

#### 5. Reduce SVD Components
Lower `--max_k` for faster SVD computation:
```bash
convert_to_quant -i model.safetensors --comfy_quant --max_k 512
```

Trade-off: Less optimal rounding but faster processing.

#### 6. Use BF16 Compute Mode
Enable BF16 on Ampere+ GPUs for faster computation:
```bash
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=on
```

Speedup: **1.5-2x** faster for large tensor operations.

### Memory Optimization

#### 1. Streaming Mode
For systems with limited VRAM:
```bash
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=aggressive
```

Effect: Uses significantly less VRAM by offloading large operations to CPU.

#### 2. Low-Memory Mode
For systems with limited system RAM:
```bash
convert_to_quant -i model.safetensors --comfy_quant --low-memory
```

Effect: Uses ~50% less RAM by streaming tensors instead of preloading.

Trade-off: **~20-30% slower** due to on-demand loading.

#### 3. Checkpointed Quantization
For extreme memory savings on large layers:
```bash
convert_to_quant -i model.safetensors --comfy_quant --optimizer quip --quip-checkpointed
```

Effect: 75-90% memory reduction on large layers.

Trade-off: Slightly slower due to recomputation.

#### 4. Process on CPU
If GPU VRAM is limited, processing on CPU works but is slower:
```bash
# Automatic fallback to CPU if CUDA out of memory
# Or force CPU:
CUDA_VISIBLE_DEVICES="" convert_to_quant -i model.safetensors --comfy_quant
```

#### 5. Close Unnecessary Applications
Free up system memory before quantization:
```bash
# Check available memory
free -h  # Linux
# Close browsers, IDEs, and other memory-intensive apps
```

#### 6. Use Smaller Block Sizes
Smaller blocks use slightly less memory during quantization:
```bash
convert_to_quant -i model.safetensors --comfy_quant --block_size 64
```

### Optimization Comparison

| Configuration | Speed | Quality | Memory | Use Case |
|--------------|-------|---------|--------|----------|
| `--simple` | ★★★★★ | ★★☆☆☆ | Low | Quick testing |
| `--num_iter 100` | ★★★★☆ | ★★★☆☆ | Medium | Fast conversion |
| `--optimizer quip` | ★★★★☆ | ★★★★★ | Medium | Best speed/quality |
| Default (1000 iter) | ★★☆☆☆ | ★★★★☆ | High | Maximum quality |
| `--streaming-mode=balanced` | ★★★☆☆ | ★★★★☆ | Low | Balanced (recommended) |
| `--streaming-mode=aggressive` | ★★☆☆☆ | ★★★★☆ | Very Low | Limited VRAM |
| `--quip-checkpointed` | ★★☆☆☆ | ★★★★☆ | Very Low | Extreme memory savings |

### Testing Optimizations

Run the optimization test suite to verify performance:
```bash
python test_optimizations.py
```

This tests:
- Triton kernel performance
- Buffer pool efficiency
- CUDA graph capture/replay
- Memory-mapped loading
- Parallel processing

### Profiling Tips

To identify bottlenecks in your specific use case:

```python
import time
from convert_to_quant.quantization import convert_to_int8

start = time.time()
convert_to_int8(
    input_file="model.safetensors",
    output_file="output.safetensors",
    comfy_quant=True,
    filter_flags={},
    calib_samples=256,
    seed=42,
    optimizer="quip",
)
print(f"Total time: {time.time() - start:.2f}s")
```

---

## License

This tool is provided as-is for research and personal use.
