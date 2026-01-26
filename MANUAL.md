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

- **Learned rounding optimization**: Uses SVD-based optimization to minimize quantization error
- **GPTQ Optimizer**: Sequential layer-wise optimization with error compensation
- **QuIP (Quantization with Incoherence Processing)**: Near-lossless INT8 quantization using random orthogonal transformations
- **SmoothQuant**: Preprocessing to migrate quantization difficulty from activations to weights
- **Multiple optimizer choices**: Original, AdamW, RAdam, GPTQ, QuIP
- **Bias correction**: Automatically adjusts biases to compensate for quantization error
- **Model-specific filters**: Keep sensitive layers in high precision for various architectures
- **ComfyUI compatible**: Generates `.comfy_quant` metadata for seamless integration

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
| `--distillation_large` | Keep: `distilled_guidance_layer`, `final_layer`, `img_in`, `txt_in` |
| `--distillation_small` | Keep: `distilled_guidance_layer` only |
| `--nerf_large` | Keep: `distilled_guidance_layer`, `nerf_blocks`, `nerf_image_embedder`, `txt_in` |
| `--nerf_small` | Keep: `distilled_guidance_layer`, `nerf_blocks`, `nerf_image_embedder` |
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
# Basic QuIP conversion
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant

# QuIP with SmoothQuant for maximum accuracy
convert_to_quant -i model.safetensors --optimizer quip --smoothquant --comfy_quant
```

---

## How It Works

### Quantization Process

1. **Load weights**: Read all tensors from the input safetensors file
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
- Use `--low-memory` flag

### Poor quality results

Output model produces artifacts or degraded results.

**Solutions:**

- Keep sensitive layers in high precision (use model-specific flags)
- Use `--heur` to avoid problematic layer shapes
- Increase optimization effort: `--num_iter 2000 --max_k 2048`
- Use `--smoothquant` and `--optimizer gptq` with real calibration data
- Try `--optimizer quip` for near-lossless results

---

## License

This tool is provided as-is for research and personal use.
