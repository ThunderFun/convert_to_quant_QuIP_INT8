# Learned Rounding Quantization - Development Workspace

## Project Overview

**Purpose**: Research and development workspace for neural network quantization methods targeting the **ComfyUI inference platform**.

This workspace develops and tests quantization algorithms that convert PyTorch model weights from full precision (FP16/FP32/BF16) to low-precision formats (INT8 or FP16) using **SVD-based learned rounding optimization**. The quantized models are designed to be compatible with ComfyUI's `quant_ops.py` system and support multiple AI model architectures (Flux, T5-XXL, Hunyuan Video, etc.).

**Development Context**:
- **Primary Goal**: Experimenting with quantization algorithms and optimization strategies
- **Target Platform**: ComfyUI inference runtime with `QuantizedTensor` support
- **Output Format**: Quantized safetensors with ComfyUI-compatible metadata (`.comfy_quant`)
- **Approach**: Modular package structure for maintainability

### Core Architecture

1. **Main CLI** ([convert_to_quant/cli/main.py](convert_to_quant/cli/main.py)): Entry point for the conversion tool.
2. **Quantization Engine** ([convert_to_quant/quantization.py](convert_to_quant/quantization.py)): Core logic for INT8 conversion and optimization.
3. **Converters** ([convert_to_quant/converters/](convert_to_quant/converters/)): Implementation of various quantization algorithms (GPTQ, QuIP, SmoothQuant, Learned Rounding).
4. **ComfyUI Integration** ([convert_to_quant/comfy/](convert_to_quant/comfy/)): ComfyUI-compatible pluggable quantization format handlers via `QuantizedTensor` wrapper.
5. **Utilities** ([convert_to_quant/utils/](convert_to_quant/utils/)): Shared utilities for tensor operations, memory management, and quality metrics.

## Key Concepts

### Learned Rounding (Core Innovation)

Standard quantization uses round-to-nearest, which is locally optimal but globally suboptimal. This tool uses **SVD-based optimization** to minimize quantization error in the most important directions:

1. Compute truncated SVD: `W ≈ U_k @ S_k @ V_k^T`
2. Optimize quantized values to minimize: `||U_k^T @ (W_dequant - W_orig) @ V_k||`
3. This preserves weight matrix structure while finding optimal per-element rounding decisions

The gradient derivation for INT8 is **critical**: `∂L/∂Q = ∂L/∂dq * scale` (multiply by scale, not divide) because dequantization multiplies `Q * scale`.

### Quantization Formats

- **INT8** (block-wise): 8-bit integer, symmetric range [-127, 127]
  - **2D block-wise scaling** for weights: `(M//block_size, N//block_size)`
  - 1D block-wise scaling for activations: `(*batch_dims, K//block_size)`
  - Requires dimensions divisible by `block_size` (default 128)

- **FP16**: 16-bit floating point fallback for sensitive layers.

### Optimizer Choices

Controlled via `--optimizer` flag:
- **`original`**: Custom adaptive learning rate, fastest convergence, no autograd overhead
- **`adamw`/`radam`**: PyTorch optimizers, good for general cases
- **`gptq`**: Sequential layer-wise optimization with error compensation
- **`quip`**: Near-lossless INT8 quantization using random orthogonal transformations

Use `--simple` to skip learned rounding entirely (simple round-to-nearest).

## Code Patterns

### 1. Layout-Based Quantization (quant_ops.py)

The codebase uses a **pluggable layout system** to support multiple quantization formats:

```python
# Register a new layout-specific operation
@register_layout_op(torch.ops.aten.linear.default, "BlockWiseINT8Layout")
def int8_linear(func, args, kwargs):
    # Extract quantized data
    # Perform quantized computation
    pass
```

### 2. Model-Specific Layer Exclusions

The tool preserves sensitive layers (norms, biases, embeddings) in high precision using exclusion lists defined in `constants.py`:

```python
# Example: T5-XXL exclusions
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", ...]
```

### 3. Bias Correction via Calibration Data

After quantization, biases are adjusted to compensate for quantization error using synthetic or real calibration data.

### 4. Triton Kernel Integration

INT8 quantization uses Triton kernels for performance when available on GPU.

## ComfyUI Integration

### Quantized Model Requirements

- **Metadata**: `.comfy_quant` tensor with JSON-encoded format information
- **Scale Storage**: `weight_scale` and optional `input_scale` tensors per layer
- **Layout Compatibility**: Matches ComfyUI's `QuantizedLayout` interface

## Development Guidelines

### Experimenting with Quantization Algorithms

1. Modify `BaseConverter` subclasses in `convert_to_quant/converters/` to test new algorithms.
2. Adjust SVD rank selection logic (`top_p`, `min_k`, `max_k`) in `LearnedRoundingConverter`.

### Adding Support for New Models

1. Identify sensitive layers in `convert_to_quant/constants.py`.
2. Add CLI flag and exclusion logic in `convert_to_quant/cli/argument_parser.py` and `main.py`.

## File Structure

```
convert_to_quant/
├── cli/                     # CLI entry point & argument parsing
├── comfy/                   # ComfyUI integration components & kernels
├── config/                  # Layer configuration & templates
├── converters/              # Core quantization logic (INT8, GPTQ, SmoothQuant)
├── utils/                   # Shared utilities (tensor, memory, metrics)
├── constants.py             # Model Filter Registry & constants
├── quantization.py          # Simplified INT8 entry point
└── convert_to_quant.py      # Backward-compatibility wrapper
tests/                       # Test suite and diagnostic scripts
README.md                    # Project overview
MANUAL.md                    # User documentation
quantization.examples.md     # ComfyUI integration patterns
```
