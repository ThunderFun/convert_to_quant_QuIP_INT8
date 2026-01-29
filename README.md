# convert_to_quant

**Convert safetensors weights to quantized formats (INT8, FP16) with learned rounding optimization for ComfyUI inference.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> [!WARNING]
> **Experimental State**: This project is a fork, currently in a rough state, and has not been extensively tested. It might not be actively maintained. Use with caution.

---

## Installation

> [!IMPORTANT]
> **PyTorch must be installed first** with the correct CUDA version for your GPU.
> This package does not install PyTorch automatically to avoid conflicts with your existing setup.

### Step 1: Install PyTorch (GPU-specific)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct install command for your system.

**Examples:**

```bash
# CUDA 12.8 (stable)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only (no GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install convert_to_quant

```bash
# Install from source
git clone https://github.com/silveroxides/convert_to_quant.git
cd convert_to_quant
pip install -e .
```

### Optional: Triton (needed for INT8 kernels)

```bash
On Linux
pip install -U triton

On Windows
# for torch>=2.6
pip install -U "triton-windows<3.3"
```

---

## Quick Start

**Recommended: QuIP for near-lossless INT8 quantization (highest weight fidelity, best for LoRA compatibility)**

> **Note:** QuIP is optimized for transformer architectures (text encoders, diffusion transformers, etc.). For other model types, the default learned rounding optimizer may be more suitable.

```bash
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant
```

> **Default Storage:** QuIP now stores weights in **transformed space** by default (`store_transformed=True`) for maximum numerical stability. This avoids numerical issues with the inverse Hadamard transform that can cause NaN values in the original-space re-quantization path. The stored weights include QuIP rotation matrices (`quip_s_u`, `quip_s_v`) and are dequantized on-the-fly during inference by compatible loaders.
>
> **For standard loaders:** Use `--quip-no-store-transformed` to store weights in original space with a single global scale (may have numerical stability issues with some models).

**QuIP with SmoothQuant for maximum accuracy (best for models with activation outliers, may reduce LoRA compatibility)**

```bash
convert_to_quant -i model.safetensors --optimizer quip --smoothquant --comfy_quant
```

**Basic INT8 quantization with ComfyUI metadata (default optimizer)**

```bash
convert_to_quant -i model.safetensors --comfy_quant
```

**Low VRAM / Memory-efficient mode**

```bash
convert_to_quant -i model.safetensors --comfy_quant --low-memory
```

**Streaming mode (memory-efficient with GPU acceleration)**

```bash
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=balanced
```

Use `--streaming-mode=balanced` for faster quantization while still saving RAM. Unlike `--low-memory`, this keeps calculations on GPU.

**Aggressive streaming mode (faster with more VRAM)**

```bash
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=aggressive
```

Use `--streaming-mode=aggressive` for 10-20% faster processing on GPUs with 12GB+ VRAM. Increases GPU computation thresholds by 2x while still avoiding OOM.

**With custom learning rate (adaptive schedule by default)**

```bash
convert_to_quant -i model.safetensors --comfy_quant --lr 0.01
```

Load the output `.safetensors` file in ComfyUI like any other model.

---

## Supported Quantization Formats

| Format | Flag | Hardware | Notes |
|--------|------|----------|-------|
| INT8 (block-wise) | *(default)* | Any GPU / CPU | Good balance of quality/speed |
| FP16 | `--fp16` | Any GPU / CPU | High precision fallback |

---

## Model-Specific Presets

| Model | Flag | Notes |
|-------|------|-------|
| Flux.2 | `--flux2` | Keep modulation/guidance/time/final high-precision |
| LoRA | `--lora` | Skip alpha/scale, quantize lora_up/down |
| T5-XXL Text Encoder | `--t5xxl` | Decoder removed, skip norms/biases |
| Mistral Text Encoder | `--mistral` | Norms/biases excluded |
| Visual Encoder | `--visual` | MLP layers excluded |
| Hunyuan Video | `--hunyuan` | Attention norms and vision_in excluded |
| WAN Video | `--wan` | Embeddings, encoders, and head excluded |
| Qwen Image | `--qwen` | Image layers and added norms excluded |
| Z-Image | `--zimage` | cap_embedder/norms excluded |
| Z-Image Refiner | `--zimage_refiner` | Context/noise refiner high-precision |
| Chroma/Distilled (Large) | `--distillation_large` | Keep distilled_guidance, final, img/txt_in high-precision |
| Chroma/Distilled (Small) | `--distillation_small` | Keep only distilled_guidance high-precision |
| NeRF (Large) | `--nerf_large` | Keep nerf_blocks, distilled_guidance, txt_in high-precision |
| NeRF (Small) | `--nerf_small` | Keep nerf_blocks, distilled_guidance high-precision |
| Radiance | `--radiance` | Keep img_in_patch, nerf_final_layer high-precision |

---

## Documentation

- 📖 **[MANUAL.md](MANUAL.md)** - Complete usage guide with examples and troubleshooting
- 🔗 **[quantization.examples.md](quantization.examples.md)** - ComfyUI integration patterns
- 📚 **[docs/API.md](docs/API.md)** - Python API reference for developers

---

## Project Structure

```
convert_to_quant/
├── convert_to_quant/            # Main package
│   ├── cli/                     # CLI entry point & argument parsing
│   ├── comfy/                   # ComfyUI integration components & kernels
│   ├── config/                  # Layer configuration & templates
│   ├── converters/              # Core quantization logic (INT8, GPTQ, SmoothQuant)
│   ├── utils/                   # Shared utilities (tensor, memory, metrics)
│   ├── constants.py             # Model Filter Registry & constants
│   ├── quantization.py          # Simplified INT8 entry point
│   └── convert_to_quant.py      # Backward-compatibility wrapper
├── pyproject.toml               # Package configuration
├── MANUAL.md                    # User documentation
└── ...
```

---

## Key Features

- **Unified Safetensors Loader**: Memory-efficient streaming loader with two modes:
  - **Standard mode**: Preloads all tensors for maximum speed
  - **Low-memory mode**: Streams tensors on-demand, loading only one tensor at a time into RAM
- **QuIP (Quantization with Incoherence Processing)**: Near-lossless INT8 quantization using randomized Hadamard transforms to eliminate outliers and make weights more quantization-friendly. Defaults to standard `int8_tensorwise` format for maximum compatibility and inference performance.
- **Learned Rounding**: SVD-based optimization minimizes quantization error in weight's principal directions
- **GPTQ Optimizer**: Sequential layer-wise optimization with error compensation
- **SmoothQuant**: Preprocessing to migrate quantization difficulty from activations to weights
- **LoRA-Informed Calibration**: Use existing LoRA tensors (`--calibration-lora`) to guide the quantization process for better compatibility
- **Multiple Optimizers**: QuIP, GPTQ, AdamW, RAdam, and the original adaptive LR optimizer
- **Bias Correction**: Automatic bias adjustment using synthetic calibration data
- **Model-Specific Support**: Exclusion lists for sensitive layers (norms, embeddings, distillation)
- **Triton Kernels**: GPU-accelerated quantization/dequantization with fallback to PyTorch
- **Layer Config JSON**: Fine-grained per-layer control with regex pattern matching
- **LR Schedules**: Adaptive, exponential, and plateau learning rate scheduling
- **Quality Metrics**: MSE and SQNR reporting for validation
- **BF16 Compute Mode**: Half-precision computation on Ampere+ GPUs for 2× memory savings
- **Checkpointed Quantization**: Extreme memory savings (75-90%) for large layers
- **Streaming Modes**: Configurable CPU/GPU offloading with auto-detection based on VRAM

---

## Optimizations

The following performance optimizations are implemented:

### 1. Triton Kernels ([`convert_to_quant/comfy/int8_kernels.py`](convert_to_quant/comfy/int8_kernels.py))
GPU-accelerated INT8 quantization/dequantization with autotuning for optimal block sizes.
- `int8_gemm`, `int8_addmm` - Optimized matrix multiplication
- `act_quant`, `weight_quant`, `act_dequant`, `weight_dequant` - Fast quantization ops
- Automatic fallback to PyTorch when Triton is unavailable

### 2. QuIP Triton Matmul ([`convert_to_quant/comfy/int8_kernels.py`](convert_to_quant/comfy/int8_kernels.py))
Specialized matrix multiplication for QuIP-quantized weights with Hadamard transform support.
- `quip_int8_matmul()` - Optimized for QuIP's transformed weight format
- Supports sign vectors (s_u, s_v) and inverse transforms

### 3. Tensor Buffer Pool ([`convert_to_quant/converters/quip_int8.py`](convert_to_quant/converters/quip_int8.py))
Efficient buffer reuse during quantization to reduce memory allocations.
- `TensorBufferPool` class with LRU eviction
- Configurable `max_buffers` limit
- Reduces GC pressure during iterative optimization

### 4. CUDA Graph Support ([`convert_to_quant/converters/base_converter.py`](convert_to_quant/converters/base_converter.py))
Capture and replay CUDA graphs to eliminate CPU launch overhead.
- `CudaGraphRunner` class for repeatable kernel sequences
- Warmup iterations before capture
- Ideal for batched inference scenarios

### 5. Parallel Processing ([`convert_to_quant/quantization.py`](convert_to_quant/quantization.py))
Thread pool for I/O-bound layer-wise operations.
- `ParallelProcessor` with `ThreadPoolExecutor`
- Configurable `max_workers`
- Falls back to sequential for single items

### 6. Lazy Logging ([`convert_to_quant/utils/logging.py`](convert_to_quant/utils/logging.py))
Defer expensive string formatting until messages are actually logged.
- `LazyString` - On-demand string evaluation
- `LazyFormat` - Dynamic value evaluation
- Reduces overhead when log level filters messages

### 7. Memory-Mapped Loading ([`convert_to_quant/utils/memory_efficient_loader.py`](convert_to_quant/utils/memory_efficient_loader.py))
Zero-copy tensor loading via memory mapping.
- `MemoryMappedTensorLoader` - Direct file mapping
- `UnifiedSafetensorsLoader` - Unified interface with optional mmap
- ~50% memory reduction for large models

### 8. BF16 Compute Mode ([`convert_to_quant/constants.py`](convert_to_quant/constants.py))
Half-precision computation on Ampere+ GPUs (RTX 30 series, A100, etc.).
- 2× memory savings for large tensor operations
- Automatic detection of BF16 support
- Per-operation threshold configuration

---

## Advanced Usage

### Layer Config JSON

Define per-layer quantization settings with regex patterns:

```bash
# Generate a template from your model
convert_to_quant -i model.safetensors --dry-run create-template

# Apply custom layer config
convert_to_quant -i model.safetensors --layer-config layers.json --comfy_quant
```

### Scaling Modes

```bash
# Block-wise scaling (default)
convert_to_quant -i model.safetensors --scaling-mode block --block_size 128 --comfy_quant

# Axis-wise (per-row) scaling
convert_to_quant -i model.safetensors --scaling-mode axis --comfy_quant

# Tensor-wise scaling
convert_to_quant -i model.safetensors --scaling-mode tensor --comfy_quant
```

### QuIP Storage Options

By default, QuIP stores weights in **original space** with a single global scale for maximum compatibility:

```bash
# Default: Standard int8_tensorwise format (recommended)
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant
```

**Benefits of original space storage:**
- Full compatibility with Z-Image, ComfyUI, and standard inference pipelines
- Maximum inference performance (no inverse transform overhead)
- Uses optimized hardware INT8 kernels

For advanced use cases, you can store weights in **transformed space** (experimental):

```bash
# Experimental: Store in transformed space (requires custom loader)
convert_to_quant -i model.safetensors --optimizer quip --comfy_quant --quip-store-transformed
```

**When to use transformed storage:**
- When implementing a custom loader that can apply inverse transforms during inference
- For experimental research purposes
- Note: Not compatible with standard inference pipelines

### Memory-Efficient Loading

The `UnifiedSafetensorsLoader` provides a unified interface for loading safetensors files with optional streaming support:

```bash
# Low-memory streaming mode - loads tensors on-demand
convert_to_quant -i model.safetensors --comfy_quant --low-memory

# Standard mode (default) - preloads all tensors for faster processing
convert_to_quant -i model.safetensors --comfy_quant
```

**When to use low-memory mode:**
- Quantizing very large models that don't fit in system RAM
- Running on machines with limited memory
- Processing models alongside other memory-intensive applications

**Mode comparison:**

| Mode | Memory Usage | Speed | Use Case |
|------|--------------|-------|----------|
| Standard | ~2x model size | Fast | Recommended for most users |
| Low-memory | ~1x model size + 1 tensor | Slower | Limited RAM environments |

### Streaming Modes

The quantizer provides several streaming modes for memory-efficient processing:

```bash
# Auto-detect based on GPU VRAM (recommended)
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=auto

# Balanced approach (default)
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=balanced

# Aggressive CPU offloading (maximum memory safety)
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=aggressive

# Minimal offloading (for 12-16GB VRAM)
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=minimal

# Disable streaming (requires 24GB+ VRAM)
convert_to_quant -i model.safetensors --comfy_quant --streaming-mode=off
```

**Streaming Mode Comparison:**

| Mode | Hadamard Threshold | Behavior | Use Case |
|------|-------------------|----------|----------|
| `off` | ∞ (infinity) | Never offload to CPU | Workstations with 24GB+ VRAM |
| `minimal` | 100M elements (~400MB) | Conservative offloading | 12-16GB VRAM |
| `balanced` | 50M elements (~200MB) | Moderate offloading | 8-12GB VRAM |
| `aggressive` | 25M elements (~100MB) | Aggressive offloading | <8GB VRAM or maximum safety |
| `auto` | Adaptive | Detects based on VRAM | Recommended default |

### BF16 Compute Mode

Enable BF16 compute on Ampere+ GPUs (RTX 30 series, A100, etc.) for 2× memory savings:

```bash
# Auto-enable for large tensors (default)
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=auto

# Force BF16 for all supported operations
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=on

# Disable BF16, use FP32 only
convert_to_quant -i model.safetensors --comfy_quant --bf16-compute=off
```

**BF16 with Custom Thresholds:**

```bash
# Adjust tensor size thresholds for BF16 (in elements)
convert_to_quant -i model.safetensors --comfy_quant \
    --bf16-compute=auto \
    --bf16-threshold 1000000 \
    --bf16-hadamard-threshold 500000 \
    --bf16-hessian-threshold 1000000
```

**Recommended for:**
- 8GB GPUs: `--streaming-mode=aggressive --bf16-compute=on`
- 12GB GPUs: `--streaming-mode=balanced --bf16-compute=on`
- 16GB+ GPUs: `--streaming-mode=auto --bf16-compute=on`

### Checkpointed Quantization

For extreme memory savings on large layers (75-90% reduction):

```bash
# Enable checkpointed quantization with default settings
convert_to_quant -i model.safetensors --comfy_quant --optimizer quip --quip-checkpointed

# Custom threshold and segments
convert_to_quant -i model.safetensors --comfy_quant --optimizer quip \
    --quip-checkpointed \
    --quip-checkpoint-threshold 8192 \
    --quip-checkpoint-segments 4
```

**Options:**
- `--quip-checkpointed` - Enable checkpointed LDLQ quantization
- `--quip-checkpoint-threshold` - Dimension threshold (default: 8192)
- `--quip-checkpoint-segments` - Number of segments (default: 4, higher = more memory savings but slower)

### No Memory Limits Mode

Disable all memory safety checks for maximum performance (use with caution):

```bash
# Maximum speed, no safety checks (requires 24GB+ VRAM)
convert_to_quant -i model.safetensors --comfy_quant --no-memory-limits
```

**Warning:** This disables ALL memory protection:
- Pre-emptive memory checking (OOMGuard)
- Adaptive threshold adjustments
- Automatic CPU fallback when VRAM is low
- OOM recovery and learning from OOM events

**Only use when:**
- You have abundant VRAM (24GB+) where OOM is unlikely
- Performance is critical and CPU fallback is unacceptable
- Debugging to isolate OOM handling slowdowns

### Quality Reporting & Calibration

```bash
# INT8 with SmoothQuant, GPTQ, and internal calibration
convert_to_quant -i model.safetensors --smoothquant --optimizer gptq --report-quality --comfy_quant

# With LoRA-informed calibration for best results
convert_to_quant -i model.safetensors --optimizer quip --calibration-lora my_lora.safetensors --comfy_quant
```

---

## LoRA Merging

Merge LoRA weights directly into the base model before quantization for a single unified quantized file:

```bash
# Merge single LoRA
convert_to_quant -i model.safetensors --merge-lora my_lora.safetensors --comfy_quant

# Merge multiple LoRAs with automatic dampening
convert_to_quant -i model.safetensors --merge-loras lora1.safetensors lora2.safetensors --comfy_quant

# Adjust merge scale (default: 1.0)
convert_to_quant -i model.safetensors --merge-lora my_lora.safetensors --merge-lora-scale 0.8 --comfy_quant
```

**Benefits:**
- Single file deployment - No separate LoRA loading at inference time
- Faster inference - No runtime LoRA computation overhead
- Better quantization quality - Optimizers can work with the merged weights

## LoRA Compatibility

For the best results when using LoRAs with quantized models:

- **Use QuIP without SmoothQuant**: Non-SmoothQuant QuIP runs provide the best LoRA compatibility. The QuIP optimizer delivers the highest weight fidelity without the activation-to-weight transformations that SmoothQuant applies, which is crucial for maintaining compatibility with LoRAs trained on the original base model.
- **LoRA-Informed Calibration**: If you have a specific LoRA you want to optimize for, use the `--calibration-lora` flag. This uses the LoRA's weight directions to inform the quantization process for that specific LoRA.

---

## Requirements

- Python 3.9+
- PyTorch 2.1+ (with CUDA for GPU acceleration)
- safetensors >= 0.4.2
- tqdm
- (Optional) triton >= 2.1.0 for INT8 kernels

---


## Acknowledgements

### Original Project (Pre-Fork)
- [Clybius](https://github.com/Clybius) – For inspiring the project and the [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) repository.
- [lyogavin](https://github.com/lyogavin) – For ComfyUI PR [#10864](https://github.com/comfyanonymous/ComfyUI/pull/10864) adding `int8_blockwise` format support and int8 kernels.

### Current Project
- [silveroxides](https://github.com/silveroxides) – For ongoing support and providing the main code for this project.
- [dxqb](https://github.com/dxqb) – For providing the axis-wise implementation (originally from [OneTrainer PR #1034](https://github.com/Nerogar/OneTrainer/pull/1034)).

---

## License

MIT License
