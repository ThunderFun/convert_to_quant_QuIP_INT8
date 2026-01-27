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

---

## Documentation

- 📖 **[MANUAL.md](MANUAL.md)** - Complete usage guide with examples and troubleshooting
- 🔗 **[quantization.examples.md](quantization.examples.md)** - ComfyUI integration patterns
- 🧪 **[tests/](tests/)** - Functional tests and diagnostic scripts

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

- **QuIP (Quantization with Incoherence Processing)**: Near-lossless INT8 quantization using randomized Hadamard transforms to eliminate outliers and make weights more quantization-friendly. Optimized for transformer architectures.
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

### Quality Reporting & Calibration

```bash
# INT8 with SmoothQuant, GPTQ, and internal calibration
convert_to_quant -i model.safetensors --smoothquant --optimizer gptq --report-quality --comfy_quant

# With LoRA-informed calibration for best results
convert_to_quant -i model.safetensors --optimizer quip --calibration-lora my_lora.safetensors --comfy_quant
```

---

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
