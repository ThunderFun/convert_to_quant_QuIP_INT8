# Learned Rounding Quantization - Development Workspace

**Research and development workspace for neural network quantization methods targeting ComfyUI inference.**

This workspace develops quantization algorithms that convert PyTorch model weights from full precision to low-precision formats (FP8, INT8) using SVD-based learned rounding optimization. The output is ComfyUI-compatible quantized models.

---

## For Developers

**Working on quantization methods?** Start here:

- 📋 **[AGENTS.md](AGENTS.md)** - Development workflows and quick reference for AI coding agents
- ✨ **[ACTIVE.md](ACTIVE.md)** - Current implementations and active development status
- 📋 **[PLANNED.md](PLANNED.md)** - Roadmap and planned features
- 🏗️ **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Complete architecture, patterns, and technical details
- 🧪 **[DEVELOPMENT.md](DEVELOPMENT.md)** - Experimental findings and research notes
- 🔗 **[quantization.examples.md](quantization.examples.md)** - ComfyUI integration patterns
- 📚 **[ComfyUI_Custom_Nodes_Agent/](ComfyUI_Custom_Nodes_Agent/)** - Reference for ComfyUI development patterns
- 🔍 **[ComfyUI/](ComfyUI/)** - ComfyUI source code reference

### Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/silveroxides/convert_to_quant.git

# Install dependencies
pip install -r requirements.txt

# Test quantization
python convert_to_quant.py -i test_model.safetensors --comfy_quant
```

### Development Focus

- **Quantization algorithms**: SVD-based learned rounding, custom optimizers
- **Format support**: FP8 (tensor cores) and INT8 (block-wise)
- **Model architectures**: Flux, T5-XXL, Hunyuan Video, WAN, and more
- **ComfyUI integration**: Compatible quantized model generation

---

## For Users

**Using quantized models?** See the user documentation:

- 📖 **[MANUAL.md](MANUAL.md)** - Complete usage guide with examples and troubleshooting

### Quick Usage

```bash
# Basic FP8 quantization
python convert_to_quant.py -i model.safetensors --comfy_quant

# INT8 with performance heuristics
python convert_to_quant.py -i model.safetensors --int8 --comfy_quant --heur

# T5-XXL text encoder
python convert_to_quant.py -i t5xxl.safetensors --t5xxl --comfy_quant

# High quality (slow)
python convert_to_quant.py -i model.safetensors --comfy_quant --num_iter 2000
```

Load the output `.safetensors` file in ComfyUI like any other model.

---

## Project Structure

```
convert_to_quant.py              # Quantization implementation and experiments
quant_ops.py                     # ComfyUI-compatible layout system
kernels/                         # Triton GPU kernels
  int8_kernels.py                # Default INT8 backend
  int8_matmul.py                 # Experimental autotuned kernels
float_utils/                     # Utility functions
AGENTS.md                        # AI agent development guide
ACTIVE.md                        # Current implementations & status
PLANNED.md                       # Roadmap and planned features
DEVELOPMENT.md                   # Research notes and findings
MANUAL.md                        # User documentation
quantization.examples.md         # ComfyUI integration examples
ComfyUI_Custom_Nodes_Agent/      # Reference submodule (patterns)
ComfyUI/                         # Reference submodule (source)
```

---

## Key Features

- **Learned Rounding**: SVD-based optimization minimizes quantization error in weight's principal directions
- **Multiple Optimizers**: Original (adaptive LR), AdamW, RAdam, ProdigyPlusScheduleFree
- **Bias Correction**: Automatic bias adjustment using synthetic calibration data
- **Model-Specific Support**: Exclusion lists for sensitive layers (norms, embeddings, distillation)
- **Triton Kernels**: GPU-accelerated quantization/dequantization with fallback to PyTorch

---

## Citation & References

- DeepSeek FP8 matmul: https://github.com/deepseek-ai/DeepSeek-V3
- JetFire paper: https://arxiv.org/abs/2403.12422

---

## License

[Add license information]