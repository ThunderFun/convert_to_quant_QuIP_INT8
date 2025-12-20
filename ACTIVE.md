# Active Development & Current Implementations

This document tracks current work in progress and actively maintained implementations in the quantization workspace.

---

## Current Focus

### Quantization Method Development
- **SVD-based learned rounding optimization** - Core algorithm for minimizing quantization error
  - File: [convert_to_quant.py](convert_to_quant.py)
  - Status: ✅ Stable, actively improved
  - Optimizers: original, adamw, radam

- **FP8 Quantization** - Floating-point 8-bit quantization
  - File: [quant_ops.py](quant_ops.py) - `TensorCoreFP8Layout` class
  - Status: ✅ Stable
  - Hardware: Ada Lovelace and newer GPUs

- **INT8 Block-Wise Quantization** - Integer 8-bit with per-block scaling
  - File: [quant_ops.py](quant_ops.py) - `BlockWiseINT8Layout` class
  - Status: ✅ Stable, good performance
  - Hardware: Broad support

- **Bias Correction** - Automatic bias adjustment using synthetic calibration data
  - File: [convert_to_quant.py](convert_to_quant.py) - `convert_to_fp8_scaled()` function
  - Status: ✅ Implemented and working

---

## Model Support

### Actively Supported

| Model | Flags | Status | Notes |
|-------|-------|--------|-------|
| Flux / Flux-Dev / Flux-Schnell | `--distillation_large` / `--nerf_large` | ✅ Tested | Distillation layers excluded |
| T5-XXL Text Encoder | `--t5xxl` | ✅ Tested | Decoder removed, always has input_scale |
| Hunyuan Video 1.5 | `--hunyuan` | ✅ Tested | Attention norms excluded |
| WAN Video | `--wan` | ✅ Implemented | Time embeddings excluded |
| Qwen Image | `--qwen` | ✅ Implemented | Image-specific layers excluded |
| Z-Image | `--zimage_l` / `--zimage_s` | ✅ Implemented | Vision encoder specific |
| Radiance Field | `--radiance` | ✅ Implemented | Specialized handling |

---

## Kernel Implementations

### Triton INT8 Kernels

#### int8_kernels.py (Default Backend)
- **Status**: ✅ Production-ready
- **Features**:
  - `act_quant()` / `act_dequant()` - Activation quantization
  - `weight_quant()` / `weight_dequant()` - Weight quantization
  - `int8_gemm()` - Matrix multiplication
  - `int8_addmm()` - Matrix multiplication with bias
  - Fallback to PyTorch for CPU/missing Triton
- **Block sizes**: Fixed (128)

#### int8_matmul.py (Experimental V2 Backend)
- **Status**: 🚧 Experimental, autotuning enabled
- **Features**:
  - `int8_gemm_v2()` - Alternative gemm implementation
  - `int8_addmm()` - Fused matmul + bias
  - `int8_gemm_quant()` / `int8_addmm_quant()` - Fused matmul + quantization
  - `int8_gelu()` - Fused GELU with quantization
- **Block sizes**: Autotuned (16-64+)
- **Use case**: Research, AB testing

---

## ComfyUI Integration

### Quantized Model Format
- **Status**: ✅ Fully implemented
- **Metadata**: `.comfy_quant` tensor (JSON-encoded)
- **Scale storage**: `weight_scale` and `input_scale` tensors per layer
- **Layout compatibility**: `QuantizedTensor` wrapper system in `quant_ops.py`
- **Torch dispatch**: `__torch_dispatch__` mechanism for quantized operations

### Layout-Based Operations
- **Status**: ✅ Implemented
- **Registered operations**:
  - `torch.ops.aten.linear.default` - Quantized linear layers
  - `torch.ops.aten.mm.default` - Quantized matrix multiplication
  - `torch.ops.aten.addmm.default` - Quantized addmm with bias
  - `torch.ops.aten.view.default` / `torch.ops.aten.t.default` - Shape operations
  - `torch.ops.aten.transpose.int` - Transpose with scale handling
  - `torch.ops.aten.to.dtype` - Dtype conversion

### Testing & Validation
- **Status**: 🔄 Ongoing per-model basis
- **Method**: Load in ComfyUI, compare visual outputs
- **Metrics**: Quality, speed, memory usage
- **Reference**: [ComfyUI](ComfyUI/) and [ComfyUI_Custom_Nodes_Agent](ComfyUI_Custom_Nodes_Agent/)

---

## CLI Features

### Core Arguments
- **Format selection**: `--int8` (default FP8)
- **Scaling modes**: `--scaling_mode tensor` / `block`
- **Block size**: `--block_size` (default 64)
- **Optimization**: `--optimizer original|adamw|radam`, `--num_iter`, `--lr`
- **SVD control**: `--top_p`, `--min_k`, `--max_k`, `--full_matrix`

### Model-Specific Flags
- Text encoders: `--t5xxl`
- Diffusion: `--distillation_large`, `--distillation_small`, `--nerf_large`, `--nerf_small`, `--radiance`
- Video: `--wan`, `--hunyuan`
- Image: `--qwen`, `--zimage_l`, `--zimage_s`

### ComfyUI Integration Flags
- **Output format**: `--comfy_quant` (adds metadata)
- **Metadata options**: `--full_precision_matrix_mult`, `--input_scale`
- **Kernel backend**: `--kernel_backend triton|triton_v2`

### Optimization Flags
- **Skip optimization**: `--simple` (round-to-nearest baseline)
- **Skip inefficient**: `--heur` (skip poorly-shaped layers)

---

## Documentation

### Active Maintenance
- ✅ **[MANUAL.md](MANUAL.md)** - User guide (comprehensive)
- ✅ **[AGENTS.md](AGENTS.md)** - AI agent workflows
- ✅ **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Architecture guide
- ✅ **[DEVELOPMENT.md](DEVELOPMENT.md)** - Research notes (being filled)
- ✅ **[quantization.examples.md](quantization.examples.md)** - ComfyUI patterns
- ✅ **[README.md](README.md)** - Project overview

### Reference Submodules
- ✅ **[ComfyUI/](ComfyUI/)** - For architecture reference
- ✅ **[ComfyUI_Custom_Nodes_Agent/](ComfyUI_Custom_Nodes_Agent/)** - For patterns

---

## Performance Baseline

### Memory Reduction
- **FP8**: ~50% reduction vs FP16
- **INT8**: ~50% reduction vs FP16
- **With heuristics**: 40-45% (some layers kept high precision)

### Speed Improvements
- **FP8 on Ada/Hopper**: 1.2-1.5× faster (with tensor cores)
- **INT8 with Triton**: 1.1-1.3× faster
- **CPU fallback**: Works but slower

### Quality
- **Typical MSE**: Minimal (<1% error)
- **Visual quality**: Imperceptible for most models
- **Sensitive models**: May need higher `--num_iter`

---

## Current Issues & Workarounds

### INT8 Dimension Validation
- **Issue**: Some layers have dimensions not divisible by block_size
- **Workaround**: Use `--heur` to auto-skip, or adjust `--block_size`
- **Status**: Automated handling in place

### FP8 Hardware Support
- **Issue**: Pre-Ada GPUs don't have FP8 tensor cores
- **Workaround**: Use `--int8` instead, or run on CPU
- **Status**: Graceful fallback implemented

### Optimization Convergence
- **Issue**: Some tensors optimize slowly
- **Workaround**: Try different optimizer or increase iterations
- **Status**: Multiple optimizers available

---

## Recent Improvements

### Latest Enhancements
- ✨ INT8 block-wise quantization with Triton kernels
- ✨ Multiple optimizer support (original, adamw, radam)
- ✨ Bias correction with synthetic calibration data
- ✨ Layout-based operation dispatch system
- ✨ Comprehensive model-specific exclusion lists
- ✨ Performance heuristics for layer skipping

### Documentation Expansion
- ✨ AGENTS.md for AI agent workflows
- ✨ Comprehensive MANUAL.md for users
- ✨ DEVELOPMENT.md template for research notes
- ✨ quantization.examples.md with ComfyUI patterns

---

## How to Contribute

### Testing New Algorithms
1. Modify `LearnedRoundingConverter._optimize_*()` in [convert_to_quant.py](convert_to_quant.py)
2. Test on model: `python convert_to_quant.py -i model.safetensors --comfy_quant`
3. Validate in ComfyUI
4. Document findings in [DEVELOPMENT.md](DEVELOPMENT.md)

### Adding Model Support
1. Identify sensitive layers in model
2. Add to exclusion lists in [convert_to_quant.py](convert_to_quant.py)
3. Add CLI flag and logic
4. Test in ComfyUI, document in [MANUAL.md](MANUAL.md)

### Improving Documentation
- Update examples in [MANUAL.md](MANUAL.md)
- Add findings to [DEVELOPMENT.md](DEVELOPMENT.md)
- Enhance [quantization.examples.md](quantization.examples.md)
- Improve [AGENTS.md](AGENTS.md) workflows

---

_Last updated: 2025-12-11_
