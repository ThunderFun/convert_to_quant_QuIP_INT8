# Planned Implementations & Roadmap

This document outlines future improvements, experimental features, and long-term development direction.

---

## 🧹 Priority: Cleanup 4-bit Leftover Code

### Overview

The 4-bit quantization feature (NF4, FP4) should be **removed** from the codebase. This cleanup task addresses any remaining artifacts from the previous implementation.

### Tasks

- [ ] Remove any orphaned 4-bit related imports
- [ ] Clean up bitsandbytes submodule reference if no longer needed
- [ ] Remove any 4-bit related CLI arguments that may still exist
- [ ] Clean up any NF4/FP4 references in documentation
- [ ] Remove unused kernel files related to 4-bit operations
- [ ] Ensure all style and structure is same as before cleanup of the codebase
---

## High Priority

### Quantization Algorithm Improvements

#### [ ] Randomized SVD for Faster Computation
- **Purpose**: Speed up SVD computation for very large models
- **Implementation**: Use `torch.svd_lowrank()` with randomized approach
- **Expected benefit**: 2-3× faster SVD, minimal quality loss
- **Effort**: Medium
- **Reference file**: [convert_to_quant.py](convert_to_quant.py) - `_convert_fp8()` method

#### [ ] Per-Channel Quantization for Weights
- **Purpose**: Fine-grained quantization per output channel
- **Implementation**: Compute scales per channel instead of per-tensor
- **Expected benefit**: Better quality for certain architectures
- **Tradeoff**: More memory for scales, slightly slower inference
- **Effort**: Medium
- **Reference**: `quant_ops.py` - Create `PerChannelLayout` class

#### [ ] Mixed Precision Strategy
- **Purpose**: Automatically decide FP8 vs INT8 per layer
- **Implementation**: Quantization quality assessment per layer
- **Expected benefit**: Better quality without manual per-model tuning
- **Effort**: High
- **Status**: Requires extensive testing

---

### Model Architecture Support

#### [ ] Additional Diffusion Models
- **Flux variants**: All versions standardize on `--distillation_large`
- **SDXL**: Add specific handling if needed
- **Stability AI models**: Research-specific layers
- **Status**: Awaiting model releases/examples

#### [ ] Video Model Improvements
- **Hunyuan Video**: Enhanced testing and benchmarks
- **WAN**: More comprehensive layer exclusions
- **New models**: Plan for upcoming video diffusion models
- **Effort**: Medium per model

---

### Kernel Optimization

#### [ ] Triton V2 Kernel Stabilization
- **Current status**: Experimental, autotuning enabled
- **Goals**: 
  - Make production-ready
  - Add comprehensive testing
  - Benchmark vs default backend
- **File**: [kernels/int8_kernels.py](kernels/int8_kernels.py)
- **Effort**: Medium

---

## Medium Priority (Planning Phase)

### Format Extensions

#### [ ] Flexible Dtype Support
- **Current**: FP8 (e4m3fn), INT8, FP32
- **Planned**: FP16, BF16, E5M2 variant of FP8
- **Effort**: Low-Medium

---

### Performance Features

#### [ ] Activation Quantization at Inference Time
- **Purpose**: Dynamic quantization during inference
- **Implementation**: Store activation scales with model
- **Benefit**: Further memory reduction
- **Challenge**: Inference-time overhead
- **Effort**: High

---

## Technical Debt & Refactoring

### Code Improvements
- [ ] Consolidate duplicate code in optimizer implementations
- [ ] Improve error messages and debugging output
- [ ] Add comprehensive type hints throughout
- [ ] Refactor `convert_to_fp8_scaled()` - too many parameters

### Testing Infrastructure
- [ ] Unit tests for individual components
- [ ] Integration tests with ComfyUI
- [ ] Regression tests for model support
- [ ] Performance regression testing

---

_Last updated: 2025-12-11_

For discussions on specific items, see [DEVELOPMENT.md](DEVELOPMENT.md) for research notes and [AGENTS.md](AGENTS.md) for active development workflows.
