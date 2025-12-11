# Planned Implementations & Roadmap

This document outlines future improvements, experimental features, and long-term development direction.

---

## 🔥 Priority: Native bitsandbytes Implementation

### Overview

The goal is to implement a **low-level native PyTorch implementation** of bitsandbytes quantization algorithms focused on:
1. **Creating pre-quantized models** for efficient storage and distribution
2. **Loading and using quantized models in ComfyUI** without requiring diffusers/transformers dependencies
3. **Native CUDA support** with optimized kernels for GPU acceleration
4. **Minimal C++ footprint** - only for custom CUDA kernels where necessary

**Key Principle**: Avoid the heavily abstracted diffusers/transformers ecosystem. This implementation should be:
- Directly usable with raw PyTorch tensors and safetensors files
- Compatible with ComfyUI's model loading infrastructure
- Self-contained without pulling in HuggingFace library dependencies

The reference implementation is now available as a git submodule: `./bitsandbytes/`

### Key Algorithms from bitsandbytes

Based on the reference code analysis, the following quantization methods need implementation:

#### 1. NF4 (Normal Float 4-bit)
- **Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Principle**: Information-theoretic 4-bit quantization where each bin has equal area under N(0,1)
- **Codebook**: 16 values normalized to [-1, 1]
  ```
  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
   0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
  ```
- **Block size**: 64 (default) or 128 (ROCm)
- **Double quantization**: absmax values can be themselves quantized (nested)

#### 2. FP4 (Floating Point 4-bit)
- **Principle**: 4-bit floating point with 1 sign, 2 exponent, 1 mantissa bit
- **Codebook**: Hardware-inspired FP4 values
  ```
  [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
  ```

#### 3. INT8 Block-wise (LLM.int8())
- **Paper**: [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- **Principle**: Vector-wise quantization with outlier handling
- **Already partially implemented** in current codebase (`BlockWiseINT8Layout`)

---

## Phase 1: Core Quantization Functions

### [ ] 1.1 NF4 Quantization Engine

**Target file**: `kernels/nf4_kernels.py` (new)

```python
# Key functions to implement:
def create_nf4_map() -> torch.Tensor:
    """Create the NF4 quantization codebook (16 values)"""
    
def quantize_nf4_blockwise(
    tensor: torch.Tensor,
    block_size: int = 64,
    compress_statistics: bool = False
) -> Tuple[torch.Tensor, QuantState]:
    """Quantize tensor to packed NF4 format"""
    
def dequantize_nf4_blockwise(
    packed_data: torch.Tensor,
    quant_state: QuantState
) -> torch.Tensor:
    """Dequantize packed NF4 back to fp16/bf16"""
```

**Implementation approach**:
1. Pure PyTorch implementation first for CPU/validation
2. Triton kernels for GPU acceleration (portable, easy to develop)
3. CUDA C++ kernels for maximum performance where needed
4. Support for nested quantization (double quant) of absmax values

**Reference files in bitsandbytes**:
- `bitsandbytes/functional.py`: `quantize_4bit()`, `dequantize_4bit()`, `get_4bit_type()`
- `bitsandbytes/triton/quantize_rowwise.py`: Triton kernel patterns
- `bitsandbytes/csrc/`: CUDA kernel implementations

### [ ] 1.2 FP4 Quantization Engine

**Target file**: `kernels/fp4_kernels.py` (new)

Similar structure to NF4, sharing most infrastructure but with different codebook.

### [ ] 1.3 QuantState Class

**Target file**: `quant_ops.py` (extend existing)

```python
class QuantState4bit:
    """Container for 4-bit quantization state (compatible with bitsandbytes format)"""
    absmax: torch.Tensor       # Per-block absolute maximum values
    shape: torch.Size          # Original tensor shape
    code: torch.Tensor         # Quantization codebook (16 values for 4-bit)
    blocksize: int             # Block size used for quantization
    quant_type: str            # "nf4" or "fp4"
    dtype: torch.dtype         # Original dtype
    # For nested/double quantization:
    offset: Optional[float]    # Mean of absmax before nested quant
    state2: Optional[QuantState4bit]  # Nested quant state for absmax
```

### [ ] 1.4 4-bit Packing/Unpacking

**Challenge**: Pack two 4-bit values into one uint8

```python
def pack_4bit(tensor: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit values: [a, b, c, d] -> [ab, cd] (2 values per byte)"""
    
def unpack_4bit(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack 4-bit values back to individual values"""
```

---

## Phase 2: Layout Integration for ComfyUI

### [ ] 2.1 NF4 Layout Class

**Target file**: `quant_ops.py`

```python
class NF4Layout(QuantizedLayout):
    """NF4 4-bit quantization layout for ComfyUI integration"""
    
    @classmethod
    def quantize(cls, tensor, block_size=64, compress_statistics=False, **kwargs):
        """Quantize float tensor to NF4"""
        
    @staticmethod  
    def dequantize(qdata, quant_state, **kwargs):
        """Dequantize NF4 back to float"""
        
    @classmethod
    def get_plain_tensors(cls, qtensor):
        """Extract raw tensors for computation"""
```

### [ ] 2.2 FP4 Layout Class

Similar to NF4Layout with FP4-specific codebook.

### [ ] 2.3 Linear Operation Handlers

```python
@register_layout_op(torch.ops.aten.linear.default, "NF4Layout")
def nf4_linear(func, args, kwargs):
    """Handle linear operations with NF4 weights"""
    # Dequantize-at-inference approach (no native 4-bit matmul in PyTorch)
```

---

## Phase 3: Model Conversion Tool

### [ ] 3.1 4-bit Conversion in convert_to_quant.py

**Add to convert_to_quant.py**:

```bash
python convert_to_quant.py model.safetensors --output model_nf4.safetensors \
    --mode nf4 \
    --block_size 64 \
    --double_quant \
    --exclude "*.norm*,*.embed*"
```

**New arguments**:
- `--mode nf4|fp4|int4` - 4-bit quantization type
- `--block_size 64|128|256` - Block size for quantization  
- `--double_quant` - Enable double quantization of absmax
- `--quant_storage uint8|int8` - Storage type for packed values

### [ ] 3.2 Metadata Format

Extend `.comfy_quant` metadata:

```python
{
    "type": "nf4",  # or "fp4", "int4"
    "block_size": 64,
    "double_quant": true,
    "quant_map": [...],  # 16 values
    "absmax_shape": [...],
    "original_dtype": "bfloat16",
    # For double quant:
    "nested_offset": 0.5,
    "nested_absmax": "key_to_nested_absmax_tensor",
}
```

---

## Phase 4: Safetensors Serialization

### [ ] 4.1 QuantState Serialization

**Challenge**: Safetensors only stores tensors, need to encode non-tensor data.

**Solution** (from bitsandbytes):
```python
def pack_quant_state_to_tensor(quant_state: QuantState4bit) -> Dict[str, torch.Tensor]:
    """Pack non-tensor state into tensors for safetensors compatibility"""
    # Use a sentinel key like "quant_state.bitsandbytes__nf4" 
    # containing packed non-tensor data
```

### [ ] 4.2 Loading Pre-quantized Models

```python
def load_4bit_model(path: str) -> Dict[str, QuantizedTensor]:
    """Load a pre-quantized 4-bit model from safetensors"""
    # 1. Load raw tensors
    # 2. Reconstruct QuantState from packed data
    # 3. Create QuantizedTensor wrappers
```

---

## Phase 5: ComfyUI Integration

### [ ] 5.1 4-bit Model Loader Node

**Target**: ComfyUI custom node or core integration

```python
class Load4BitModel:
    """Load NF4/FP4 quantized models in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING",),
                "quant_type": (["nf4", "fp4"],),
            }
        }
```

### [ ] 5.2 Runtime Dequantization

For inference, 4-bit weights are dequantized on-the-fly during forward pass:
- Dequantize weight blocks as needed
- Keep weights in 4-bit format in VRAM
- ~4x memory reduction vs fp16

### [ ] 5.3 Update quant_ops.py in ComfyUI

Sync the following to ComfyUI's `comfy/quant_ops.py`:
- `NF4Layout` and `FP4Layout` classes
- Registration in `LAYOUTS` and `QUANT_ALGOS` dicts
- Linear operation handlers

---

## Phase 6: Testing & Validation

### [ ] 6.1 Numerical Accuracy Tests

```python
def test_nf4_roundtrip():
    """Verify quantize->dequantize produces acceptable error"""
    original = torch.randn(4096, 4096, dtype=torch.float16)
    quant, state = quantize_nf4_blockwise(original)
    reconstructed = dequantize_nf4_blockwise(quant, state)
    mse = ((original - reconstructed) ** 2).mean()
    assert mse < threshold
```

### [ ] 6.2 bitsandbytes Compatibility Tests

```python
def test_compat_with_bnb():
    """Verify our implementation matches bitsandbytes output"""
    import bitsandbytes as bnb
    
    # Create identical input
    tensor = torch.randn(4096, 4096).cuda()
    
    # Quantize with both implementations
    bnb_quant, bnb_state = bnb.functional.quantize_nf4(tensor)
    our_quant, our_state = quantize_nf4_blockwise(tensor)
    
    # Compare outputs
    assert torch.allclose(bnb_quant, our_quant)
```

### [ ] 6.3 ComfyUI Integration Tests

- Load quantized Flux model
- Generate images with quantized model
- Compare quality vs fp16 baseline

---

## Implementation Timeline

### Sprint 1 (Current Focus)
- [x] Add bitsandbytes as submodule for reference
- [ ] Implement NF4 quantization in pure PyTorch
- [ ] Implement NF4 dequantization in pure PyTorch
- [ ] Add QuantState4bit class

### Sprint 2
- [ ] Triton kernels for NF4 quant/dequant
- [ ] FP4 implementation
- [ ] 4-bit packing utilities

### Sprint 3
- [ ] NF4Layout and FP4Layout classes
- [ ] Linear operation handlers
- [ ] Integration with QuantizedTensor

### Sprint 4
- [ ] convert_to_quant.py 4-bit mode
- [ ] Safetensors serialization
- [ ] Model loading utilities

### Sprint 5
- [ ] ComfyUI integration
- [ ] Testing and validation
- [ ] Documentation

---

## High Priority (Previous Items)

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
- **Purpose**: Automatically decide FP8 vs INT8 vs NF4 vs FP4 per layer
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

#### [ ] 4-bit Matmul Optimization
- **Purpose**: Fused dequant+matmul for 4-bit inference
- **Status**: Research phase
- **Challenge**: No native 4-bit tensor cores, must dequantize first
- **Approach**: Tile-wise dequantization in L2 cache

---

## Medium Priority (Planning Phase)

### Format Extensions

#### [ ] INT4 Linear Quantization
- **Purpose**: Simple linear 4-bit without codebook
- **Challenges**: Very limited precision range
- **Effort**: Medium

#### [ ] Flexible Dtype Support
- **Current**: FP8 (e4m3fn), INT8, FP32
- **Planned**: NF4, FP4, FP16, BF16, E5M2 variant of FP8
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

## bitsandbytes Reference Architecture

### Key Files in Submodule

| File | Purpose |
|------|---------|
| `bitsandbytes/functional.py` | Core quantization functions (quantize_4bit, dequantize_4bit, QuantState) |
| `bitsandbytes/nn/modules.py` | Linear4bit, Params4bit, QuantState handling |
| `bitsandbytes/triton/quantize_rowwise.py` | Triton kernel for row-wise INT8 quantization |
| `bitsandbytes/triton/dequantize_rowwise.py` | Triton kernel for row-wise INT8 dequantization |
| `bitsandbytes/triton/quantize_global.py` | Triton kernel for global INT8 quantization |

### Algorithm Flow

```
Original Tensor (fp16/bf16/fp32)
         │
         ▼
    ┌─────────────┐
    │ Block-wise  │  Divide into blocks of size 64/128
    │  Reshape    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Compute     │  absmax = max(abs(block))
    │  absmax     │  per block
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Normalize   │  normalized = block / absmax
    │  to [-1,1]  │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Find        │  Find closest value in NF4/FP4 codebook
    │  Nearest    │  for each element
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Pack 4-bit  │  Pack 2 values per byte
    │             │
    └─────────────┘
         │
         ▼
Quantized Data (uint8) + QuantState (absmax, shape, dtype, codebook)
```

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

## Success Criteria

### For 4-bit Implementation
- **Numerical accuracy**: MSE < 0.01 vs bitsandbytes reference
- **Memory reduction**: ~4x vs fp16 for weights
- **Performance**: Inference speed within 2x of fp16 (due to dequant overhead)
- **Compatibility**: Load bitsandbytes-quantized models
- **ComfyUI**: Full integration with existing quantization system

---

_Last updated: 2025-12-11_

For discussions on specific items, see [DEVELOPMENT.md](DEVELOPMENT.md) for research notes and [AGENTS.md](AGENTS.md) for active development workflows.
