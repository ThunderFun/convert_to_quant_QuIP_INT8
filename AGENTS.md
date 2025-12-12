# AI Agent Instructions for Quantization Method Development

## Project Purpose

**Development workspace for researching and implementing neural network quantization methods targeting ComfyUI inference.**

This workspace focuses on:
- Developing quantization algorithms (SVD-based learned rounding, custom optimizers)
- Testing different optimization strategies (original, AdamW, RAdam, ProdigyPlusScheduleFree)
- Creating ComfyUI-compatible quantized models (FP8, INT8 formats)
- Prototyping new quantization techniques in simple script format

**Not a production tool** - This is a research environment for rapid experimentation.

---

## Quick Links

- **[Comprehensive Architecture](.github/copilot-instructions.md)** - Full technical details, code patterns, gotchas
- **[Active Development](ACTIVE.md)** - Current implementations and development status
- **[Roadmap & Plans](PLANNED.md)** - Planned features and long-term direction
- **[Python Standards](python.instructions.md)** - Coding conventions (PEP 8, type hints, 79-char limit)
- **[ComfyUI Integration Examples](quantization.examples.md)** - Target platform patterns and QuantizedTensor usage
- **[ComfyUI Development Reference](ComfyUI_Custom_Nodes_Agent/)** - Reference submodule for ComfyUI patterns
- **[ComfyUI Source Reference](ComfyUI/)** - ComfyUI source code for easy searching
- **[User Documentation](MANUAL.md)** - Guide for using quantized models (end-user focused)
- **[Development Notes](DEVELOPMENT.md)** - Experimental findings and research notes
- **[bitsandbytes Reference](bitsandbytes/)** - Reference submodule for NF4/FP4 algorithms

---

## 📝 Documentation Guidelines

**After completing a task**, save the walkthrough as an entry in `DEVELOPMENT.md`:
- Add new entries at the top (reverse chronological order)
- Include: Session Summary, Files Created/Modified, Usage examples
- Follow the existing format in [DEVELOPMENT.md](DEVELOPMENT.md)

---

## 🔧 bitsandbytes Integration Scope

### Directory Structure
- **`convert_to_quant.py`** - Central script for all quantization methods (FP8, INT8, NF4, FP4)
- **`kernels/`** - Quantization kernel implementations (Triton/CUDA)
- **`quant_ops.py`** - Layout classes and QuantizedTensor (mirrors ComfyUI)
- **`bitsandbytes/`** - Reference submodule (read-only, for algorithm study)
- **`ComfyUI/`** - Development branch for ComfyUI integration

### ComfyUI Integration Rules

**Minimal Code Changes** - Keep modifications focused:
1. **Add support files** - New kernel files in `comfy/` (e.g., `nf4_kernels.py`)
2. **Extend `quant_ops.py`** - Add Layout classes, register in `LAYOUTS`/`QUANT_ALGOS`
3. **Avoid core changes** - Do NOT modify `model_management.py`, memory management, or model loaders

**What to add to ComfyUI `quant_ops.py`**:
- New `QuantizedLayout` subclasses (e.g., `NF4Layout`, `FP4Layout`)
- Register in `QUANT_ALGOS` and `LAYOUTS` dicts
- Operation handlers via `@register_layout_op()` decorator
- Helper functions for the new formats

**What NOT to change**:
- Core memory management (`model_management.py`)
- Model loading infrastructure (`sd.py`, `model_base.py`)
- QuantizedTensor base class (unless absolutely necessary)

### Key Constraints

1. **No diffusers/transformers** - Implementation must be self-contained
2. **Native CUDA support** - Use Triton kernels, with optional CUDA C++ for performance
3. **Minimal C++** - Only for custom CUDA kernels where Triton is insufficient
4. **ComfyUI compatible** - Models must load via standard ComfyUI loaders
5. **Safetensors serialization** - All quantization state must be serializable

### Reference Files

| bitsandbytes | Purpose |
|--------------|---------|
| `functional.py` | Core quantize/dequantize functions, QuantState, codebooks |
| `nn/modules.py` | Linear4bit, Params4bit, weight handling |
| `triton/*.py` | Triton kernel patterns |
| `csrc/` | CUDA kernel implementations |

---

## Development Workflows

### 1. Experimenting with New Quantization Algorithms

**Goal**: Test modifications to the optimization process

```python
# Key file: convert_to_quant.py -> LearnedRoundingConverter class

# Common experiments:
# - Modify loss function in _optimize_original()
# - Adjust SVD rank selection (top_p, min_k, max_k)
# - Test new optimizer implementations (_optimize_<name>)
# - Change scale computation strategies
```

**Workflow**:
1. Modify optimizer method in `LearnedRoundingConverter`
2. Run on test model: `python convert_to_quant.py -i test_model.safetensors --comfy_quant`
3. Load in ComfyUI and validate quality
4. Document results in `DEVELOPMENT.md`

### 2. Adding New Model Architecture Support

**Goal**: Enable quantization for new model types (e.g., new diffusion models)

**Pattern from code**:
```python
# 1. Identify sensitive layers by inspecting model structure
# 2. Add exclusion list
MODEL_AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", ...]
SPECIFIC_MODEL_LAYERS = ["special_layer_1", "special_layer_2"]

# 3. Add CLI flag
parser.add_argument("--my_model", action='store_true', help="Exclude MY_MODEL layers")

# 4. Add logic in convert_to_fp8_scaled()
if my_model and any(n in key for n in SPECIFIC_MODEL_LAYERS):
    skip_reason = "MY_MODEL layer keep in high precision"
```

**Testing**: Load in ComfyUI, generate outputs, compare quality to original

### 3. Implementing New Quantization Layouts

**Goal**: Add support for new quantization formats (e.g., INT4, custom FP formats)

**Required steps** (see [quant_ops.py](quant_ops.py)):
```python
# 1. Create QuantizedLayout subclass
class MyNewLayout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, **kwargs):
        # Implement quantization
        return qdata, layout_params
    
    @staticmethod
    def dequantize(qdata, **layout_params):
        # Implement dequantization
        return tensor
    
    @classmethod
    def get_plain_tensors(cls, qtensor):
        # Extract data for computation
        return qdata, scale, ...

# 2. Register operations
@register_layout_op(torch.ops.aten.linear.default, MyNewLayout)
def my_layout_linear(func, args, kwargs):
    # Implement efficient linear operation
    pass

# 3. Add to QUANT_ALGOS and LAYOUTS dicts
```

**ComfyUI compatibility**: Test with [quantization.examples.md](quantization.examples.md) patterns

### 4. Testing Quantization Quality

**Validation checklist**:
- ✅ Load model in ComfyUI without errors
- ✅ Visual quality comparison (generate same prompt, compare outputs)
- ✅ Memory usage reduction (check model size)
- ✅ Inference speed improvement
- ✅ Numerical accuracy (MSE, PSNR if applicable)

**Common commands**:
```bash
# FP8 with learned rounding
python convert_to_quant.py -i model.safetensors --comfy_quant --num_iter 1000

# INT8 block-wise
python convert_to_quant.py -i model.safetensors --int8 --comfy_quant --block_size 128

# Simple quantization (baseline)
python convert_to_quant.py -i model.safetensors --comfy_quant --simple
```

---

## Critical Patterns (from Architecture Guide)

### 1. Learned Rounding Core Concept
Standard quantization uses round-to-nearest. This workspace uses **SVD-based optimization** to minimize error in the most important weight directions:
- Compute truncated SVD: `W ≈ U_k @ S_k @ V_k^T`
- Optimize to minimize: `||U_k^T @ (W_dequant - W_orig) @ V_k||`

### 2. INT8 Gradient Derivation (Critical!)
For INT8 optimization: `∂L/∂Q = ∂L/∂dq * scale` (multiply by scale, not divide)
- Why: Dequantization is `dq = Q * scale`, so chain rule gives `∂dq/∂Q = scale`

### 3. Layout System Pattern
Each quantization format has a `QuantizedLayout` subclass with:
- `quantize()` - convert float → quantized
- `dequantize()` - convert quantized → float
- `get_plain_tensors()` - extract raw data for computation
- Registered operations via `@register_layout_op` decorator

### 4. Model-Specific Exclusions
Preserve sensitive layers in high precision:
```python
# Check against exclusion lists before quantizing
if any(name in key for name in AVOID_KEY_NAMES):
    skip_quantization()
```

### 5. Bias Correction with Synthetic Data
```python
# Generate random calibration data (no real dataset needed)
X_calib = torch.randn(samples, in_features)
weight_error = W_orig - W_dequant
bias_correction = (X_calib @ weight_error.T).mean(dim=0)
b_corrected = b_orig - bias_correction
```

---

## ComfyUI Integration Requirements

### Metadata Format
Quantized models need `.comfy_quant` tensor:
```python
comfy_quant = {
    "format": "float8_e4m3fn",  # or "int8_blockwise"
    "group_size": 128,  # for int8_blockwise
    # optional: "full_precision_matrix_mult": true
}
```

### Scale Tensors
- **FP8**: `weight_scale` (per-tensor or per-block)
- **INT8**: `weight_scale` (2D: M//bs × N//bs) + `input_scale` (1D: per-block)

### QuantizedTensor Compatibility
Must work with ComfyUI's `__torch_dispatch__` mechanism. See [quantization.examples.md](quantization.examples.md) for usage patterns.

---

## Gotchas (Quick Reference)

1. **INT8 transpose**: Transposing INT8 weights requires transposing scale tensor too
2. **Scale minimum**: Always clamp scale to `1e-8` to prevent division by zero
3. **Weight format**: Triton kernels expect `(N, K)` format, not `(K, N)`
4. **Device compatibility**: FP8 requires PyTorch 2.1+ and Ada/Hopper GPU
5. **Dimension requirements**: INT8 requires dimensions divisible by `block_size`

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for complete gotcha list and technical details.

---

## Quick Commands

```bash
# Basic FP8 quantization
python convert_to_quant.py -i model.safetensors --comfy_quant

# INT8 with performance heuristics
python convert_to_quant.py -i model.safetensors --int8 --comfy_quant --heur

# T5-XXL text encoder
python convert_to_quant.py -i t5xxl.safetensors --t5xxl --comfy_quant --block_size 64

# High-quality (slow) quantization
python convert_to_quant.py -i model.safetensors --comfy_quant --optimizer original --num_iter 2000

# Skip learned rounding (fast baseline)
python convert_to_quant.py -i model.safetensors --comfy_quant --simple

# Update ComfyUI reference materials
git submodule update --remote ComfyUI_Custom_Nodes_Agent
```

---

## Development Status & Roadmap

For tracking what's currently being worked on and planned improvements:

- **What's ready now?** → [ACTIVE.md](ACTIVE.md) - Current implementations, tested features, known issues
- **What's planned?** → [PLANNED.md](PLANNED.md) - Roadmap, research items, future improvements
- **Past findings?** → [DEVELOPMENT.md](DEVELOPMENT.md) - Experimental results and research notes

---

## When in Doubt

1. **Architecture questions?** → [.github/copilot-instructions.md](.github/copilot-instructions.md)
2. **What's being worked on?** → [ACTIVE.md](ACTIVE.md)
3. **What's planned?** → [PLANNED.md](PLANNED.md)
4. **ComfyUI integration?** → [quantization.examples.md](quantization.examples.md)
5. **Need to search ComfyUI source?** → [ComfyUI/](ComfyUI/)
6. **Python style?** → [python.instructions.md](python.instructions.md)
7. **ComfyUI patterns?** → [ComfyUI_Custom_Nodes_Agent/](ComfyUI_Custom_Nodes_Agent/)
8. **Usage docs?** → [MANUAL.md](MANUAL.md)
