# ComfyUI Quantization Integration Examples

This document provides ComfyUI integration patterns for quantization. These examples show how the quantization implementations in this workspace integrate with ComfyUI's inference runtime.

**Related files in this workspace**:
- [quant_ops.py](quant_ops.py) - Layout system matching ComfyUI's QuantizedTensor interface
- [convert_to_quant.py](convert_to_quant.py) - Generates ComfyUI-compatible quantized models
- [AGENTS.md](AGENTS.md) - Development workflows for this workspace
- [ComfyUI_Custom_Nodes_Agent/](ComfyUI_Custom_Nodes_Agent/) - Reference patterns for ComfyUI development

---

## How This Workspace Fits with ComfyUI

1. **Development**: This workspace develops quantization methods (FP8/INT8 algorithms, learned rounding)
2. **Output**: Generates `.safetensors` files with `.comfy_quant` metadata compatible with ComfyUI
3. **Runtime**: ComfyUI loads these models using its `quant_ops.py` (mirrored in this workspace)
4. **Testing**: Load quantized models in ComfyUI to validate quality and performance

**Compatibility**: The `QuantizedTensor` and layout system in [quant_ops.py](quant_ops.py) matches ComfyUI's quantization interface.

---

## [TAG:quant:quantized-tensor]

QuantizedTensor class structure:

```python
from comfy.quant_ops import QuantizedTensor

class QuantizedTensor(torch.Tensor):
    _qdata: torch.Tensor      # Quantized data storage
    _layout_type: str         # Layout identifier (e.g., "TensorCoreFP8Layout")
    _layout_params: dict      # Scale, orig_dtype, etc.
    
    @classmethod
    def from_float(cls, tensor, layout_type, **kwargs):
        """Create quantized tensor from float tensor."""
        pass
    
    def dequantize(self):
        """Convert back to original dtype."""
        pass
```

### [TAG:quant:custom-layout]

Creating custom quantization layouts:

```python
from comfy.quant_ops import QuantizedLayout, register_layout_op
import torch

class MyCustomLayout(QuantizedLayout):
    """Custom quantization layout for specific use case."""
    
    @classmethod
    def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs):
        """
        Quantize a float tensor.
        
        Args:
            tensor: Input float tensor
            scale: Quantization scale (computed if None)
            dtype: Target quantized dtype
            
        Returns:
            Tuple of (quantized_data, layout_params_dict)
        """
        if scale is None:
            scale = tensor.abs().max() / torch.finfo(dtype).max
        
        qdata = (tensor / scale).to(dtype)
        layout_params = {
            "scale": scale,
            "orig_dtype": tensor.dtype,
        }
        return qdata, layout_params
    
    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        """Dequantize back to original dtype."""
        return qdata.to(orig_dtype) * scale


# Register custom operation handler for your layout
@register_layout_op(torch.ops.aten.linear.default, "MyCustomLayout")
def my_custom_linear(func, args, kwargs):
    """
    Custom linear operation for MyCustomLayout tensors.
    
    Args:
        func: Original torch function
        args: Positional arguments (input, weight, bias)
        kwargs: Keyword arguments
    """
    input_tensor, weight, bias = args[0], args[1], args[2] if len(args) > 2 else None
    
    # Dequantize weight if needed
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    
    # Perform operation
    return torch.nn.functional.linear(input_tensor, weight, bias)
```

### [TAG:quant:mixed-precision]

Mixed precision operations factory:

```python
from comfy.ops import mixed_precision_ops

# Create ops class with quantization config
quant_config = {
    "layer_name.weight": {"dtype": torch.float8_e4m3fn, "scale": 0.1},
    # ... per-layer configs
}

CustomQuantOps = mixed_precision_ops(
    quant_config=quant_config,
    compute_dtype=torch.bfloat16,
    full_precision_mm=False
)

# Use in model loading
model_options = {"custom_operations": CustomQuantOps}
```

---

## Testing Quantized Models in ComfyUI

### Loading Quantized Models

Models quantized with this workspace can be loaded directly in ComfyUI:

```python
# ComfyUI automatically detects .comfy_quant metadata and creates QuantizedTensor wrappers
# No special loading code needed - just use the normal model loader

# The quantized model will have:
# - weight: QuantizedTensor (int8 or fp8)
# - weight_scale: float32 tensor
# - input_scale: float32 tensor (for INT8)
# - .comfy_quant: metadata tensor
```

### Validation Workflow

1. **Load model** in ComfyUI using standard loader nodes
2. **Generate test images** with known prompts
3. **Compare outputs** with original (non-quantized) model
4. **Check metrics**: visual quality, inference speed, memory usage
5. **Test edge cases**: long prompts, high resolution, multiple batches

**Implementation reference**: See [convert_to_quant.py](convert_to_quant.py) bias correction logic for quality optimization techniques.

---

## Custom Layout Development Workflow

### From Research to Integration

1. **Develop in this workspace**: Implement new quantization format in [quant_ops.py](quant_ops.py)
2. **Test with converter**: Use [convert_to_quant.py](convert_to_quant.py) to quantize models
3. **Validate in ComfyUI**: Load and test quantized models
4. **Document findings**: Record results in [DEVELOPMENT.md](DEVELOPMENT.md)
5. **Refine implementation**: Iterate based on quality/performance metrics

**Example**: The INT8 block-wise layout was developed using this workflow.

---

### [TAG:quant:node-example]

Quantization-aware custom node:

```python
import torch
import comfy.ops
import comfy.model_management
from comfy.quant_ops import QuantizedTensor

class QuantizeModelWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "quantize_dtype": (["fp8_e4m3fn", "fp8_e5m2", "int8"],),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    CATEGORY = "custom/quantization"
    
    def quantize(self, model, quantize_dtype):
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "int8": torch.int8,
        }
        target_dtype = dtype_map[quantize_dtype]
        
        # Clone model to avoid modifying original
        model = model.clone()
        
        # Apply quantization via model patcher
        def quantize_weight(weight, **kwargs):
            if weight.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                scale = weight.abs().max() / torch.finfo(target_dtype).max
                return (weight / scale).to(target_dtype), {"scale": scale}
            return weight, {}
        
        # Add weight hook
        model.add_weight_hook(quantize_weight)
        
        return (model,)
```

### [TAG:quant:fp8-ops]

FP8 operations reference:

```python
import comfy.ops

# Check if FP8 is available
if comfy.model_management.supports_fp8_compute(device):
    # Use FP8 optimized operations
    model_options = {
        "custom_operations": comfy.ops.fp8_ops,
        "fp8_optimizations": True,
    }
```
