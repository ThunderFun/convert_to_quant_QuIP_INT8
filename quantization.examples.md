# ComfyUI Quantization Integration Examples

This document provides ComfyUI integration patterns for quantization. These examples show how the quantization implementations in this workspace integrate with ComfyUI's inference runtime.

**Related files in this workspace**:
- [quant_ops.py](convert_to_quant/comfy/quant_ops.py) - Layout system matching ComfyUI's QuantizedTensor interface
- [convert_to_quant.py](convert_to_quant/convert_to_quant.py) - Generates ComfyUI-compatible quantized models
- [MANUAL.md](MANUAL.md) - Complete usage guide

---

## How This Workspace Fits with ComfyUI

1. **Development**: This workspace develops quantization methods (INT8 algorithms, learned rounding)
2. **Output**: Generates `.safetensors` files with `.comfy_quant` metadata compatible with ComfyUI
3. **Runtime**: ComfyUI loads these models using its `quant_ops.py` (mirrored in this workspace)
4. **Testing**: Load quantized models in ComfyUI to validate quality and performance

**Compatibility**: The `QuantizedTensor` and layout system in [quant_ops.py](convert_to_quant/comfy/quant_ops.py) matches ComfyUI's quantization interface.

---

## [TAG:quant:quantized-tensor]

QuantizedTensor class structure:

```python
from convert_to_quant.comfy.quant_ops import QuantizedTensor

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
from convert_to_quant.comfy.quant_ops import QuantizedLayout, register_layout_op
import torch

class MyCustomLayout(QuantizedLayout):
    """Custom quantization layout for specific use case."""
    
    @classmethod
    def quantize(cls, tensor, scale=None, dtype=torch.int8, **kwargs):
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
            scale = tensor.abs().max() / 127
        
        qdata = (tensor / scale).round().clamp(-128, 127).to(dtype)
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
from convert_to_quant.comfy.quant_ops import QuantizedTensor
# Note: ComfyUI's mixed_precision_ops can be configured to use these tensors
```

---

## Testing Quantized Models in ComfyUI

### Loading Quantized Models

Models quantized with this workspace can be loaded directly in ComfyUI:

```python
# ComfyUI automatically detects .comfy_quant metadata and creates QuantizedTensor wrappers
# No special loading code needed - just use the normal model loader

# The quantized model will have:
# - weight: QuantizedTensor (int8)
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

**Implementation reference**: See [convert_to_quant.py](convert_to_quant/convert_to_quant.py) bias correction logic for quality optimization techniques.

---

## LoRA Merging Workflow

### Converting a LoRA-Enhanced Model

Instead of loading LoRA separately at inference time, you can merge it into the base model and quantize the result:

```python
from convert_to_quant.quantization import convert_to_int8

# Merge single LoRA and quantize
convert_to_int8(
    input_file="base_model.safetensors",
    output_file="merged_quantized.safetensors",
    comfy_quant=True,
    merge_lora_path="style_lora.safetensors",
    merge_lora_scale=1.0,
    optimizer="quip"
)

# Merge multiple LoRAs with automatic dampening
convert_to_int8(
    input_file="base_model.safetensors",
    output_file="merged_multi_lora.safetensors",
    comfy_quant=True,
    merge_lora_paths=["style_lora.safetensors", "character_lora.safetensors"],
    merge_lora_scale=1.0,
    merge_lora_dampen=True,
    optimizer="quip"
)
```

### CLI Examples

```bash
# Merge and quantize with QuIP
convert_to_quant -i base_model.safetensors \
    --merge-lora style_lora.safetensors \
    --optimizer quip \
    --comfy_quant

# Merge multiple LoRAs with custom scale
convert_to_quant -i base_model.safetensors \
    --merge-loras lora1.safetensors lora2.safetensors \
    --merge-lora-scale 0.8 \
    --comfy_quant
```

---

## Memory-Efficient & Performance Options

### Streaming Mode for Low VRAM

Process large models on GPUs with limited VRAM by offloading heavy operations to CPU:

```bash
# Auto-detect best streaming settings based on available VRAM
convert_to_quant -i large_model.safetensors --streaming-mode auto --comfy_quant

# Aggressive streaming for <8GB VRAM GPUs
convert_to_quant -i large_model.safetensors --streaming-mode aggressive --comfy_quant
```

### BF16 Compute for Speed

Use BF16 precision for internal calculations on Ampere+ GPUs (RTX 30/40 series) to speed up quantization:

```bash
# Enable BF16 compute (auto mode uses BF16 for large tensors)
convert_to_quant -i model.safetensors --bf16-compute auto --comfy_quant
```

### Checkpointed Quantization

Extreme memory savings (75-90%) for very large layers (e.g., 16k+ dimensions) using gradient checkpointing-style recomputation:

```bash
# Enable checkpointed LDLQ for large layers
convert_to_quant -i model.safetensors --optimizer quip --quip-checkpointed --comfy_quant
```

---

## Advanced Utilities

### Dry Run & Analysis

Analyze a model before quantization to see which layers will be processed:

```bash
# Show analysis of layers and expected quantization formats
convert_to_quant -i model.safetensors --dry-run analyze --flux2
```

### Layer Config Template Generation

Generate a JSON template for fine-grained per-layer quantization control:

```bash
# Create a template JSON file based on the model structure
convert_to_quant -i model.safetensors --dry-run create-template
```

### Editing Existing Quantized Models

Modify metadata or remove/add tensors in an already quantized file:

```bash
# Remove specific tensors and update metadata
convert_to_quant -i quantized_model.safetensors --edit-quant \
    --remove-keys "layer1.weight_scale,layer2.weight_scale" \
    --save-quant-metadata
```

---

### Benefits of Pre-Merging LoRAs

1. **Single file deployment** - No separate LoRA loading needed at inference
2. **Faster inference** - No runtime LoRA computation overhead
3. **Better quantization quality** - QuIP can optimize for the merged weights rather than base + adapter separately
4. **Simpler workflow** - One quantized file contains everything needed

### Multi-LoRA Dampening

When merging multiple LoRAs, automatic dampening prevents over-saturation:

| LoRA Index | Scale Applied | Description |
|------------|---------------|-------------|
| 1st | 1.0 × `--merge-lora-scale` | Full strength |
| 2nd | 0.9 × `--merge-lora-scale` | 10% reduction |
| 3rd | 0.81 × `--merge-lora-scale` | 19% reduction |
| nth | 0.9^(n-1) × `--merge-lora-scale` | Progressive dampening |

Disable with `--merge-lora-dampen=False` if you want equal weighting.

---

## Custom Layout Development Workflow

### From Research to Integration

1. **Develop in this workspace**: Implement new quantization format in [quant_ops.py](convert_to_quant/comfy/quant_ops.py)
2. **Test with converter**: Use [convert_to_quant.py](convert_to_quant/convert_to_quant.py) to quantize models
3. **Validate in ComfyUI**: Load and test quantized models
4. **Document findings**: Record results in research notes
5. **Refine implementation**: Iterate based on quality/performance metrics

**Example**: The INT8 block-wise layout was developed using this workflow.
