# API Documentation

This document provides detailed API reference for developers using `convert_to_quant` programmatically.

## Core Functions

### `convert_to_int8()`

Main quantization function located in [`convert_to_quant/quantization.py`](../convert_to_quant/quantization.py).

```python
from convert_to_quant.quantization import convert_to_int8

convert_to_int8(
    input_file: str,
    output_file: str,
    comfy_quant: bool,
    filter_flags: Dict[str, bool],
    calib_samples: int,
    seed: int,
    fp16: bool = False,
    fallback: Optional[str] = None,
    custom_layers: Optional[str] = None,
    exclude_layers: Optional[str] = None,
    custom_type: Optional[str] = None,
    custom_block_size: Optional[int] = None,
    custom_scaling_mode: Optional[str] = None,
    custom_simple: bool = False,
    custom_heur: bool = False,
    fallback_block_size: Optional[int] = None,
    fallback_simple: bool = False,
    full_precision_matrix_mult: bool = False,
    skip_inefficient_layers: bool = False,
    include_input_scale: bool = False,
    no_learned_rounding: bool = False,
    save_quant_metadata: bool = False,
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    low_memory: bool = False,
    streaming_mode: str = "balanced",
    streaming_thresholds: Optional[Dict[str, Optional[int]]] = None,
    no_memory_limits: bool = False,
    report_quality: bool = False,
    quality_threshold: float = 30.0,
    smoothquant: bool = False,
    smoothquant_alpha: float = 0.5,
    calibration_data_path: Optional[str] = None,
    calibration_lora_path: Optional[str] = None,
    gptq_actorder: bool = False,
    gptq_fast: bool = True,
    gptq_turbo: bool = False,
    quip_actorder: bool = True,
    quip_hadamard: bool = True,
    quip_seed: Optional[int] = None,
    quip_store_transformed: bool = False,
    quip_requant_scheme: str = "tensor",
    quip_requant_tensor_per_row: bool = True,
    quip_checkpointed: bool = False,
    quip_checkpoint_threshold: int = 8192,
    quip_checkpoint_segments: int = 4,
    merge_lora_path: Optional[str] = None,
    merge_lora_paths: Optional[List[str]] = None,
    merge_lora_scale: float = 1.0,
    merge_lora_dampen: bool = True,
    **converter_kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | str | required | Path to input safetensors file |
| `output_file` | str | required | Path to output safetensors file |
| `comfy_quant` | bool | required | Enable ComfyUI-compatible quantization format |
| `filter_flags` | Dict[str, bool] | required | Model filter flags (e.g., `{"t5xxl": True}`) |
| `calib_samples` | int | required | Number of calibration samples |
| `seed` | int | required | Random seed for reproducibility |
| `fp16` | bool | False | Convert to FP16 instead of INT8 |
| `fallback` | str | None | Fallback quantization type (`"int8"` or `"fp16"`) |
| `custom_layers` | str | None | Regex pattern for custom quantization layers |
| `exclude_layers` | str | None | Regex pattern for layer exclusion |
| `custom_type` | str | None | Quantization type for custom layers |
| `custom_block_size` | int | None | Block size for custom layers |
| `custom_scaling_mode` | str | None | Scaling mode for custom layers |
| `custom_simple` | bool | False | Use simple quantization for custom layers |
| `custom_heur` | bool | False | Apply heuristics to custom layers |
| `fallback_block_size` | int | None | Block size for fallback layers |
| `fallback_simple` | bool | False | Use simple quantization for fallback layers |
| `full_precision_matrix_mult` | bool | False | Use full precision matrix multiplication |
| `skip_inefficient_layers` | bool | False | Skip layers with poor quantization characteristics |
| `include_input_scale` | bool | False | Include input scale tensor in output |
| `no_learned_rounding` | bool | False | Skip learned rounding optimization |
| `save_quant_metadata` | bool | False | Save quantization metadata in header |
| `layer_config` | Dict | None | Layer-specific configuration dictionary |
| `layer_config_fullmatch` | bool | False | Use fullmatch for layer config patterns |
| `low_memory` | bool | False | Use low-memory streaming mode (forces CPU) |
| `streaming_mode` | str | `"balanced"` | Streaming mode: `"off"`, `"minimal"`, `"balanced"`, `"aggressive"`, `"auto"` |
| `streaming_thresholds` | Dict | None | Manual overrides for streaming thresholds |
| `no_memory_limits` | bool | False | Disable all memory limits and OOM prevention |
| `report_quality` | bool | False | Output quality metrics (MSE, SQNR) |
| `quality_threshold` | float | 30.0 | SQNR threshold for warnings in dB |
| `smoothquant` | bool | False | Enable SmoothQuant preprocessing |
| `smoothquant_alpha` | float | 0.5 | SmoothQuant migration strength |
| `calibration_data_path` | str | None | Path to calibration data file |
| `calibration_lora_path` | str | None | Path to LoRA for calibration guidance |
| `gptq_actorder` | bool | False | Enable GPTQ activation ordering |
| `gptq_fast` | bool | True | Enable GPTQ vectorized processing |
| `gptq_turbo` | bool | False | Enable GPTQ Triton kernel |
| `quip_actorder` | bool | True | Enable activation ordering for QuIP |
| `quip_hadamard` | bool | True | Use Hadamard transform for QuIP |
| `quip_seed` | int | None | Seed for QuIP random matrices |
| `quip_store_transformed` | bool | False | Store QuIP weights in transformed space |
| `quip_requant_scheme` | str | `"tensor"` | Re-quantization scheme: `"tensor"` or `"block"` |
| `quip_requant_tensor_per_row` | bool | True | Use per-row scales for tensor-wise re-quantization |
| `quip_checkpointed` | bool | False | Enable checkpointed LDLQ quantization |
| `quip_checkpoint_threshold` | int | 8192 | Dimension threshold for checkpointing |
| `quip_checkpoint_segments` | int | 4 | Number of segments for checkpointing |
| `merge_lora_path` | str | None | Path to LoRA file to merge before quantization |
| `merge_lora_paths` | List[str] | None | Multiple LoRA files to merge |
| `merge_lora_scale` | float | 1.0 | Scale factor for LoRA merging |
| `merge_lora_dampen` | bool | True | Apply dampening for multiple LoRAs |

#### Example Usage

```python
from convert_to_quant.quantization import convert_to_int8

# Basic INT8 quantization
convert_to_int8(
    input_file="model.safetensors",
    output_file="model_int8.safetensors",
    comfy_quant=True,
    filter_flags={},
    calib_samples=6144,
    seed=42
)

# QuIP quantization with LoRA merging
convert_to_int8(
    input_file="base_model.safetensors",
    output_file="quantized_model.safetensors",
    comfy_quant=True,
    filter_flags={},
    calib_samples=6144,
    seed=42,
    optimizer="quip",
    merge_lora_path="style_lora.safetensors",
    merge_lora_scale=1.0
)
```

---

## Utility Classes

### `UnifiedSafetensorsLoader`

Memory-efficient loader for safetensors files located in [`convert_to_quant/utils/memory_efficient_loader.py`](../convert_to_quant/utils/memory_efficient_loader.py).

```python
from convert_to_quant.utils.memory_efficient_loader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    # Get all tensor keys
    keys = loader.keys()
    
    # Get metadata
    metadata = loader.metadata()
    
    # Get tensor shape without loading
    shape = loader.get_shape("layer.weight")
    
    # Load specific tensor
    tensor = loader.get_tensor("layer.weight")
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `keys()` | List[str] | Returns list of all tensor keys |
| `metadata()` | Dict | Returns file metadata dictionary |
| `get_shape(key)` | Tuple | Gets tensor shape without loading data |
| `get_ndim(key)` | int | Gets tensor dimensionality |
| `get_tensor(key)` | torch.Tensor | Loads and returns the tensor |
| `mark_processed(key)` | None | Marks tensor for cleanup |
| `close()` | None | Closes file handles and releases resources |

---

### `QuIPInt8Converter`

QuIP optimizer class located in [`convert_to_quant/converters/quip_int8.py`](../convert_to_quant/converters/quip_int8.py).

```python
from convert_to_quant.converters.quip_int8 import QuIPInt8Converter

converter = QuIPInt8Converter(
    block_size=128,
    device="cuda",
    actorder=True,
    use_hadamard=True,
    seed=None,
    use_triton=False,
    lazy_updates=True,
    store_transformed=False,
    requant_scheme="tensor",
    requant_tensor_use_per_row_scale=True,
    streaming_mode="balanced",
    streaming_thresholds={},
    no_memory_limits=False,
    use_checkpointed_ldlq=False,
    checkpointed_ldlq_threshold=8192,
    checkpoint_segments=4
)

# Convert a weight tensor
q_tensor, scale, dequantized = converter.convert(
    weight_tensor,
    H=None,  # Hessian matrix (optional)
    activation_scales=None,
    smoothquant_alpha=0.5
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | int | 128 | Block size for quantization |
| `device` | str | "cuda" | Device for computation |
| `actorder` | bool | True | Enable activation ordering |
| `use_hadamard` | bool | True | Use Hadamard transform |
| `seed` | int | None | Random seed for reproducibility |
| `use_triton` | bool | False | Use Triton kernels |
| `lazy_updates` | bool | True | Use lazy weight updates |
| `store_transformed` | bool | False | Store in transformed space |
| `requant_scheme` | str | `"tensor"` | Re-quantization scheme (`"tensor"`, `"block"`) |
| `requant_tensor_use_per_row_scale` | bool | True | Use per-row scales for tensor-wise |
| `streaming_mode` | str | `"balanced"` | Streaming mode tier |
| `streaming_thresholds` | Dict | `{}` | Manual threshold overrides |
| `no_memory_limits` | bool | False | Disable OOM protection |
| `use_checkpointed_ldlq` | bool | False | Enable checkpointed LDLQ |
| `checkpointed_ldlq_threshold` | int | 8192 | Threshold for checkpointing |
| `checkpoint_segments` | int | 4 | Segments for checkpointing |

---

### `LearnedRoundingConverter`

Standard learned rounding optimizer located in [`convert_to_quant/converters/learned_rounding.py`](../convert_to_quant/converters/learned_rounding.py).

```python
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter

converter = LearnedRoundingConverter(
    block_size=128,
    num_iter=1000,
    lr=0.008,
    optimizer="original",  # or "adamw", "radam"
    lr_schedule="adaptive",
    top_p=0.2,
    min_k=64,
    max_k=1024
)

q_tensor, scale, dequantized = converter.convert(
    weight_tensor,
    activation_scales=None
)
```

---

### `GPTQInt8Converter`

GPTQ optimizer class located in [`convert_to_quant/converters/gptq_int8.py`](../convert_to_quant/converters/gptq_int8.py).

```python
from convert_to_quant.converters.gptq_int8 import GPTQInt8Converter

converter = GPTQInt8Converter(
    block_size=128,
    device="cuda",
    actorder=True,
    lazy_updates=True,
    use_triton=False
)

# Convert with Hessian matrix
q_tensor, scale, dequantized = converter.convert(
    weight_tensor,
    H=hessian_matrix
)
```

---

### `QualityReporter`

Quality metrics reporter located in [`convert_to_quant/utils/quality_metrics.py`](../convert_to_quant/utils/quality_metrics.py).

```python
from convert_to_quant.utils.quality_metrics import QualityReporter

reporter = QualityReporter(threshold=30.0)

# Add layer results
reporter.add_layer("layer_name", original_tensor, dequantized_tensor)

# Get report
report = reporter.get_report_string()
print(report)
```

---

## LoRA Utilities

### `load_lora_tensors()`

Load LoRA tensors for calibration guidance.

```python
from convert_to_quant.utils.tensor_utils import load_lora_tensors

lora_tensors = load_lora_tensors("lora.safetensors")
# Returns: Dict[str, torch.Tensor] mapping layer names to LoRA weights
```

### `load_lora_for_merging()`

Load LoRA for merging into base model.

```python
from convert_to_quant.utils.tensor_utils import load_lora_for_merging

lora_data = load_lora_for_merging("lora.safetensors")
# Returns: Dict[str, Dict] with lora_A, lora_B, alpha, rank for each layer
```

### `merge_lora_into_weight()`

Merge single LoRA into weight tensor.

```python
from convert_to_quant.utils.tensor_utils import merge_lora_into_weight

merged_weight = merge_lora_into_weight(
    weight_tensor,
    lora_A,
    lora_B,
    alpha,
    rank,
    scale=1.0
)
```

### `merge_multiple_loras()`

Merge multiple LoRAs with dampening.

```python
from convert_to_quant.utils.tensor_utils import merge_multiple_loras

lora_configs = [
    {"lora_A": a1, "lora_B": b1, "alpha": alpha1, "rank": r1, "scale": 1.0},
    {"lora_A": a2, "lora_B": b2, "alpha": alpha2, "rank": r2, "scale": 0.9},
]

merged_weight = merge_multiple_loras(weight_tensor, lora_configs)
```

---

## Configuration

### Layer Config JSON

Fine-grained per-layer control via JSON configuration:

```python
from convert_to_quant.config.layer_config import load_layer_config

layer_config = load_layer_config("layer_config.json")

convert_to_int8(
    input_file="model.safetensors",
    output_file="output.safetensors",
    comfy_quant=True,
    filter_flags={},
    calib_samples=6144,
    seed=42,
    layer_config=layer_config
)
```

#### Layer Config Format

```json
{
  "layer_patterns": {
    ".*attention.*": {
      "format": "int8",
      "block_size": 64,
      "skip": false
    },
    ".*final_layer.*": {
      "format": "fp16",
      "skip": false
    }
  }
}
```

---

## Constants

Key constants from [`convert_to_quant/constants.py`](../convert_to_quant/constants.py):

```python
from convert_to_quant.constants import (
    TARGET_INT8_DTYPE,  # torch.int8
    COMPUTE_DTYPE,      # torch.float32
    SCALE_DTYPE,        # torch.float32
    INT8_SYMMETRIC_MAX, # 127
    MODEL_FILTERS       # Registry of model-specific filters
)
```

---

## Logging

Configure logging verbosity:

```python
from convert_to_quant.utils.logging import setup_logging

setup_logging("DEBUG")  # DEBUG, VERBOSE, NORMAL, MINIMAL
```

---

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ENABLE_MIXED_HADAMARD` | `auto`, `1`, `0` | `auto` | Enable mixed-precision Hadamard |
| `MIXED_HADAMARD_DTYPE` | `bfloat16`, `float16` | `bfloat16` | Mixed precision compute dtype |
| `STREAMING_AGGRESSIVENESS` | `conservative`, `balanced`, `aggressive` | `balanced` | Streaming mode thresholds |

---

## Type Aliases

Common type annotations used throughout the codebase:

```python
from typing import Dict, Any, Optional, List, Tuple
import torch

# Common tensor shapes
WeightTensor = torch.Tensor  # Shape: [out_features, in_features]
ScaleTensor = torch.Tensor   # Shape: [out_features, num_blocks] or scalar
HessianMatrix = torch.Tensor # Shape: [in_features, in_features]

# Configuration types
FilterFlags = Dict[str, bool]
LayerConfig = Dict[str, Any]
```
