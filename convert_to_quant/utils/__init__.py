"""Utils package for convert_to_quant."""
from .tensor_utils import (
    dict_to_tensor,
    tensor_to_dict,
    normalize_tensorwise_scales,
    generate_calibration_data,
    adaptive_lr_update,
    compute_bias_correction,
)
from .comfy_quant import (
    create_comfy_quant_tensor,
    fix_comfy_quant_params_structure,
    parse_add_keys_string,
    edit_comfy_quant,
    should_skip_layer_for_performance,
)
from .logging import (
    setup_logging,
    info,
    verbose,
    debug,
    minimal,
    warning,
    error,
    log_debug,
)
from .hadamard import (
    is_power_of_two,
    hadamard_matrix,
    fast_hadamard_transform,
    random_orthogonal_matrix,
)

__all__ = [
    "dict_to_tensor",
    "tensor_to_dict",
    "normalize_tensorwise_scales",
    "generate_calibration_data",
    "adaptive_lr_update",
    "compute_bias_correction",
    "create_comfy_quant_tensor",
    "fix_comfy_quant_params_structure",
    "parse_add_keys_string",
    "edit_comfy_quant",
    "should_skip_layer_for_performance",
    "setup_logging",
    "info",
    "verbose",
    "debug",
    "minimal",
    "warning",
    "error",
    "log_debug",
    "is_power_of_two",
    "hadamard_matrix",
    "fast_hadamard_transform",
    "random_orthogonal_matrix",
]
