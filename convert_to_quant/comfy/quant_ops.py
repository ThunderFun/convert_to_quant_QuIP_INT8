import torch
import logging
from typing import Tuple, Dict

_LAYOUT_REGISTRY = {}
_GENERIC_UTILS = {}

# Try to import Triton-based INT8 kernels
try:
    from .int8_kernels import (
        act_quant as triton_act_quant,
        act_dequant as triton_act_dequant,
        weight_quant as triton_weight_quant,
        weight_dequant as triton_weight_dequant,
        int8_gemm as triton_int8_gemm,
        int8_addmm as triton_int8_addmm,
        int8_gemm_quant as triton_int8_gemm_quant,
        int8_addmm_quant as triton_int8_addmm_quant,
    )
    _HAS_TRITON_INT8 = True
except ImportError:
    _HAS_TRITON_INT8 = False
    logging.warning("Triton INT8 kernels not available, using PyTorch fallback")

# Try to import optimized INT8 matmul kernels
try:
    from .int8_matmul_kernels import mm_8bit as optimized_mm_8bit
    _HAS_OPTIMIZED_KERNELS = True
except ImportError:
    _HAS_OPTIMIZED_KERNELS = False

def register_layout_op(torch_op, layout_type):
    def decorator(handler_func):
        if torch_op not in _LAYOUT_REGISTRY:
            _LAYOUT_REGISTRY[torch_op] = {}
        _LAYOUT_REGISTRY[torch_op][layout_type] = handler_func
        return handler_func
    return decorator

def register_generic_util(torch_op):
    def decorator(handler_func):
        _GENERIC_UTILS[torch_op] = handler_func
        return handler_func
    return decorator

def _get_layout_from_args(args):
    for arg in args:
        if isinstance(arg, QuantizedTensor):
            return arg._layout_type
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, QuantizedTensor):
                    return item._layout_type
    return None

def _move_layout_params_to_device(params, device):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.to(device=device)
        else:
            new_params[k] = v
    return new_params

def _copy_layout_params(params):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.clone()
        else:
            new_params[k] = v
    return new_params

def _copy_layout_params_inplace(src, dst, non_blocking=False):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v, non_blocking=non_blocking)
        else:
            dst[k] = v

class QuantizedLayout:
    @classmethod
    def quantize(cls, tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError(f"{cls.__name__} must implement quantize()")

    @staticmethod
    def dequantize(qdata, **layout_params) -> torch.Tensor:
        raise NotImplementedError("TensorLayout must implement dequantize()")

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__} must implement get_plain_tensors()")

class QuantizedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, qdata, layout_type, layout_params):
        return torch.Tensor._make_wrapper_subclass(
            cls, qdata.shape, device=qdata.device, dtype=qdata.dtype, requires_grad=False,
        )

    def __init__(self, qdata, layout_type, layout_params):
        self._qdata = qdata
        self._layout_type = layout_type
        self._layout_params = layout_params

    def __repr__(self):
        return f"QuantizedTensor(shape={self.shape}, layout={self._layout_type})"

    @property
    def layout_type(self):
        return self._layout_type

    def __tensor_flatten__(self):
        inner_tensors = ["_qdata"]
        ctx = {"layout_type": self._layout_type}
        tensor_params = {}
        non_tensor_params = {}
        for k, v in self._layout_params.items():
            if isinstance(v, torch.Tensor):
                tensor_params[k] = v
            else:
                non_tensor_params[k] = v
        ctx["tensor_param_keys"] = list(tensor_params.keys())
        ctx["non_tensor_params"] = non_tensor_params
        for k, v in tensor_params.items():
            attr_name = f"_layout_param_{k}"
            object.__setattr__(self, attr_name, v)
            inner_tensors.append(attr_name)
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        layout_type = ctx["layout_type"]
        layout_params = dict(ctx["non_tensor_params"])
        for key in ctx["tensor_param_keys"]:
            attr_name = f"_layout_param_{key}"
            layout_params[key] = inner_tensors[attr_name]
        return QuantizedTensor(inner_tensors["_qdata"], layout_type, layout_params)

    @classmethod
    def from_float(cls, tensor, layout_type, **quantize_kwargs) -> "QuantizedTensor":
        qdata, layout_params = LAYOUTS[layout_type].quantize(tensor, **quantize_kwargs)
        return cls(qdata, layout_type, layout_params)

    def dequantize(self) -> torch.Tensor:
        return LAYOUTS[self._layout_type].dequantize(self._qdata, **self._layout_params)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _GENERIC_UTILS:
            return _GENERIC_UTILS[func](func, args, kwargs)
        layout_type = _get_layout_from_args(args)
        if layout_type and func in _LAYOUT_REGISTRY:
            handler = _LAYOUT_REGISTRY[func].get(layout_type)
            if handler:
                return handler(func, args, kwargs)
        return cls._dequant_and_fallback(func, args, kwargs)

    @classmethod
    def _dequant_and_fallback(cls, func, args, kwargs):
        def dequant_arg(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize()
            elif isinstance(arg, (list, tuple)):
                return type(arg)(dequant_arg(a) for a in arg)
            return arg
        new_args = dequant_arg(args)
        new_kwargs = dequant_arg(kwargs)
        return func(*new_args, **new_kwargs)

    def data_ptr(self): return self._qdata.data_ptr()
    def is_pinned(self): return self._qdata.is_pinned()
    def is_contiguous(self, *arg, **kwargs): return self._qdata.is_contiguous(*arg, **kwargs)
    def storage(self): return self._qdata.storage()

def _create_transformed_qtensor(qt, transform_fn):
    new_data = transform_fn(qt._qdata)
    new_params = _copy_layout_params(qt._layout_params)
    return QuantizedTensor(new_data, qt._layout_type, new_params)

def _handle_device_transfer(qt, target_device, target_dtype=None, target_layout=None, op_name="to"):
    current_device = qt._qdata.device
    if target_device is not None:
        if isinstance(target_device, str): target_device = torch.device(target_device)
        if isinstance(current_device, str): current_device = torch.device(current_device)
        if target_device != current_device:
            new_q_data = qt._qdata.to(device=target_device)
            new_params = _move_layout_params_to_device(qt._layout_params, target_device)
            if target_dtype is not None: new_params["orig_dtype"] = target_dtype
            return QuantizedTensor(new_q_data, qt._layout_type, new_params)
    return qt

@register_generic_util(torch.ops.aten.detach.default)
def generic_detach(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor): return _create_transformed_qtensor(qt, lambda x: x.detach())
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten.clone.default)
def generic_clone(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor): return _create_transformed_qtensor(qt, lambda x: x.clone())
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten._to_copy.default)
def generic_to_copy(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(qt, target_device=kwargs.get("device"), target_dtype=kwargs.get("dtype"), op_name="_to_copy")
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten.to.dtype_layout)
def generic_to_dtype_layout(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(qt, target_device=kwargs.get("device"), target_dtype=kwargs.get("dtype"), target_layout=kwargs.get("layout"), op_name="to")
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten.to.dtype)
def generic_to_dtype(func, args, kwargs):
    src = args[0]
    if isinstance(src, QuantizedTensor):
        target_dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
        src._layout_params["orig_dtype"] = target_dtype
        return src
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten.copy_.default)
def generic_copy_(func, args, kwargs):
    qt_dest, src = args[0], args[1]
    non_blocking = args[2] if len(args) > 2 else False
    if isinstance(qt_dest, QuantizedTensor):
        if isinstance(src, QuantizedTensor):
            qt_dest._qdata.copy_(src._qdata, non_blocking=non_blocking)
            qt_dest._layout_type = src._layout_type
            orig_dtype = qt_dest._layout_params["orig_dtype"]
            _copy_layout_params_inplace(src._layout_params, qt_dest._layout_params, non_blocking=non_blocking)
            qt_dest._layout_params["orig_dtype"] = orig_dtype
        else:
            qt_dest._qdata.copy_(src)
        return qt_dest
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten._has_compatible_shallow_copy_type.default)
def generic_has_compatible_shallow_copy_type(func, args, kwargs): return True

@register_generic_util(torch.ops.aten.empty_like.default)
def generic_empty_like(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        hp_dtype = kwargs.pop("dtype", qt._layout_params["orig_dtype"])
        new_qdata = torch.empty_like(qt._qdata, **kwargs)
        target_device = kwargs.get("device", new_qdata.device)
        new_params = _move_layout_params_to_device(qt._layout_params, target_device)
        new_params["orig_dtype"] = hp_dtype
        return QuantizedTensor(new_qdata, qt._layout_type, new_params)
    return func(*args, **kwargs)

class BlockWiseINT8Layout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, scale=None, block_size=128, is_weight=False, **kwargs):
        orig_dtype = tensor.dtype
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        if is_weight:
            M, N = tensor.shape
            if _HAS_TRITON_INT8 and scale is None and tensor.is_cuda:
                qdata, scale = triton_weight_quant(tensor, block_size=block_size)
            else:
                qdata, scale = cls._weight_quantize_pytorch(tensor, block_size, scale)
        else:
            if _HAS_TRITON_INT8 and tensor.is_cuda:
                qdata, scale = triton_act_quant(tensor, block_size=block_size)
            else:
                qdata, scale = cls._activation_quantize_pytorch(tensor, block_size, scale)
        return qdata, {"scale": scale.to(torch.float32), "block_size": block_size, "is_weight": is_weight, "orig_dtype": orig_dtype}

    @staticmethod
    def _weight_quantize_pytorch(tensor, block_size, scale=None):
        M, N = tensor.shape
        tensor_blocked = tensor.reshape(M // block_size, block_size, N // block_size, block_size).permute(0, 2, 1, 3)
        if scale is None:
            scale = torch.maximum(tensor_blocked.abs().amax(dim=(-2, -1)) / 127.0, torch.tensor(1e-8, device=tensor.device, dtype=tensor.dtype))
        qdata = torch.clamp(tensor_blocked / scale.unsqueeze(-1).unsqueeze(-1), -127.0, 127.0).to(torch.int8)
        return qdata.permute(0, 2, 1, 3).contiguous().reshape(M, N), scale

    @staticmethod
    def _activation_quantize_pytorch(tensor, block_size, scale=None):
        K = tensor.shape[-1]
        batch_shape = tensor.shape[:-1]
        tensor_blocked = tensor.reshape(*batch_shape, K // block_size, block_size)
        if scale is None:
            scale = torch.maximum(tensor_blocked.abs().amax(dim=-1) / 127.0, torch.tensor(1e-8, device=tensor.device, dtype=tensor.dtype))
        qdata = torch.clamp(tensor_blocked / scale.unsqueeze(-1), -127.0, 127.0).to(torch.int8)
        return qdata.reshape(tensor.shape), scale

    @staticmethod
    def dequantize(qdata, scale, block_size, is_weight=False, orig_dtype=None, output_dtype=None, **kwargs):
        if not qdata.is_contiguous(): qdata = qdata.contiguous()
        if not scale.is_contiguous(): scale = scale.contiguous()
        out_dtype = output_dtype if output_dtype is not None else orig_dtype
        if is_weight:
            if _HAS_TRITON_INT8 and qdata.dim() == 2 and qdata.is_cuda:
                return triton_weight_dequant(qdata, scale, block_size=block_size, output_dtype=out_dtype)
            M, N = qdata.shape
            qdata_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size).permute(0, 2, 1, 3)
            return (qdata_blocked.to(orig_dtype) * scale.reshape(M // block_size, N // block_size).unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).contiguous().reshape(M, N)
        else:
            if _HAS_TRITON_INT8 and qdata.is_cuda:
                return triton_act_dequant(qdata, scale, block_size=block_size, output_dtype=out_dtype)
            batch_shape, K = qdata.shape[:-1], qdata.shape[-1]
            qdata_blocked = qdata.reshape(*batch_shape, K // block_size, block_size)
            return (qdata_blocked.to(orig_dtype) * scale.reshape(*batch_shape, K // block_size).unsqueeze(-1)).reshape(qdata.shape)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._layout_params["scale"], qtensor._layout_params["block_size"], qtensor._layout_params["is_weight"]

class TensorWiseINT8Layout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, scale=None, **kwargs):
        from ..utils.float_utils import quantize_int8_tensorwise
        qdata, scale = quantize_int8_tensorwise(tensor)
        return qdata, {"scale": scale, "orig_dtype": tensor.dtype}
    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs): return qdata.to(orig_dtype) * scale
    @classmethod
    def get_plain_tensors(cls, qtensor): return qtensor._qdata, qtensor._layout_params["scale"]

class AxisWiseINT8Layout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, scale=None, **kwargs):
        from ..utils.float_utils import quantize_int8_axiswise
        qdata, scale = quantize_int8_axiswise(tensor)
        return qdata, {"scale": scale.squeeze(1), "orig_dtype": tensor.dtype}
    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs): return qdata.to(orig_dtype) * scale.unsqueeze(1)
    @classmethod
    def get_plain_tensors(cls, qtensor): return qtensor._qdata, qtensor._layout_params["scale"]

QUANT_ALGOS = {
    "int8_blockwise": {
        "storage_t": torch.int8,
        "required_parameters": {"weight_scale"},
        "optional_parameters": {"input_scale"},  # Only present with static activation quantization
        "comfy_tensor_layout": "BlockWiseINT8Layout",
        "group_size": 128,
        "asymmetric_layout": True
    },
}

LAYOUTS = {
    "BlockWiseINT8Layout": BlockWiseINT8Layout,
    "TensorWiseINT8Layout": TensorWiseINT8Layout,
    "AxisWiseINT8Layout": AxisWiseINT8Layout,
}

def _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias=None, out_quant=False):
    K, batch_shape, N = a_int8.shape[-1], a_int8.shape[:-1], b_int8.shape[0]
    if _HAS_TRITON_INT8 and a_int8.is_cuda:
        a_2d, a_scale_2d = a_int8.reshape(-1, K).contiguous(), a_scale.reshape(-1, a_scale.shape[-1]).contiguous()
        if out_quant:
            res_2d, res_s_2d = triton_int8_addmm_quant(a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous(), bias, out_block_size=block_size) if bias is not None else triton_int8_gemm_quant(a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous(), out_block_size=block_size)
            return res_2d.reshape(*batch_shape, N), res_s_2d.reshape(*batch_shape, N // block_size)
        res_2d = triton_int8_addmm(a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous(), bias) if bias is not None else triton_int8_gemm(a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous())
        return res_2d.reshape(*batch_shape, N)
    
    a_fp32 = (a_int8.reshape(*batch_shape, K // block_size, block_size).to(torch.float32) * a_scale.reshape(*batch_shape, K // block_size).unsqueeze(-1)).reshape(*batch_shape, K)
    b_fp32 = (b_int8.reshape(N // block_size, block_size, K // block_size, block_size).permute(0, 2, 1, 3).to(torch.float32) * b_scale.reshape(N // block_size, K // block_size).unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(N, K)
    res = torch.nn.functional.linear(a_fp32, b_fp32, bias)
    return BlockWiseINT8Layout._activation_quantize_pytorch(res, block_size) if out_quant else res

@register_layout_op(torch.ops.aten.linear.default, "BlockWiseINT8Layout")
def int8_linear(func, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        a_int8, a_scale, a_bs, _ = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_bs, _ = BlockWiseINT8Layout.get_plain_tensors(weight)
        out_dtype, out_quant = kwargs.get("out_dtype", input_tensor._layout_params["orig_dtype"]), kwargs.get("out_quant", False)
        res = _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, a_bs, bias=bias, out_quant=out_quant)
        if out_quant:
            return QuantizedTensor(res[0], "BlockWiseINT8Layout", {"scale": res[1], "block_size": a_bs, "is_weight": False, "orig_dtype": out_dtype})
        return res.to(out_dtype)
    return torch.nn.functional.linear(input_tensor.dequantize() if isinstance(input_tensor, QuantizedTensor) else input_tensor, weight.dequantize() if isinstance(weight, QuantizedTensor) else weight, bias)

@register_layout_op(torch.ops.aten.mm.default, "BlockWiseINT8Layout")
def int8_mm(func, args, kwargs):
    input_tensor, weight = args[0], args[1]
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        a_int8, a_scale, a_bs, _ = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_bs, _ = BlockWiseINT8Layout.get_plain_tensors(weight)
        out_dtype, out_quant = kwargs.get("out_dtype", input_tensor._layout_params["orig_dtype"]), kwargs.get("out_quant", False)
        K = a_int8.shape[-1]
        if b_int8.shape[0] == K and b_int8.shape[1] != K:
            b_int8, b_scale = b_int8.t().contiguous(), b_scale.t().contiguous()
        res = _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, a_bs, out_quant=out_quant)
        if out_quant:
            return QuantizedTensor(res[0], "BlockWiseINT8Layout", {"scale": res[1], "block_size": a_bs, "is_weight": False, "orig_dtype": out_dtype})
        return res.to(out_dtype)
    return func(args[0].dequantize() if isinstance(args[0], QuantizedTensor) else args[0], args[1].dequantize() if isinstance(args[1], QuantizedTensor) else args[1])

@register_layout_op(torch.ops.aten.addmm.default, "BlockWiseINT8Layout")
def int8_addmm(func, args, kwargs):
    bias, input_tensor, weight = args[0], args[1], args[2]
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        a_int8, a_scale, a_bs, _ = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_bs, _ = BlockWiseINT8Layout.get_plain_tensors(weight)
        out_dtype, out_quant = kwargs.get("out_dtype", input_tensor._layout_params["orig_dtype"]), kwargs.get("out_quant", False)
        K = a_int8.shape[-1]
        if b_int8.shape[0] == K:
            b_int8, b_scale = b_int8.t().contiguous(), b_scale.t().contiguous()
        res = _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, a_bs, bias=bias, out_quant=out_quant)
        if out_quant:
            return QuantizedTensor(res[0], "BlockWiseINT8Layout", {"scale": res[1], "block_size": a_bs, "is_weight": False, "orig_dtype": out_dtype})
        return res.to(out_dtype)
    return func(args[0].dequantize() if isinstance(args[0], QuantizedTensor) else args[0], args[1].dequantize() if isinstance(args[1], QuantizedTensor) else args[1], args[2].dequantize() if isinstance(args[2], QuantizedTensor) else args[2], **kwargs)

@register_layout_op(torch.ops.aten.view.default, "BlockWiseINT8Layout")
def int8_view(func, args, kwargs):
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        return QuantizedTensor(func(input_tensor._qdata, *args[1:], **kwargs), "BlockWiseINT8Layout", input_tensor._layout_params)
    return func(*args, **kwargs)

@register_layout_op(torch.ops.aten.t.default, "BlockWiseINT8Layout")
def int8_transpose(func, args, kwargs):
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        new_params = input_tensor._layout_params.copy()
        if new_params.get("is_weight"): new_params["scale"] = new_params["scale"].t().contiguous()
        return QuantizedTensor(func(input_tensor._qdata, *args[1:], **kwargs), "BlockWiseINT8Layout", new_params)
    return func(*args, **kwargs)

@register_layout_op(torch.ops.aten.gelu.default, "BlockWiseINT8Layout")
def int8_gelu(func, args, kwargs):
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        qdata, scale, bs, _ = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        if _HAS_TRITON_INT8 and qdata.is_cuda:
            from .int8_kernels import int8_gelu as triton_int8_gelu
            res_q, res_s = triton_int8_gelu(qdata, scale, block_size=bs)
            return QuantizedTensor(res_q, "BlockWiseINT8Layout", {"scale": res_s.to(torch.float32), "block_size": bs, "is_weight": False, "orig_dtype": input_tensor._layout_params["orig_dtype"]})
        res_fp = torch.nn.functional.gelu(input_tensor.dequantize())
        res_q, res_params = BlockWiseINT8Layout.quantize(res_fp, block_size=bs, is_weight=False)
        res_params["orig_dtype"] = input_tensor._layout_params["orig_dtype"]
        return QuantizedTensor(res_q, "BlockWiseINT8Layout", res_params)
    return func(*args, **kwargs)

@register_layout_op(torch.ops.aten.add_.Tensor, "BlockWiseINT8Layout")
def int8_add_(func, args, kwargs):
    target = args[0]
    if isinstance(target, QuantizedTensor) and target._layout_params.get("is_weight"):
        res_fp = target.dequantize() + (args[1].dequantize() if isinstance(args[1], QuantizedTensor) else args[1])
        res_q, res_params = BlockWiseINT8Layout.quantize(res_fp, block_size=target._layout_params["block_size"], is_weight=True)
        target._qdata.copy_(res_q)
        target._layout_params["scale"].copy_(res_params["scale"])
        return target
    return QuantizedTensor._dequant_and_fallback(func, args, kwargs)

@register_layout_op(torch.ops.aten.linear.default, "TensorWiseINT8Layout")
@register_layout_op(torch.ops.aten.linear.default, "AxisWiseINT8Layout")
def optimized_linear(func, args, kwargs):
    input_tensor, weight = args[0], args[1]
    if isinstance(weight, QuantizedTensor):
        from ..utils.float_utils import quantize_int8_axiswise
        w_q, w_s = weight.layout_type.get_plain_tensors(weight)
        if not isinstance(input_tensor, QuantizedTensor):
            x_q, x_s = quantize_int8_axiswise(input_tensor.reshape(-1, input_tensor.shape[-1]))
            if _HAS_OPTIMIZED_KERNELS and x_q.is_cuda:
                res = optimized_mm_8bit(x_q, w_q.t().contiguous())
                res_scaled = res.float() * (x_s * w_s) if w_s.ndim == 0 else res.float() * (x_s * w_s.unsqueeze(0))
                if len(args) > 2 and args[2] is not None: res_scaled += args[2]
                return res_scaled.reshape(input_tensor.shape[:-1] + (weight.shape[0],)).to(input_tensor.dtype)
    return QuantizedTensor._dequant_and_fallback(func, args, kwargs)
