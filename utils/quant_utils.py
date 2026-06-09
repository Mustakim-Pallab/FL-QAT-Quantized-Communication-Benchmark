import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from config import KEEP_SMALL_TENSORS_FP32


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, bits):
        return fake_quantize_tensor(tensor, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class WeightFakeQuantParametrization(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def forward(self, weight):
        return FakeQuantSTE.apply(weight, self.bits)


def enable_weight_qat(model, bits):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad:
            if not parametrize.is_parametrized(module, "weight"):
                parametrize.register_parametrization(
                    module,
                    "weight",
                    WeightFakeQuantParametrization(bits),
                )


def remove_weight_qat(model, leave_quantized=True):
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module,
                "weight",
                leave_parametrized=leave_quantized,
            )


def should_quantize_tensor(tensor, key=None):
    if not torch.is_floating_point(tensor):
        return False
    if KEEP_SMALL_TENSORS_FP32 and tensor.dim() <= 1:
        return False
    if key is not None:
        key_parts = key.split(".")
        parameter_name = key_parts[-1]
        if parameter_name not in {"weight", "in_proj_weight"}:
            return False
        sensitive_names = {
            "class_token",
            "memory_tokens",
            "pos_embedding",
        }
        if any(part in sensitive_names for part in key_parts):
            return False
    return True


def fake_quantize_tensor(tensor, bits, per_channel=True):
    if not torch.is_floating_point(tensor):
        return tensor.detach().clone()

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    if per_channel and tensor.dim() >= 2:
        reduce_dims = tuple(range(1, tensor.dim()))
        max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.where(
            max_abs > 0,
            max_abs / qmax,
            torch.ones_like(max_abs),
        )
        q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return (q * scale).to(dtype=tensor.dtype)

    max_abs = tensor.detach().abs().max()

    if max_abs == 0:
        return tensor.detach().clone()

    scale = max_abs / qmax
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return (q * scale).to(dtype=tensor.dtype)


def quantization_scale_count(tensor):
    if tensor.dim() >= 2:
        return tensor.shape[0]
    return 1


def communication_tensor_size_mb(tensor, bits=None, key=None):
    if bits is not None and should_quantize_tensor(tensor, key):
        return (tensor.numel() * bits / 8 + quantization_scale_count(tensor) * 4) / 1e6
    return tensor.numel() * tensor.element_size() / 1e6


def quantize_for_communication(state_dict, keys, bits):
    dequantized = OrderedDict()
    total_bytes = 0.0

    for key in keys:
        value = state_dict[key].detach().cpu()

        if should_quantize_tensor(value, key):
            dequantized[key] = fake_quantize_tensor(value, bits)
            total_bytes += value.numel() * bits / 8 + quantization_scale_count(value) * 4
        else:
            dequantized[key] = value.clone()
            total_bytes += value.numel() * value.element_size()

    return dequantized, total_bytes / 1e6


def fp32_for_communication(state_dict, keys):
    payload = OrderedDict((key, state_dict[key].detach().cpu().clone()) for key in keys)
    total_bytes = sum(value.numel() * value.element_size() for value in payload.values())
    return payload, total_bytes / 1e6


def model_delta(client_state, server_state, keys):
    delta = OrderedDict()
    for key in keys:
        client_value = client_state[key].detach().cpu()
        server_value = server_state[key].detach().cpu()
        if torch.is_floating_point(client_value):
            delta[key] = client_value - server_value
        else:
            delta[key] = client_value.clone()
    return delta


def quantize_update_for_communication(delta, keys, bits, residual=None):
    payload = OrderedDict()
    next_residual = OrderedDict()
    total_mb = 0.0

    for key in keys:
        value = delta[key].detach().cpu()

        if should_quantize_tensor(value, key):
            corrected = value
            if residual is not None and key in residual:
                corrected = corrected + residual[key]

            quantized = fake_quantize_tensor(corrected, bits)
            payload[key] = quantized
            if residual is not None:
                next_residual[key] = corrected - quantized
            total_mb += communication_tensor_size_mb(value, bits, key)
        else:
            payload[key] = value.clone()
            if residual is not None and torch.is_floating_point(value):
                next_residual[key] = torch.zeros_like(value)
            total_mb += communication_tensor_size_mb(value)

    return payload, total_mb, next_residual


def fp32_update_for_communication(delta, keys):
    payload = OrderedDict((key, delta[key].detach().cpu().clone()) for key in keys)
    total_mb = sum(communication_tensor_size_mb(value) for value in payload.values())
    return payload, total_mb


def estimate_quantized_model_size_mb(model, keys, bits):
    total_bytes = 0.0
    state = model.state_dict()

    for key in keys:
        value = state[key]
        if should_quantize_tensor(value, key):
            total_bytes += value.numel() * bits / 8 + quantization_scale_count(value) * 4
        else:
            total_bytes += value.numel() * value.element_size()

    return total_bytes / 1e6


def communicated_keys(model):
    from model import AdapterModel

    if isinstance(model, AdapterModel):
        return [
            key
            for key in model.state_dict().keys()
            if (
                key.startswith("adapter.")
                or key.startswith("classifier.")
            )
        ]

    return list(model.state_dict().keys())


def load_communicated_state(model, payload):
    state = model.state_dict()
    for key, value in payload.items():
        state[key] = value
    model.load_state_dict(state, strict=True)


def average_payloads(client_payloads, client_sizes):
    total_size = sum(client_sizes)
    averaged = OrderedDict()

    for key in client_payloads[0].keys():
        first_value = client_payloads[0][key]

        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value, dtype=torch.float32, device="cpu")
            for payload, size in zip(client_payloads, client_sizes):
                value += payload[key].float().cpu() * (size / total_size)
            averaged[key] = value.to(dtype=first_value.dtype)
        else:
            averaged[key] = first_value.clone()

    return averaged


def apply_averaged_update(model, averaged_update):
    state = model.state_dict()
    for key, value in averaged_update.items():
        if torch.is_floating_point(state[key]):
            state[key] = state[key].cpu() + value.cpu()
        else:
            state[key] = value
    model.load_state_dict(state, strict=True)


def fake_quantize_model_for_final_eval(model, keys, bits):
    quantized_model = copy.deepcopy(model).cpu()
    state = quantized_model.state_dict()

    for key in keys:
        value = state[key].cpu()
        if should_quantize_tensor(value, key):
            state[key] = fake_quantize_tensor(value, bits)

    quantized_model.load_state_dict(state, strict=True)
    return quantized_model
