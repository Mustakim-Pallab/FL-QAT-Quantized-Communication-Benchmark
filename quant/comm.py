import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Sequence

from config import Config
from quant.core import fake_quantize_tensor, quantization_scale_count


def communication_tensor_size_mb(
        tensor: torch.Tensor,
        config: Config,
        bits: int | None = None,
        key: str | None = None,
) -> float:
    if bits is not None and config.should_quantize_tensor(tensor, key):
        return (tensor.numel() * bits / 8 + quantization_scale_count(tensor) * 4) / 1e6
    return tensor.numel() * tensor.element_size() / 1e6


def quantize_for_communication(
        state_dict: OrderedDict[str, torch.Tensor],
        keys: Sequence[str],
        bits: int,
        config: Config,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    payload = OrderedDict[str, torch.Tensor]()
    total_bytes = 0.0

    for key in keys:
        value = state_dict[key].detach().cpu()
        if config.should_quantize_tensor(value, key):
            payload[key] = fake_quantize_tensor(value, bits)
            total_bytes += value.numel() * bits / 8 + quantization_scale_count(value) * 4
        else:
            payload[key] = value.clone()
            total_bytes += value.numel() * value.element_size()

    return payload, total_bytes / 1e6


def fp32_for_communication(
        state_dict: OrderedDict[str, torch.Tensor],
        keys: Sequence[str],
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    payload = OrderedDict((key, state_dict[key].detach().cpu().clone()) for key in keys)
    total_bytes = sum(v.numel() * v.element_size() for v in payload.values())
    return payload, total_bytes / 1e6


def model_delta(
        client_state: OrderedDict[str, torch.Tensor],
        server_state: OrderedDict[str, torch.Tensor],
        keys: Sequence[str],
) -> OrderedDict[str, torch.Tensor]:
    delta = OrderedDict[str, torch.Tensor]()
    for key in keys:
        client_value = client_state[key].detach().cpu()
        server_value = server_state[key].detach().cpu()
        if client_value.is_floating_point():
            delta[key] = client_value - server_value
        else:
            delta[key] = client_value.clone()
    return delta


def quantize_update_for_communication(
        delta: OrderedDict[str, torch.Tensor],
        keys: Sequence[str],
        bits: int,
        config: Config,
        residual: OrderedDict[str, torch.Tensor] | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], float, OrderedDict[str, torch.Tensor]]:
    payload = OrderedDict[str, torch.Tensor]()
    next_residual = OrderedDict[str, torch.Tensor]()
    total_mb = 0.0

    for key in keys:
        value = delta[key].detach().cpu()

        if config.should_quantize_tensor(value, key):
            corrected = value
            if residual is not None and key in residual:
                corrected = corrected + residual[key]

            quantized = fake_quantize_tensor(corrected, bits)
            payload[key] = quantized
            if residual is not None:
                next_residual[key] = corrected - quantized
            total_mb += communication_tensor_size_mb(value, config, bits, key)
        else:
            payload[key] = value.clone()
            if residual is not None and value.is_floating_point():
                next_residual[key] = torch.zeros_like(value)
            total_mb += communication_tensor_size_mb(value, config)

    return payload, total_mb, next_residual


def fp32_update_for_communication(
        delta: OrderedDict[str, torch.Tensor],
        keys: Sequence[str],
        config: Config,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    payload = OrderedDict((key, delta[key].detach().cpu().clone()) for key in keys)
    total_mb = sum(communication_tensor_size_mb(v, config) for v in payload.values())
    return payload, total_mb


def estimate_quantized_model_size_mb(
        model: nn.Module,
        keys: Sequence[str],
        bits: int,
        config: Config,
) -> float:
    total_bytes = 0.0
    state = model.state_dict()

    for key in keys:
        value = state[key]
        if config.should_quantize_tensor(value, key):
            total_bytes += value.numel() * bits / 8 + quantization_scale_count(value) * 4
        else:
            total_bytes += value.numel() * value.element_size()

    return total_bytes / 1e6


def communicated_keys(model: nn.Module) -> list[str]:
    from models.adapter_model import AdapterModel

    if isinstance(model, AdapterModel):
        return [k for k in model.state_dict().keys() if k.startswith("adapter.") or k.startswith("classifier.")]
    return list(model.state_dict().keys())


def load_communicated_state(model: nn.Module, payload: OrderedDict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    state.update(payload)
    model.load_state_dict(state, strict=True)
