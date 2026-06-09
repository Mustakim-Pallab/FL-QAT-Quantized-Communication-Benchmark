import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Sequence

from config import Config
from quant.core import fake_quantize_tensor


def average_payloads(
        client_payloads: list[OrderedDict[str, torch.Tensor]],
        client_sizes: list[int],
) -> OrderedDict[str, torch.Tensor]:
    total_size = sum(client_sizes)
    averaged = OrderedDict[str, torch.Tensor]()

    for key in client_payloads[0].keys():
        first_value = client_payloads[0][key]

        if first_value.is_floating_point():
            value = torch.zeros_like(first_value, dtype=torch.float32, device="cpu")
            for payload, size in zip(client_payloads, client_sizes):
                value += payload[key].float().cpu() * (size / total_size)
            averaged[key] = value.to(dtype=first_value.dtype)
        else:
            averaged[key] = first_value.clone()

    return averaged


def apply_averaged_update(model: nn.Module, averaged_update: OrderedDict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for key, value in averaged_update.items():
        if state[key].is_floating_point():
            state[key] = state[key].cpu() + value.cpu()
        else:
            state[key] = value
    model.load_state_dict(state, strict=True)


def fake_quantize_model_for_final_eval(
        model: nn.Module,
        keys: Sequence[str],
        bits: int,
        config: Config,
) -> nn.Module:
    quantized_model = copy.deepcopy(model).cpu()
    state = quantized_model.state_dict()

    for key in keys:
        value = state[key].cpu()
        if config.should_quantize_tensor(value, key):
            state[key] = fake_quantize_tensor(value, bits)

    quantized_model.load_state_dict(state, strict=True)
    return quantized_model
