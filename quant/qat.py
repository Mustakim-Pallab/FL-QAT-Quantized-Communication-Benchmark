import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from quant.core import WeightFakeQuantParametrization


def enable_weight_qat(model: nn.Module, bits: int) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad:
            if not parametrize.is_parametrized(module, "weight"):
                parametrize.register_parametrization(
                    module, "weight", WeightFakeQuantParametrization(bits),
                )


def remove_weight_qat(model: nn.Module, leave_quantized: bool = True) -> None:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=leave_quantized,
            )
