import torch
import torch.nn as nn


def fake_quantize_tensor(tensor: torch.Tensor, bits: int, per_channel: bool = True) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor.detach().clone()

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    if per_channel and tensor.dim() >= 2:
        reduce_dims = tuple(range(1, tensor.dim()))
        max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
        q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return (q * scale).to(dtype=tensor.dtype)

    max_abs = tensor.detach().abs().max()
    if max_abs == 0:
        return tensor.detach().clone()

    scale = max_abs / qmax
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return (q * scale).to(dtype=tensor.dtype)


def quantization_scale_count(tensor: torch.Tensor) -> int:
    return tensor.shape[0] if tensor.dim() >= 2 else 1


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        return fake_quantize_tensor(tensor, bits)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad_output, None


class WeightFakeQuantParametrization(nn.Module):
    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = bits

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return FakeQuantSTE.apply(weight, self.bits)
