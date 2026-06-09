import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.utils.data import DataLoader


class FLClient(ABC):
    @abstractmethod
    def train(
            self,
            model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            epochs: int,
            proximal_state: OrderedDict[str, torch.Tensor] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def enable_qat(self, model: nn.Module, bits: int) -> None:
        pass

    @abstractmethod
    def disable_qat(self, model: nn.Module, leave_quantized: bool = True) -> None:
        pass


class FLServer(ABC):
    @abstractmethod
    def build_criterion(self, dataset: object, device: torch.device) -> nn.Module:
        pass

    @abstractmethod
    def build_optimizer(self, model: nn.Module, dataset_name: str | None = None) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def aggregate(
            self,
            client_payloads: list[OrderedDict[str, torch.Tensor]],
            client_sizes: list[int],
    ) -> OrderedDict[str, torch.Tensor]:
        pass

    @abstractmethod
    def apply_update(
            self,
            model: nn.Module,
            averaged_update: OrderedDict[str, torch.Tensor],
    ) -> None:
        pass
