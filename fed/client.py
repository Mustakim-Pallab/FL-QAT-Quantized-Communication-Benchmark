import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

from config import Config
from utils.logging_utils import get_logger

logger = get_logger()


def train_one_client(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        config: Config,
        proximal_state: OrderedDict[str, torch.Tensor] | None = None,
) -> None:
    model.train()
    proximal_params: list[tuple[nn.Parameter, torch.Tensor]] = []

    if proximal_state is not None and config.fedprox_mu > 0:
        for name, param in model.named_parameters():
            state_name = name.replace(".parametrizations.weight.original", ".weight")
            if param.requires_grad and state_name in proximal_state:
                proximal_params.append((param, proximal_state[state_name].to(device)))

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if proximal_params:
                prox_loss = torch.zeros((), device=device)
                for param, global_param in proximal_params:
                    prox_loss = prox_loss + torch.sum((param - global_param) ** 2)
                loss = loss + 0.5 * config.fedprox_mu * prox_loss

            loss.backward()

            if config.grad_clip_norm is not None and config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.grad_clip_norm,
                )

            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        acc = correct / max(total, 1)
        logger.info("  Epoch %d/%d: Loss=%.4f, Acc=%.2f%%", epoch + 1, epochs, avg_loss, acc * 100)
