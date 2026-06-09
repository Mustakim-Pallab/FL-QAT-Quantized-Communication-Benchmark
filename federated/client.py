import torch
import torch.nn as nn

from config import FEDPROX_MU, GRAD_CLIP_NORM


def train_one_client(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epochs,
    proximal_state=None,
):
    model.train()
    proximal_params = []
    if proximal_state is not None and FEDPROX_MU > 0:
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
                loss = loss + 0.5 * FEDPROX_MU * prox_loss

            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    GRAD_CLIP_NORM,
                )
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        acc = correct / max(total, 1)
        print(f"    Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2%}")
