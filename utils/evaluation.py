import os
import tempfile

import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: list[str],
        verbose: bool = False,
) -> float:
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            correct += preds.eq(y).sum().item()
            total += y.size(0)

    if verbose:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        print(confusion_matrix(y_true, y_pred))

    return correct / max(total, 1)


def get_full_state_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1e6
    finally:
        if os.path.exists(path):
            os.remove(path)
