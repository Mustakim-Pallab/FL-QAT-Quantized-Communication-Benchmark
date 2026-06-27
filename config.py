from __future__ import annotations

import numpy as np
import random
import torch
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

DATASET_ROOT = Path(__file__).resolve().parent / "datasets"

DATASET_DEFAULTS: tuple[dict[str, str], ...] = (
    {
        "name": "Lung Ultrasound",
        "train_dir": str(DATASET_ROOT / "lungs_ultrasound" / "train"),
        "test_dir": str(DATASET_ROOT / "lungs_ultrasound" / "test"),
    },
    {
        "name": "Brain Tumor",
        "train_dir": str(DATASET_ROOT / "brain_tumor" / "train"),
        "test_dir": str(DATASET_ROOT / "brain_tumor" / "test"),
    },
    {
        "name": "Fundus Diabetic Retinopathy",
        "train_dir": str(DATASET_ROOT / "fundus_diabetic_retinopathy" / "train"),
        "test_dir": str(DATASET_ROOT / "fundus_diabetic_retinopathy" / "test"),
    },
)

MODEL_NAME_DEFAULTS: tuple[str, ...] = (
    "mobilenet_v2",
    "resnet18",
    # "densenet121",
    # "alexnet",
    "vit_b_16",
)

ADAPTER_TAIL_BLOCKS_DEFAULTS: dict[str, int] = {
    "mobilenet_v2": 2,
    "resnet18": 1,
    "densenet121": 1,
    "alexnet": 1,
    "vit_b_16": 2,
}


def is_brain_tumor_dataset(dataset_name: str | None) -> bool:
    return dataset_name == "Brain Tumor"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Config:
    datasets: tuple[dict[str, str], ...] = DATASET_DEFAULTS
    model_names: tuple[str, ...] = MODEL_NAME_DEFAULTS

    num_clients: int = 2
    dirichlet_beta: float = 0.8
    global_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 32

    vanilla_lr: float = 1e-4
    adapter_lr: float = 1e-3
    adapter_backbone_lr: float = 5e-5
    weight_decay: float = 1e-4

    adapter_hidden_dim: int = 512
    adapter_dropout: float = 0.1
    adapter_type: str = "attention"
    adapter_attention_heads: int = 8
    adapter_attention_tokens: int = 8
    adapter_train_backbone_tail: bool = False
    adapter_tail_blocks: dict[str, int] | None = None

    use_class_balanced_loss: bool = True
    use_pretrained: bool = True
    quantize_downlink: bool = True
    quantized_warmup_rounds: int = 1
    use_error_feedback: bool = True
    keep_small_tensors_fp32: bool = True
    fedprox_mu: float = 0.01
    vanilla_fedprox_only: bool = False
    grad_clip_norm: float = 1.0
    seed: int = 42

    brain_tumor_global_rounds: int = 5
    brain_tumor_local_epochs: int = 3
    brain_tumor_quantized_warmup_rounds: int = 2
    brain_tumor_vanilla_lr: float = 5e-5
    brain_tumor_adapter_lr: float = 5e-4
    brain_tumor_adapter_backbone_lr: float = 1e-5
    brain_tumor_tune_adapter_tail: bool = True

    def __post_init__(self) -> None:
        if self.adapter_tail_blocks is None:
            object.__setattr__(self, "adapter_tail_blocks", dict(ADAPTER_TAIL_BLOCKS_DEFAULTS))

    def with_overrides(self, **kwargs) -> Config:
        return replace(self, **kwargs)

    def training_schedule(self, dataset_name: str | None) -> dict[str, int]:
        if is_brain_tumor_dataset(dataset_name):
            return {
                "global_rounds": self.brain_tumor_global_rounds,
                "local_epochs": self.brain_tumor_local_epochs,
                "quantized_warmup_rounds": self.brain_tumor_quantized_warmup_rounds,
            }
        return {
            "global_rounds": self.global_rounds,
            "local_epochs": self.local_epochs,
            "quantized_warmup_rounds": self.quantized_warmup_rounds,
        }

    def vanilla_lr_for(self, dataset_name: str | None) -> float:
        return self.brain_tumor_vanilla_lr if is_brain_tumor_dataset(dataset_name) else self.vanilla_lr

    def adapter_lr_for(self, dataset_name: str | None) -> float:
        return self.brain_tumor_adapter_lr if is_brain_tumor_dataset(dataset_name) else self.adapter_lr

    def adapter_backbone_lr_for(self, dataset_name: str | None) -> float:
        return self.brain_tumor_adapter_backbone_lr if is_brain_tumor_dataset(
            dataset_name) else self.adapter_backbone_lr

    def should_tune_adapter_tail(self, dataset_name: str | None) -> bool:
        return self.adapter_train_backbone_tail or (
                self.brain_tumor_tune_adapter_tail and is_brain_tumor_dataset(dataset_name)
        )

    def should_quantize_tensor(self, tensor: torch.Tensor, key: str | None = None) -> bool:
        if not tensor.is_floating_point():
            return False
        if self.keep_small_tensors_fp32 and tensor.dim() <= 1:
            return False
        if key is not None:
            key_parts = key.split(".")
            parameter_name = key_parts[-1]
            if parameter_name not in {"weight", "in_proj_weight"}:
                return False
            sensitive_names = {"class_token", "memory_tokens", "pos_embedding"}
            if any(part in sensitive_names for part in key_parts):
                return False
        return True
