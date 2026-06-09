import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable

from config import Config
from config import set_global_seed
from fed.client import train_one_client
from models import AdapterModel
from quant import (
    apply_averaged_update,
    average_payloads,
    communicated_keys,
    enable_weight_qat,
    fp32_for_communication,
    fp32_update_for_communication,
    load_communicated_state,
    model_delta,
    quantize_for_communication,
    quantize_update_for_communication,
    remove_weight_qat,
)
from utils.logging_utils import get_logger

logger = get_logger()


def build_criterion(
        train_dataset: Dataset,
        device: torch.device,
        config: Config,
) -> nn.Module:
    if not config.use_class_balanced_loss:
        return nn.CrossEntropyLoss()

    labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(labels, minlength=len(train_dataset.classes))
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * len(class_counts)
    logger.info("Class-balanced loss weights: %s", weights.tolist())
    return nn.CrossEntropyLoss(weight=weights.to(device))


def build_optimizer(
        model: nn.Module,
        config: Config,
        dataset_name: str | None = None,
) -> torch.optim.Optimizer:
    if isinstance(model, AdapterModel):
        head_params: list[nn.Parameter] = []
        backbone_params: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("feature_extractor."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": config.adapter_lr_for(dataset_name)})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": config.adapter_backbone_lr_for(dataset_name)})

        return optim.AdamW(param_groups, weight_decay=config.weight_decay)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable_params, lr=config.vanilla_lr_for(dataset_name), weight_decay=config.weight_decay)


def federated_train(
        model_factory: Callable[[str], nn.Module],
        model_name: str,
        train_dataset: Dataset,
        partitions: dict[int, list[int]],
        device: torch.device,
        label: str,
        config: Config,
        comm_bits: int | None = None,
        qat_bits: int | None = None,
        initial_state: OrderedDict[str, torch.Tensor] | None = None,
        experiment_seed: int = 42,
        dataset_name: str | None = None,
) -> tuple[nn.Module, float, list[str]]:
    logger.info("=" * 80)
    logger.info("STARTING %s: %s", label, model_name)
    logger.info("=" * 80)

    set_global_seed(experiment_seed)
    global_model = model_factory(model_name).to(device)
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state), strict=True)

    keys = communicated_keys(global_model)
    criterion = build_criterion(train_dataset, device, config)
    communication_sizes: list[float] = []
    schedule = config.training_schedule(dataset_name)
    global_rounds = schedule["global_rounds"]
    local_epochs = schedule["local_epochs"]
    quantized_warmup_rounds = schedule["quantized_warmup_rounds"]

    logger.info("Schedule: rounds=%d, local_epochs=%d, quantized_warmup_rounds=%d",
                global_rounds, local_epochs, quantized_warmup_rounds)

    error_buffers: dict[int, OrderedDict[str, torch.Tensor]] = {
        client_id: OrderedDict() for client_id in partitions
    }

    for round_idx in range(global_rounds):
        logger.info("Global round %d/%d", round_idx + 1, global_rounds)
        use_quantized_round = comm_bits is not None and round_idx >= quantized_warmup_rounds

        global_model.cpu()
        global_state = global_model.state_dict()

        proximal_state: OrderedDict[str, torch.Tensor] = OrderedDict(
            (key, value.detach().clone())
            for key, value in global_state.items()
            if torch.is_floating_point(value)
        )

        if use_quantized_round and config.quantize_downlink:
            server_payload, server_payload_mb = quantize_for_communication(
                global_state, keys, comm_bits, config,
            )
        else:
            server_payload, server_payload_mb = fp32_for_communication(global_state, keys)

        client_payloads: list[OrderedDict[str, torch.Tensor]] = []
        client_sizes: list[int] = []

        for client_id, data_indices in partitions.items():
            logger.info("  Client %d: training on %d samples", client_id, len(data_indices))

            set_global_seed(experiment_seed + round_idx * 1000 + client_id)
            client_model = model_factory(model_name).to(device)
            load_communicated_state(client_model, server_payload)

            if qat_bits is not None and round_idx >= quantized_warmup_rounds:
                enable_weight_qat(client_model, qat_bits)

            loader_generator = torch.Generator()
            loader_generator.manual_seed(experiment_seed + round_idx * 1000 + client_id)
            client_loader = DataLoader(
                Subset(train_dataset, data_indices),
                batch_size=config.batch_size,
                shuffle=True,
                generator=loader_generator,
            )
            optimizer = build_optimizer(client_model, config, dataset_name)

            train_one_client(
                client_model, client_loader, criterion, optimizer, device,
                local_epochs, config, proximal_state=proximal_state,
            )

            client_model.cpu()

            if qat_bits is not None and round_idx >= quantized_warmup_rounds:
                remove_weight_qat(client_model, leave_quantized=True)

            client_state = client_model.state_dict()
            delta = model_delta(client_state, global_state, keys)

            if not use_quantized_round:
                payload, payload_mb = fp32_update_for_communication(delta, keys, config)
            else:
                residual = error_buffers[client_id] if config.use_error_feedback else None
                payload, payload_mb, next_residual = quantize_update_for_communication(
                    delta, keys, comm_bits, config, residual=residual,
                )
                if config.use_error_feedback:
                    error_buffers[client_id] = next_residual

            logger.info("    Upload payload:   %.4f MB", payload_mb)
            logger.info("    Download payload: %.4f MB", server_payload_mb)
            client_payloads.append(payload)
            client_sizes.append(len(data_indices))
            communication_sizes.append(payload_mb + server_payload_mb)

        averaged_payload = average_payloads(client_payloads, client_sizes)
        apply_averaged_update(global_model, averaged_payload)
        global_model.to(device)

    avg_communication_mb = float(np.mean(communication_sizes)) if communication_sizes else 0.0
    return global_model, avg_communication_mb, keys
