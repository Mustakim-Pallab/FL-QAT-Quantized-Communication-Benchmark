import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import (
    ADAPTER_BACKBONE_LR,
    ADAPTER_LR,
    BATCH_SIZE,
    BRAIN_TUMOR_ADAPTER_BACKBONE_LR,
    BRAIN_TUMOR_ADAPTER_LR,
    BRAIN_TUMOR_GLOBAL_ROUNDS,
    BRAIN_TUMOR_LOCAL_EPOCHS,
    BRAIN_TUMOR_QUANTIZED_WARMUP_ROUNDS,
    BRAIN_TUMOR_VANILLA_LR,
    FEDPROX_MU,
    GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    QUANTIZED_WARMUP_ROUNDS,
    QUANTIZE_DOWNLINK,
    SEED,
    USE_CLASS_BALANCED_LOSS,
    USE_ERROR_FEEDBACK,
    VANILLA_LR,
    WEIGHT_DECAY,
    is_brain_tumor_dataset,
)
from config import set_global_seed
from federated.client import train_one_client
from model import AdapterModel
from utils.quant_utils import (
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


def build_criterion(train_dataset, device):
    if not USE_CLASS_BALANCED_LOSS:
        return nn.CrossEntropyLoss()

    labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(labels, minlength=len(train_dataset.classes))
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * len(class_counts)
    print(f"Class-balanced loss weights: {weights.tolist()}")
    return nn.CrossEntropyLoss(weight=weights.to(device))


def build_optimizer(model):
    vanilla_lr = BRAIN_TUMOR_VANILLA_LR if is_brain_tumor_dataset() else VANILLA_LR
    adapter_lr = BRAIN_TUMOR_ADAPTER_LR if is_brain_tumor_dataset() else ADAPTER_LR
    adapter_backbone_lr = (
        BRAIN_TUMOR_ADAPTER_BACKBONE_LR
        if is_brain_tumor_dataset()
        else ADAPTER_BACKBONE_LR
    )

    if isinstance(model, AdapterModel):
        head_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("feature_extractor."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": adapter_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": adapter_backbone_lr})

        return optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return optim.AdamW(trainable_params, lr=vanilla_lr, weight_decay=WEIGHT_DECAY)


def training_schedule(dataset_name, model_name):
    if is_brain_tumor_dataset(dataset_name=dataset_name):
        return {
            "global_rounds": BRAIN_TUMOR_GLOBAL_ROUNDS,
            "local_epochs": BRAIN_TUMOR_LOCAL_EPOCHS,
            "quantized_warmup_rounds": BRAIN_TUMOR_QUANTIZED_WARMUP_ROUNDS,
        }

    return {
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "quantized_warmup_rounds": QUANTIZED_WARMUP_ROUNDS,
    }


def federated_train(
    model_factory,
    model_name,
    train_dataset,
    partitions,
    device,
    label,
    comm_bits=None,
    qat_bits=None,
    initial_state=None,
    experiment_seed=SEED,
    dataset_name=None,
):
    print("\n" + "=" * 80)
    print(f"STARTING {label}: {model_name}")
    print("=" * 80)

    set_global_seed(experiment_seed)
    global_model = model_factory(model_name).to(device)
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    keys = communicated_keys(global_model)
    criterion = build_criterion(train_dataset, device)
    communication_sizes = []
    schedule = training_schedule(dataset_name, model_name)
    global_rounds = schedule["global_rounds"]
    local_epochs = schedule["local_epochs"]
    quantized_warmup_rounds = schedule["quantized_warmup_rounds"]
    print(
        "Schedule: "
        f"rounds={global_rounds}, local_epochs={local_epochs}, "
        f"quantized_warmup_rounds={quantized_warmup_rounds}"
    )
    error_buffers = {client_id: OrderedDict() for client_id in partitions}

    for round_idx in range(global_rounds):
        print(f"\nGlobal round {round_idx + 1}/{global_rounds}")
        use_quantized_round = comm_bits is not None and round_idx >= quantized_warmup_rounds

        global_model.cpu()
        global_state = global_model.state_dict()

        proximal_state = {
            key: value.detach().clone()
            for key, value in global_state.items()
            if torch.is_floating_point(value)
        }

        if use_quantized_round and QUANTIZE_DOWNLINK:
            server_payload, server_payload_mb = quantize_for_communication(
                global_state, keys, comm_bits,
            )
        else:
            server_payload, server_payload_mb = fp32_for_communication(global_state, keys)

        client_payloads = []
        client_sizes = []

        for client_id, data_indices in partitions.items():
            print(f"  Client {client_id}: training on {len(data_indices)} samples")

            set_global_seed(experiment_seed + round_idx * 1000 + client_id)
            client_model = model_factory(model_name).to(device)

            load_communicated_state(client_model, server_payload)

            if qat_bits is not None and round_idx >= quantized_warmup_rounds:
                enable_weight_qat(client_model, qat_bits)

            loader_generator = torch.Generator()
            loader_generator.manual_seed(experiment_seed + round_idx * 1000 + client_id)
            client_loader = DataLoader(
                Subset(train_dataset, data_indices),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=loader_generator,
            )
            optimizer = build_optimizer(client_model)
            train_one_client(
                client_model,
                client_loader,
                criterion,
                optimizer,
                device,
                local_epochs,
                proximal_state=proximal_state,
            )

            client_model.cpu()

            if qat_bits is not None and round_idx >= quantized_warmup_rounds:
                remove_weight_qat(client_model, leave_quantized=True)

            client_state = client_model.state_dict()
            delta = model_delta(client_state, global_state, keys)

            if not use_quantized_round:
                payload, payload_mb = fp32_update_for_communication(delta, keys)
                _ = OrderedDict()
            else:
                residual = error_buffers[client_id] if USE_ERROR_FEEDBACK else None
                payload, payload_mb, next_residual = quantize_update_for_communication(
                    delta, keys, comm_bits, residual=residual,
                )
                if USE_ERROR_FEEDBACK:
                    error_buffers[client_id] = next_residual

            print(f"    Upload payload:   {payload_mb:.4f} MB")
            print(f"    Download payload: {server_payload_mb:.4f} MB")
            client_payloads.append(payload)
            client_sizes.append(len(data_indices))
            communication_sizes.append(payload_mb + server_payload_mb)

        averaged_payload = average_payloads(client_payloads, client_sizes)
        apply_averaged_update(global_model, averaged_payload)
        global_model.to(device)

    avg_communication_mb = float(np.mean(communication_sizes)) if communication_sizes else 0.0
    return global_model, avg_communication_mb, keys
