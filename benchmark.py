import copy
import torch
from functools import partial
from typing import Any

from config import Config, set_global_seed
from data_loaders import build_dataloaders, partition_data
from fed.server import federated_train
from models import build_adapter_model, build_vanilla_model
from quant import (
    estimate_quantized_model_size_mb,
    fake_quantize_model_for_final_eval,
)
from utils.device import get_device_info, resolve_device
from utils.display import display_name, print_results_table
from utils.evaluation import evaluate_model, get_full_state_size_mb
from utils.logging_utils import get_logger

logger = get_logger()

EXPERIMENTS: list[dict[str, Any]] = [
    {"label": "VANILLA FP32 FL", "is_adapter": False, "comm_bits": None, "qat_bits": None},
    {"label": "ADAPTER FP32 FL", "is_adapter": True, "comm_bits": None, "qat_bits": None},
    {"label": "VANILLA QAT INT8", "is_adapter": False, "comm_bits": 8, "qat_bits": 8},
    {"label": "ADAPTER QAT INT8", "is_adapter": True, "comm_bits": 8, "qat_bits": 8},
    {"label": "VANILLA QAT INT4", "is_adapter": False, "comm_bits": 4, "qat_bits": 4},
    {"label": "ADAPTER QAT INT4", "is_adapter": True, "comm_bits": 4, "qat_bits": 4},
]


def _build_vanilla(name: str, num_classes: int, use_pretrained: bool) -> torch.nn.Module:
    return build_vanilla_model(name, num_classes, use_pretrained)


def _build_adapter(name: str, num_classes: int, config: Config, dataset_name: str | None) -> torch.nn.Module:
    return build_adapter_model(name, num_classes, config, dataset_name=dataset_name)


def _eval_acc(
        trained: dict[str, tuple[torch.nn.Module, float, list[str]]],
        label: str,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: list[str],
) -> float:
    return evaluate_model(trained[label][0], test_loader, device, class_names)


def _eval_acc_quantized(
        trained: dict[str, tuple[torch.nn.Module, float, list[str]]],
        label: str,
        bits: int,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: list[str],
        config: Config,
) -> float:
    m, _, keys = trained[label]
    eq = fake_quantize_model_for_final_eval(m, keys, bits, config)
    return evaluate_model(eq, test_loader, device, class_names)


def make_factories(num_classes: int, config: Config, dataset_name: str | None):
    return (
        partial(_build_vanilla, num_classes=num_classes, use_pretrained=config.use_pretrained),
        partial(_build_adapter, num_classes=num_classes, config=config, dataset_name=dataset_name),
    )


def build_initial_states(vanilla_factory, adapter_factory, model_name: str, experiment_seed: int):
    set_global_seed(experiment_seed)
    vanilla_init = copy.deepcopy(vanilla_factory(model_name).cpu().state_dict())

    set_global_seed(experiment_seed)
    adapter_init = copy.deepcopy(adapter_factory(model_name).cpu().state_dict())

    return vanilla_init, adapter_init


def train_experiments(
        factories: dict[str, Any],
        initial_states: dict[str, Any],
        model_name: str,
        train_dataset: torch.utils.data.Dataset,
        partitions: dict[int, list[int]],
        device: torch.device,
        config: Config,
        dataset_name: str | None,
        experiment_seed: int,
) -> dict[str, tuple[torch.nn.Module, float, list[str]]]:
    trained: dict[str, tuple[torch.nn.Module, float, list[str]]] = {}

    for exp in EXPERIMENTS:
        factory = factories["adapter"] if exp["is_adapter"] else factories["vanilla"]
        init_state = initial_states["adapter"] if exp["is_adapter"] else initial_states["vanilla"]

        model, comm_mb, keys = federated_train(
            factory, model_name, train_dataset, partitions, device,
            exp["label"], config,
            comm_bits=exp["comm_bits"], qat_bits=exp["qat_bits"],
            initial_state=init_state,
            experiment_seed=experiment_seed,
            dataset_name=dataset_name,
        )
        trained[exp["label"]] = (model, comm_mb, keys)

    return trained


def build_row(
        trained: dict[str, tuple[torch.nn.Module, float, list[str]]],
        model_name: str,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: list[str],
        config: Config,
) -> dict[str, Any]:
    return {
        "model": display_name(model_name),
        "vanilla_fp32_acc": _eval_acc(trained, "VANILLA FP32 FL", test_loader, device, class_names),
        "vanilla_fp32_upload": trained["VANILLA FP32 FL"][1],
        "adapter_fp32_acc": _eval_acc(trained, "ADAPTER FP32 FL", test_loader, device, class_names),
        "adapter_fp32_upload": trained["ADAPTER FP32 FL"][1],
        "vanilla_int8_acc": _eval_acc_quantized(trained, "VANILLA QAT INT8", 8, test_loader, device, class_names,
                                                config),
        "vanilla_int8_upload": trained["VANILLA QAT INT8"][1],
        "adapter_int8_acc": _eval_acc_quantized(trained, "ADAPTER QAT INT8", 8, test_loader, device, class_names,
                                                config),
        "adapter_int8_upload": trained["ADAPTER QAT INT8"][1],
        "vanilla_int4_acc": _eval_acc_quantized(trained, "VANILLA QAT INT4", 4, test_loader, device, class_names,
                                                config),
        "vanilla_int4_upload": trained["VANILLA QAT INT4"][1],
        "adapter_int4_acc": _eval_acc_quantized(trained, "ADAPTER QAT INT4", 4, test_loader, device, class_names,
                                                config),
        "adapter_int4_upload": trained["ADAPTER QAT INT4"][1],
    }


def log_model_sizes(trained: dict, config: Config) -> None:
    logger.info("Final full state sizes for reference:")
    for tag, bits in [("VANILLA FP32 FL", None), ("ADAPTER FP32 FL", None),
                      ("VANILLA QAT INT8", 8), ("ADAPTER QAT INT8", 8),
                      ("VANILLA QAT INT4", 4), ("ADAPTER QAT INT4", 4)]:
        model, _, keys = trained[tag]
        if bits is None:
            logger.info("  %-42s %.2f MB", tag, get_full_state_size_mb(model))
        else:
            logger.info("  %-42s %.2f MB", tag, estimate_quantized_model_size_mb(model, keys, bits, config))


def _run_dataset_experiments(config: Config, dataset_config: dict, device: torch.device, dataset_idx: int) -> list[
    dict[str, Any]]:
    dataset_name = dataset_config["name"]
    logger.info("#" * 80)
    logger.info("DATASET: %s", dataset_name)
    logger.info("Train dir: %s", dataset_config["train_dir"])
    logger.info("Test dir:  %s", dataset_config["test_dir"])
    logger.info("#" * 80)

    dataset_seed = config.seed + dataset_idx * 100000
    set_global_seed(dataset_seed)

    train_dataset, _, _, test_loader = build_dataloaders(
        dataset_config["train_dir"], dataset_config["test_dir"],
        dataset_name, config,
    )
    class_names = train_dataset.classes
    num_classes = len(class_names)
    logger.info("Classes: %s", class_names)

    partitions = partition_data(
        train_dataset, n_clients=config.num_clients,
        beta=config.dirichlet_beta, logger=logger,
    )
    results: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(config.model_names):
        experiment_seed = dataset_seed + model_idx * 10000

        vanilla_factory, adapter_factory = make_factories(num_classes, config, dataset_name)
        vanilla_init, adapter_init = build_initial_states(
            vanilla_factory, adapter_factory, model_name, experiment_seed,
        )

        factories = {"vanilla": vanilla_factory, "adapter": adapter_factory}
        init_states = {"vanilla": vanilla_init, "adapter": adapter_init}

        trained = train_experiments(
            factories, init_states, model_name, train_dataset,
            partitions, device, config, dataset_name, experiment_seed,
        )

        row = build_row(trained, model_name, test_loader, device, class_names, config)
        results.append(row)

        log_model_sizes(trained, config)
        print_results_table(results, dataset_name)

    print_results_table(results, dataset_name)
    return results


def run_benchmark(config: Config) -> None:
    set_global_seed(config.seed)

    device = resolve_device("cuda")
    logger.info("Using device: %s", get_device_info(device))
    logger.info("Method: FP32 server master weights, FedProx local training, "
                "quantized update communication with error feedback")
    logger.info("Quantized downlink enabled: %s", config.quantize_downlink)

    for dataset_idx, dataset_config in enumerate(config.datasets):
        _run_dataset_experiments(config, dataset_config, device, dataset_idx)
