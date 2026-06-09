import copy
import os
import tempfile

import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CURRENT_DATASET_NAME,
    CURRENT_MODEL_NAME,
    DATASETS,
    DIRICHLET_BETA,
    KEEP_SMALL_TENSORS_FP32,
    MODEL_NAMES,
    NUM_CLIENTS,
    QUANTIZE_DOWNLINK,
    SEED,
)
from config import set_global_seed
from data_loaders import build_dataloaders, partition_data
from federated.server import federated_train
from model import build_adapter_model, build_vanilla_model
from utils.quant_utils import (
    estimate_quantized_model_size_mb,
    fake_quantize_model_for_final_eval,
)


def maybe_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass


def evaluate_model(model, loader, device, class_names, verbose=False):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += preds.eq(y).sum().item()
            total += y.size(0)

    if verbose:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        print(confusion_matrix(y_true, y_pred))

    return correct / max(total, 1)


def get_full_state_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1e6
    finally:
        if os.path.exists(path):
            os.remove(path)


def display_name(model_name):
    names = {
        "mobilenet_v2": "MobileNetV2",
        "resnet18": "ResNet18",
        "densenet121": "DenseNet121",
        "alexnet": "AlexNet",
        "vit_b_16": "ViT-B/16",
    }
    return names[model_name]


def metric_cell(accuracy, payload_mb):
    return f"{accuracy * 100:.2f}% ({payload_mb:.2f} MB)"


def print_results_table(results, dataset_name):
    columns = [
        "Model",
        "Vanilla FL (FP32)",
        "Adapter (FP32)",
        "Vanilla (INT8)",
        "Adapter (INT8)",
        "Vanilla (INT4)",
        "Adapter (INT4)",
    ]

    print("\n" + "=" * 155)
    print(f"FINAL BENCHMARK SUMMARY: {dataset_name}")
    print("Accuracy is final model accuracy. MB is average client communication per round.")
    print("Communication MB includes download + upload for the communicated parameters.")
    print("Adapter FL transmits only adapter+classifier params; backbone is never sent.")
    print("=" * 155)
    print(
        f"{columns[0]:<14} {columns[1]:<24} {columns[2]:<24} "
        f"{columns[3]:<24} {columns[4]:<24} {columns[5]:<24} {columns[6]:<24}"
    )
    print("-" * 155)

    for row in results:
        print(
            f"{row['model']:<14} "
            f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload']):<24} "
            f"{metric_cell(row['adapter_fp32_acc'], row['adapter_fp32_upload']):<24} "
            f"{metric_cell(row['vanilla_int8_acc'], row['vanilla_int8_upload']):<24} "
            f"{metric_cell(row['adapter_int8_acc'], row['adapter_int8_upload']):<24} "
            f"{metric_cell(row['vanilla_int4_acc'], row['vanilla_int4_upload']):<24} "
            f"{metric_cell(row['adapter_int4_acc'], row['adapter_int4_upload']):<24}"
        )

    print("\nLaTeX table rows:")
    print(
        r"\textbf{Model} & \textbf{Vanilla FL (FP32)} & \textbf{Adapter (FP32)} "
        r"& \textbf{Vanilla (INT8)} & \textbf{Adapter (INT8)} "
        r"& \textbf{Vanilla (INT4)} & \textbf{Adapter (INT4)} \\"
    )
    print(r"\hline")
    for row in results:
        print(
            f"{row['model']} & "
            f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload'])} & "
            f"{metric_cell(row['adapter_fp32_acc'], row['adapter_fp32_upload'])} & "
            f"{metric_cell(row['vanilla_int8_acc'], row['vanilla_int8_upload'])} & "
            f"{metric_cell(row['adapter_int8_acc'], row['adapter_int8_upload'])} & "
            f"{metric_cell(row['vanilla_int4_acc'], row['vanilla_int4_upload'])} & "
            f"{metric_cell(row['adapter_int4_acc'], row['adapter_int4_upload'])} \\\\"
        )


def run_benchmark():
    set_global_seed(SEED)
    maybe_mount_drive()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(
        "Method: FP32 server master weights, FedProx local training, "
        "quantized update communication with error feedback"
    )
    print(f"Quantized downlink enabled: {QUANTIZE_DOWNLINK}")
    print(
        "Adapter FL: backbone frozen locally on all clients, "
        "only adapter+classifier communicated per round"
    )

    for dataset_idx, dataset_config in enumerate(DATASETS):
        dataset_name = dataset_config["name"]
        print("\n" + "#" * 80)
        print(f"DATASET: {dataset_name}")
        print(f"Train dir: {dataset_config['train_dir']}")
        print(f"Test dir:  {dataset_config['test_dir']}")
        print("#" * 80)

        dataset_seed = SEED + dataset_idx * 100000
        set_global_seed(dataset_seed)
        train_dataset, _, _, test_loader = build_dataloaders(
            dataset_config["train_dir"],
            dataset_config["test_dir"],
            dataset_name,
        )
        class_names = train_dataset.classes
        num_classes = len(class_names)
        print(f"Classes: {class_names}")

        partitions = partition_data(
            train_dataset,
            n_clients=NUM_CLIENTS,
            beta=DIRICHLET_BETA,
        )
        results = []

        for model_idx, model_name in enumerate(MODEL_NAMES):
            global CURRENT_DATASET_NAME, CURRENT_MODEL_NAME
            CURRENT_DATASET_NAME = dataset_name
            CURRENT_MODEL_NAME = model_name

            def vanilla_factory(name, nc=num_classes):
                return build_vanilla_model(name, nc)

            def adapter_factory(name, nc=num_classes):
                return build_adapter_model(name, nc)

            experiment_seed = dataset_seed + model_idx * 10000

            set_global_seed(experiment_seed)
            vanilla_initial_state = copy.deepcopy(
                vanilla_factory(model_name).cpu().state_dict()
            )
            set_global_seed(experiment_seed)
            adapter_initial_state = copy.deepcopy(
                adapter_factory(model_name).cpu().state_dict()
            )

            vanilla_fp32, vanilla_fp32_upload, vanilla_keys = federated_train(
                vanilla_factory, model_name, train_dataset, partitions, device,
                "VANILLA FP32 FL",
                initial_state=vanilla_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )
            adapter_fp32, adapter_fp32_upload, adapter_keys = federated_train(
                adapter_factory, model_name, train_dataset, partitions, device,
                "ADAPTER FP32 FL",
                initial_state=adapter_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )

            vanilla_int8, vanilla_int8_upload, vanilla_int8_keys = federated_train(
                vanilla_factory, model_name, train_dataset, partitions, device,
                "VANILLA QAT + INT8 COMMUNICATION",
                comm_bits=8, qat_bits=8,
                initial_state=vanilla_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )
            adapter_int8, adapter_int8_upload, adapter_int8_keys = federated_train(
                adapter_factory, model_name, train_dataset, partitions, device,
                "ADAPTER QAT + INT8 COMMUNICATION",
                comm_bits=8, qat_bits=8,
                initial_state=adapter_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )

            vanilla_int4, vanilla_int4_upload, vanilla_int4_keys = federated_train(
                vanilla_factory, model_name, train_dataset, partitions, device,
                "VANILLA QAT + INT4 COMMUNICATION",
                comm_bits=4, qat_bits=4,
                initial_state=vanilla_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )
            adapter_int4, adapter_int4_upload, adapter_int4_keys = federated_train(
                adapter_factory, model_name, train_dataset, partitions, device,
                "ADAPTER QAT + INT4 COMMUNICATION",
                comm_bits=4, qat_bits=4,
                initial_state=adapter_initial_state,
                experiment_seed=experiment_seed,
                dataset_name=dataset_name,
            )

            vanilla_int8_eval = fake_quantize_model_for_final_eval(
                vanilla_int8, vanilla_int8_keys, bits=8,
            )
            adapter_int8_eval = fake_quantize_model_for_final_eval(
                adapter_int8, adapter_int8_keys, bits=8,
            )
            vanilla_int4_eval = fake_quantize_model_for_final_eval(
                vanilla_int4, vanilla_int4_keys, bits=4,
            )
            adapter_int4_eval = fake_quantize_model_for_final_eval(
                adapter_int4, adapter_int4_keys, bits=4,
            )

            row = {
                "model": display_name(model_name),
                "vanilla_fp32_acc":   evaluate_model(vanilla_fp32,      test_loader, device, class_names),
                "vanilla_fp32_upload": vanilla_fp32_upload,
                "adapter_fp32_acc":   evaluate_model(adapter_fp32,      test_loader, device, class_names),
                "adapter_fp32_upload": adapter_fp32_upload,
                "vanilla_int8_acc":   evaluate_model(vanilla_int8_eval, test_loader, device, class_names),
                "vanilla_int8_upload": vanilla_int8_upload,
                "adapter_int8_acc":   evaluate_model(adapter_int8_eval, test_loader, device, class_names),
                "adapter_int8_upload": adapter_int8_upload,
                "vanilla_int4_acc":   evaluate_model(vanilla_int4_eval, test_loader, device, class_names),
                "vanilla_int4_upload": vanilla_int4_upload,
                "adapter_int4_acc":   evaluate_model(adapter_int4_eval, test_loader, device, class_names),
                "adapter_int4_upload": adapter_int4_upload,
            }
            results.append(row)

            print("\nFinal full state sizes for reference:")
            print(f"  Vanilla FP32 full state:                  {get_full_state_size_mb(vanilla_fp32):.2f} MB")
            print(f"  Adapter FP32 full state:                  {get_full_state_size_mb(adapter_fp32):.2f} MB")
            print(f"  Vanilla INT8 communicated final params:   {estimate_quantized_model_size_mb(vanilla_int8, vanilla_int8_keys, 8):.2f} MB")
            print(f"  Adapter INT8 communicated final params:   {estimate_quantized_model_size_mb(adapter_int8, adapter_int8_keys, 8):.2f} MB")
            print(f"  Vanilla INT4 communicated final params:   {estimate_quantized_model_size_mb(vanilla_int4, vanilla_int4_keys, 4):.2f} MB")
            print(f"  Adapter INT4 communicated final params:   {estimate_quantized_model_size_mb(adapter_int4, adapter_int4_keys, 4):.2f} MB")

            print_results_table(results, dataset_name)

        print_results_table(results, dataset_name)


def run_with_keep_small_tensors_fp32(value: bool):
    global KEEP_SMALL_TENSORS_FP32
    KEEP_SMALL_TENSORS_FP32 = value
    print("=" * 80)
    print(f"RUNNING BENCHMARK WITH KEEP_SMALL_TENSORS_FP32 = {KEEP_SMALL_TENSORS_FP32}")
    print("=" * 80)
    return run_benchmark()


if __name__ == "__main__":
    run_with_keep_small_tensors_fp32(True)
