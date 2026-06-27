from typing import Any

DISPLAY_NAMES: dict[str, str] = {
    "mobilenet_v2": "MobileNetV2",
    "resnet18": "ResNet18",
    "densenet121": "DenseNet121",
    "alexnet": "AlexNet",
    "vit_b_16": "ViT-B/16",
}

COLUMNS = [
    "Model",
    "Vanilla FL (FP32)",
    "Adapter (FP32)",
    "Vanilla (INT8)",
    "Adapter (INT8)",
    "Vanilla (INT4)",
    "Adapter (INT4)",
]


def display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def metric_cell(accuracy: float, payload_mb: float) -> str:
    return f"{accuracy * 100:.2f}% ({payload_mb:.2f} MB)"


def print_results_table(results: list[dict[str, Any]], dataset_name: str) -> None:
    print("\n" + "=" * 155)
    print(f"FINAL BENCHMARK SUMMARY: {dataset_name}")
    print("Accuracy is final model accuracy. MB is average client communication per round.")
    print("Communication MB includes download + upload for the communicated parameters.")
    print("Adapter FL transmits only adapter+classifier params; backbone is never sent.")
    print("=" * 155)
    print(f"{COLUMNS[0]:<14} {COLUMNS[1]:<24} {COLUMNS[2]:<24} "
          f"{COLUMNS[3]:<24} {COLUMNS[4]:<24} {COLUMNS[5]:<24} {COLUMNS[6]:<24}")
    print("-" * 155)

    for row in results:
        print(f"{row['model']:<14} "
              f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload']):<24} "
              f"{metric_cell(row['adapter_fp32_acc'], row['adapter_fp32_upload']):<24} "
              f"{metric_cell(row['vanilla_int8_acc'], row['vanilla_int8_upload']):<24} "
              f"{metric_cell(row['adapter_int8_acc'], row['adapter_int8_upload']):<24} "
              f"{metric_cell(row['vanilla_int4_acc'], row['vanilla_int4_upload']):<24} "
              f"{metric_cell(row['adapter_int4_acc'], row['adapter_int4_upload']):<24}")

    print("\nLaTeX table rows:")
    print(r"\textbf{Model} & \textbf{Vanilla FL (FP32)} & \textbf{Adapter (FP32)} "
          r"& \textbf{Vanilla (INT8)} & \textbf{Adapter (INT8)} "
          r"& \textbf{Vanilla (INT4)} & \textbf{Adapter (INT4)} \\")
    print(r"\hline")
    for row in results:
        print(f"{row['model']} & "
              f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload'])} & "
              f"{metric_cell(row['adapter_fp32_acc'], row['adapter_fp32_upload'])} & "
              f"{metric_cell(row['vanilla_int8_acc'], row['vanilla_int8_upload'])} & "
              f"{metric_cell(row['adapter_int8_acc'], row['adapter_int8_upload'])} & "
              f"{metric_cell(row['vanilla_int4_acc'], row['vanilla_int4_upload'])} & "
              f"{metric_cell(row['adapter_int4_acc'], row['adapter_int4_upload'])} \\\\")


def print_vanilla_fedprox_results_table(results: list[dict[str, Any]], dataset_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"FINAL VANILLA FEDPROX SUMMARY: {dataset_name}")
    print("Accuracy is final model accuracy. MB is average client communication per round.")
    print("Communication MB includes download + upload for the communicated parameters.")
    print("=" * 72)
    print(f"{'Model':<14} {'Vanilla FedProx (FP32)':<24}")
    print("-" * 72)

    for row in results:
        print(f"{row['model']:<14} "
              f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload']):<24}")

    print("\nLaTeX table rows:")
    print(r"\textbf{Model} & \textbf{Vanilla FedProx (FP32)} \\")
    print(r"\hline")
    for row in results:
        print(f"{row['model']} & "
              f"{metric_cell(row['vanilla_fp32_acc'], row['vanilla_fp32_upload'])} \\")
