# FL QAT Quantized Communication Benchmark

Federated learning benchmark for comparing vanilla FL and adapter-based FL under FP32, INT8, and INT4 quantized communication. Local client training uses FedProx.

## What It Runs

By default, the benchmark runs every configured dataset with every configured default model.

Default datasets are configured in `config.py`:

```text
Lung Ultrasound
Brain Tumor
Fundus Diabetic Retinopathy
```

Default models are configured in `config.py`:

```text
mobilenet_v2
resnet18
vit_b_16
```

For each model and dataset, the full benchmark runs:

```text
VANILLA FP32 FL
ADAPTER FP32 FL
VANILLA QAT INT8
ADAPTER QAT INT8
VANILLA QAT INT4
ADAPTER QAT INT4
```

If you only want vanilla FedProx, use `--vanilla-fedprox-only`. That runs only:

```text
VANILLA FP32 FL
```

## Dataset Setup

Put datasets inside the root `datasets/` folder. The project uses `torchvision.datasets.ImageFolder`, so each `train` and `test` directory must contain one subfolder per class.

Expected structure:

```text
datasets/
├── lungs_ultrasound/
│   ├── train/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_1/
│       └── class_2/
├── brain_tumor/
│   ├── train/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_1/
│       └── class_2/
└── fundus_diabetic_retinopathy/
    ├── train/
    │   ├── class_1/
    │   └── class_2/
    └── test/
        ├── class_1/
        └── class_2/
```

Example:

```text
datasets/brain_tumor/train/glioma/image_001.png
datasets/brain_tumor/train/meningioma/image_002.png
datasets/brain_tumor/test/glioma/image_101.png
datasets/brain_tumor/test/meningioma/image_102.png
```

The folder names under `train/` and `test/` become the class names. The same class folders should exist in both splits.

## Run With `run.sh`

`run.sh` creates `venv`, installs the project in editable mode, and then forwards all arguments to `main.py`.

Run the full benchmark with defaults:

```bash
./run.sh
```

This uses all default datasets and all default models.

Run only vanilla FedProx with defaults:

```bash
./run.sh --vanilla-fedprox-only
```

This uses all default datasets and all default models, but skips adapter, INT8, and INT4 experiments.

Run one model only:

```bash
./run.sh --model resnet18
```

Run vanilla FedProx for one model only:

```bash
./run.sh --vanilla-fedprox-only --model resnet18
```

Run with custom training arguments:

```bash
./run.sh --model resnet18 --num-clients 4 --global-rounds 50 --local-epochs 5 --batch-size 32
```

## Manual Run

Use this if you do not want `run.sh` to manage the environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
python main.py
```

Manual full benchmark with arguments:

```bash
python main.py --model resnet18 --num-clients 4 --global-rounds 50 --local-epochs 5
```

Manual vanilla FedProx only:

```bash
python main.py --vanilla-fedprox-only
```

Manual vanilla FedProx for one model:

```bash
python main.py --vanilla-fedprox-only --model resnet18
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Random seed used for reproducibility. |
| `--num-clients` | `2` | Number of federated clients. |
| `--global-rounds` | `3` | Number of FL communication rounds for most datasets. |
| `--local-epochs` | `2` | Local training epochs per client per round for most datasets. |
| `--batch-size` | `32` | Batch size for train and test loaders. |
| `--dirichlet-beta` | `0.8` | Dirichlet beta for non-IID client partitioning. Lower values make client data more heterogeneous. |
| `--adapter-type` | `attention` | Adapter type. Choices: `attention`, `mlp`. |
| `--adapter-hidden-dim` | `512` | Hidden dimension used by adapter modules. |
| `--fedprox-mu` | `0.01` | FedProx proximal term strength. Use `0` to disable the proximal loss. |
| `--vanilla-fedprox-only` | `False` | Run only vanilla FP32 FedProx and skip adapter/QAT variants. |
| `--keep-small-tensors-fp32` | `True` | Keep small tensors such as biases and normalization parameters in FP32 during quantization. |
| `--no-keep-small-tensors-fp32` | - | Disable `--keep-small-tensors-fp32`. |
| `--use-error-feedback` | `True` | Use error feedback for quantized client updates. |
| `--no-use-error-feedback` | - | Disable error feedback. |
| `--quantize-downlink` | `True` | Quantize server-to-client downlink when running quantized communication rounds. |
| `--no-quantize-downlink` | - | Disable downlink quantization. |
| `--model` | Default model list | Run a single model instead of all models in `MODEL_NAME_DEFAULTS`. Choices: `mobilenet_v2`, `resnet18`, `densenet121`, `alexnet`, `vit_b_16`. |
| `--log-level` | `INFO` | Logging verbosity. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## Dataset-Specific Defaults

Most datasets use:

```text
global_rounds = 3
local_epochs = 2
quantized_warmup_rounds = 1
vanilla_lr = 1e-4
adapter_lr = 1e-3
adapter_backbone_lr = 5e-5
```

Brain Tumor has separate defaults in `config.py`:

```text
global_rounds = 5
local_epochs = 3
quantized_warmup_rounds = 2
vanilla_lr = 5e-5
adapter_lr = 5e-4
adapter_backbone_lr = 1e-5
```

## Output

The benchmark prints a summary table per dataset. Each result cell shows:

```text
accuracy% (average communication MB)
```

Communication MB includes both server-to-client download and client-to-server upload for communicated parameters.

## Project Layout

```text
main.py          - Entry point
benchmark.py     - Experiment orchestration
cli.py           - Argument parsing
config.py        - Config dataclass, dataset paths, default models
data_loaders.py  - Dataset loading and client partitioning
datasets/        - Local dataset folder
fed/             - FL server and client logic
models/          - Model architectures
quant/           - Quantization primitives, QAT, communication helpers
utils/           - Device, logging, evaluation, display
run.sh           - Automatic environment setup and runner
```
