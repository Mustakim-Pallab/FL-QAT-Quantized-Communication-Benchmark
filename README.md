# FL QAT Quantized Communication Benchmark

Federated learning benchmark comparing vanilla FL vs adapter-based FL under FP32, INT8, and INT4 quantization with quantized communication.

## Usage

### Option 1 — run.sh (automatic)

```bash
./run.sh                                          # defaults
./run.sh --model resnet18 --num-clients 4         # with options(add as per requirements)
```

Creates a venv and installs dependencies on first run automatically.

### Option 2 — manual

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
python main.py --model resnet18 --num-clients 4 --global-rounds 50 --local-epochs 5
```

Full options:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--num-clients` | 2 | Number of FL clients |
| `--global-rounds` | 3 | FL communication rounds |
| `--local-epochs` | 2 | Local training epochs per round |
| `--batch-size` | 32 | Batch size |
| `--dirichlet-beta` | 0.8 | Dirichlet partition beta |
| `--adapter-type` | attention | `attention` or `mlp` |
| `--adapter-hidden-dim` | 512 | Adapter hidden dimension |
| `--fedprox-mu` | 0.01 | FedProx proximal term |
| `--keep-small-tensors-fp32` | True | Keep small tensors in FP32 during quantization |
| `--no-keep-small-tensors-fp32` | — | Disable above |
| `--use-error-feedback` | True | Error feedback for quantized communication |
| `--no-use-error-feedback` | — | Disable above |
| `--quantize-downlink` | True | Quantize server-to-client downlink |
| `--no-quantize-downlink` | — | Disable above |
| `--model` | all | Single model to run |
| `--log-level` | INFO | Logging verbosity |

## Project Layout

```
main.py          — Entry point
benchmark.py     — Experiment orchestration
cli.py           — Argument parsing
config.py        — Config dataclass
data_loaders.py  — Dataset loading & partitioning
models/          — Model architectures
federation/      — FL server & client logic
quant/           — Quantization primitives (QAT, fake quant, comm)
utils/           — Device, logging, evaluation, display
```
