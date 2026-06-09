import argparse

from config import Config
from utils.display import DISPLAY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Federated Learning QAT Benchmark with Quantized Communication",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--global-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dirichlet-beta", type=float, default=0.8)
    parser.add_argument("--adapter-type", type=str, default="attention", choices=["attention", "mlp"])
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--keep-small-tensors-fp32", action="store_true", default=True)
    parser.add_argument("--no-keep-small-tensors-fp32", dest="keep_small_tensors_fp32", action="store_false")
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--use-error-feedback", action="store_true", default=True)
    parser.add_argument("--no-use-error-feedback", dest="use_error_feedback", action="store_false")
    parser.add_argument("--quantize-downlink", action="store_true", default=True)
    parser.add_argument("--no-quantize-downlink", dest="quantize_downlink", action="store_false")
    parser.add_argument("--model", type=str, default=None, choices=list(DISPLAY_NAMES.keys()))
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        seed=args.seed, num_clients=args.num_clients,
        global_rounds=args.global_rounds, local_epochs=args.local_epochs,
        batch_size=args.batch_size, dirichlet_beta=args.dirichlet_beta,
        adapter_type=args.adapter_type, adapter_hidden_dim=args.adapter_hidden_dim,
        keep_small_tensors_fp32=args.keep_small_tensors_fp32,
        fedprox_mu=args.fedprox_mu, use_error_feedback=args.use_error_feedback,
        quantize_downlink=args.quantize_downlink,
    )
    if args.model:
        config = config.with_overrides(model_names=(args.model,))
    return config
