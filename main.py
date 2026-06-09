#!/usr/bin/env python3

from benchmark import run_benchmark
from cli import build_parser, config_from_args
from utils.logging_utils import setup_logging


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    config = config_from_args(args)
    run_benchmark(config)


if __name__ == "__main__":
    main()
