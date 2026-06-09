import logging
import sys
from typing import TextIO


def setup_logging(
        level: int = logging.INFO,
        stream: TextIO = sys.stdout,
        fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    logger = logging.getLogger("fl_qat")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("fl_qat")
