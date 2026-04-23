"""
Logging setup for training runs.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = 'gcn',
                 log_dir: str = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and optional file output.

    Args:
        name: Logger name.
        log_dir: If provided, also write logs to a file in this directory.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / 'train.log', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
