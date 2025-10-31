"""
Logging configuration for the WildTrain data pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level="INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, level.upper(), logging.INFO)
    handlers: list = [
        logging.StreamHandler(sys.stdout),
    ]
    if isinstance(log_file, str):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Ensure our config is always applied
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
