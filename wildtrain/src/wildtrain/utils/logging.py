import logging
import sys
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).parents[3]
ENV_FILE = ROOT / ".env"

def setup_logging(level="INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, level.upper(), logging.INFO)
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    if isinstance(log_file, str):
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
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
