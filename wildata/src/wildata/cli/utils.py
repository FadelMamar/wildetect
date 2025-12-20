"""
Utility functions for CLI commands.
"""

from pathlib import Path


def create_dataset_name(file_name: str) -> str:
    """Create a dataset name from a file name."""
    return Path(file_name).stem.replace(" ", "").replace(",", "-").lower()
