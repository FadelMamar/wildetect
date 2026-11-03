"""Result of a harness training run.

Both wrapper tasks fill this from the existing trainers. Callers should read
these fields instead of reaching into Lightning or Ultralytics objects.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FitResult:
    """Checkpoint pointer shared by classification and detection wrappers."""

    best_path: Optional[str]
    best_score: Optional[float]
    score_name: Optional[str] = None
    metrics: Optional[dict] = None
