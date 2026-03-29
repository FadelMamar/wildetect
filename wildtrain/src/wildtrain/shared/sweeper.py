"""Base Sweeper abstract class for hyperparameter optimization.

This module provides the abstract base class for both training sweepers
and inference calibrators that use Optuna for optimization.
"""

from abc import ABC, abstractmethod
from typing import Any

import optuna


class Sweeper(ABC):
    """Abstract base class for Optuna-based hyperparameter optimization.

    Subclasses should implement:
    - __call__: The objective function for a single trial
    - run: The main optimization loop
    """

    @abstractmethod
    def __call__(self, trial: optuna.Trial) -> Any:
        """Objective function called by Optuna for each trial.

        Args:
            trial: Optuna trial object for suggesting hyperparameters.

        Returns:
            The objective value to optimize (e.g., F1 score, mAP).
        """
        pass

    @abstractmethod
    def run(self):
        """Run the optimization process.

        This method should create an Optuna study and call study.optimize()
        with self as the objective function.
        """
        pass
