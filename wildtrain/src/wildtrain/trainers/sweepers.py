"""Legacy sweep entry points.

``ClassifierSweeper`` and ``DetectionSweeper`` still accept the old sweep YAML.
They convert it and run :class:`wildtrain.harness.sweep.ExperimentSweeper`.
"""

from pathlib import Path

import optuna

from ..harness.sweep import (
    ExperimentSweeper,
    experiment_from_legacy_sweep,
    save_study,
)
from ..shared.models import ClassificationSweepConfig, DetectionSweepConfig
from ..shared.sweeper import Sweeper
from ..utils.logging import get_logger

logger = get_logger(__name__)


class _LegacySweeper(Sweeper):
    """Shared adapter around the harness sweep runner."""

    sweep_config_cls = ClassificationSweepConfig

    def __init__(self, sweep_config_path: str, debug: bool = False):
        self.sweep_config_path = sweep_config_path
        self.sweep_cfg = self.sweep_config_cls.from_yaml(self.sweep_config_path)
        self.debug = debug
        self._runner = ExperimentSweeper(
            experiment_from_legacy_sweep(self.sweep_cfg), debug=debug
        )
        self.study = None

    def __call__(self, trial: optuna.Trial):
        return self._runner(trial)

    def run(self):
        self.study = self._runner.run()
        output = self.sweep_cfg.output
        if output is not None and output.save_results and self.study is not None:
            directory = output.directory or (
                f"results/sweeps/{self.sweep_cfg.sweep_name}"
            )
            try:
                save_study(
                    self.study,
                    Path(directory),
                    output_format=output.format,
                    include_history=output.include_optimization_history,
                    save_plots=output.save_plots,
                )
                logger.info("Sweep results saved to %s", directory)
            except Exception as exc:
                logger.warning("Failed to save sweep results: %s", exc)
        return self.study


class ClassifierSweeper(_LegacySweeper):
    """Classification sweep adapter. Direction is maximize, score is val F1."""

    sweep_config_cls = ClassificationSweepConfig


class DetectionSweeper(_LegacySweeper):
    """Detection sweep adapter. Direction and objective come from the YAML."""

    sweep_config_cls = DetectionSweepConfig
