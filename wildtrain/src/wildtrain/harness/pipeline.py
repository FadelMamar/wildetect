"""One train-then-eval pipeline for either task.

The YAML shape is unchanged: ``train.config``, ``eval.config``, ``disable_train``,
``disable_eval``, and ``results_dir``. Training goes through ``run_experiment``.
Evaluation still uses the existing classification or detection evaluator and
points the weights at ``FitResult.best_path``.
"""

import os
from pathlib import Path
from typing import Optional

from ..shared.models import (
    ClassificationEvalConfig,
    ClassificationPipelineConfig,
    DetectionEvalConfig,
    DetectionPipelineConfig,
)
from ..utils.logging import get_logger
from .config import load_experiment
from .runner import run_experiment

logger = get_logger(__name__)


class ExperimentPipeline:
    """Run the train config, then the eval config, for one task."""

    def __init__(self, config_path: str, task: str):
        if task == "classification":
            self.config = ClassificationPipelineConfig.from_yaml(config_path)
        elif task == "detection":
            self.config = DetectionPipelineConfig.from_yaml(config_path)
        else:
            raise ValueError(
                f"Pipeline task must be classification or detection, got {task!r}"
            )
        self.task = task
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        self.best_model_path: Optional[str] = None
        self.run_name: Optional[str] = None

    def train(self):
        logger.info("[Pipeline] Starting %s training...", self.task)
        experiment = load_experiment(self.config.train.config)
        if experiment.task != self.task:
            raise ValueError(
                f"Pipeline is {self.task}, but {self.config.train.config} "
                f"is task {experiment.task}"
            )
        result = run_experiment(experiment, debug=self.config.train.debug)
        self.best_model_path = result.best_path
        self.run_name = experiment.run.name
        logger.info("[Pipeline] Best model path: %s", self.best_model_path)
        return result

    def evaluate(self):
        logger.info("[Pipeline] Starting %s evaluation...", self.task)
        if self.task == "classification":
            return self._evaluate_classifier()
        return self._evaluate_detector()

    def run(self):
        if not self.config.disable_train:
            self.train()
        else:
            logger.info("[Pipeline] Training disabled.")
        if not self.config.disable_eval:
            return self.evaluate()
        logger.info("[Pipeline] Evaluation disabled.")
        return None

    def _evaluate_classifier(self):
        from ..evaluators.classification import ClassificationEvaluator

        eval_config = ClassificationEvalConfig.from_yaml(self.config.eval.config)
        if self.best_model_path is not None:
            eval_config.classifier = self.best_model_path
        evaluator = ClassificationEvaluator(config=eval_config)
        return evaluator.evaluate(
            debug=self.config.eval.debug,
            save_path=os.path.join(self.config.results_dir, "eval_report.json"),
        )

    def _evaluate_detector(self):
        from ..evaluators.ultralytics import UltralyticsEvaluator

        eval_config = DetectionEvalConfig.from_yaml(self.config.eval.config)
        eval_config.weights.localizer = self.best_model_path
        evaluator = UltralyticsEvaluator(config=eval_config)
        name = self.run_name or "detection"
        return evaluator.evaluate(
            debug=self.config.eval.debug,
            save_path=os.path.join(self.config.results_dir, f"{name}_eval_report.json"),
        )
