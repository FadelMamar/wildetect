"""One train-then-eval pipeline for either task.

The YAML shape is unchanged: ``train.config``, ``eval.config``, ``disable_train``,
``disable_eval``, and ``results_dir``. The eval file is a standalone eval document.
"""

import json
from pathlib import Path
from typing import Optional

from ..shared.models import ClassificationPipelineConfig, DetectionPipelineConfig
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
        self.run_name = experiment.name
        logger.info("[Pipeline] Best model path: %s", self.best_model_path)
        return result

    def evaluate(self):
        from .ops import load_eval_document
        from .runner import default_registry

        logger.info("[Pipeline] Starting %s evaluation...", self.task)
        document = load_eval_document(self.config.eval.config)
        if document.task != self.task:
            raise ValueError(
                f"Pipeline is {self.task}, but the eval file is {document.task}"
            )
        if self.best_model_path is not None:
            document.weights = self.best_model_path
        report = default_registry().get(self.task).evaluate(
            document, weights=self.best_model_path, debug=self.config.eval.debug
        )
        destination = Path(self.config.results_dir) / "eval_report.json"
        destination.write_text(json.dumps(report, default=str), encoding="utf-8")
        return report

    def run(self):
        if not self.config.disable_train:
            self.train()
        else:
            logger.info("[Pipeline] Training disabled.")
        if not self.config.disable_eval:
            return self.evaluate()
        logger.info("[Pipeline] Evaluation disabled.")
        return None
