import os
from pathlib import Path
from typing import Optional
from ..trainers.detection_trainer import UltralyticsDetectionTrainer
from ..evaluators.ultralytics import UltralyticsEvaluator
from ..utils.logging import get_logger
from ..shared.models import DetectionPipelineConfig, DetectionEvalConfig, DetectionConfig

logger = get_logger(__name__)

class DetectionPipeline:
    """
    Orchestrates the full object detection pipeline: training, evaluation, and report saving.
    """
    def __init__(self, config_path: str):
        self.config = DetectionPipelineConfig.from_yaml(config_path)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        self.best_model_path: Optional[str] = None
        self.run_name: Optional[str] = None

    def train(self):
        logger.info("[Pipeline] Starting training...")
        train_config = DetectionConfig.from_yaml(self.config.train.config)
        self.run_name = train_config.name
        trainer = UltralyticsDetectionTrainer(train_config)
        trainer.run(debug=self.config.train.debug)
        self.best_model_path = trainer.best_model_path
        logger.info(f"[Pipeline] Best model path: {self.best_model_path}")
        logger.info("[Pipeline] Training completed.")

    def evaluate(self):
        logger.info("[Pipeline] Starting evaluation...")
        eval_config = DetectionEvalConfig.from_yaml(self.config.eval.config)
        eval_config.weights.localizer = self.best_model_path

        evaluator = UltralyticsEvaluator(config=eval_config)
        results = evaluator.evaluate(debug=self.config.eval.debug,
                                     save_path=os.path.join(self.config.results_dir, f"{self.run_name}_eval_report.json")
                                    )
        logger.info("[Pipeline] Evaluation completed.")
        return results

    def run(self):
        if not self.config.disable_train:
            self.train()
        else:
            logger.info("[Pipeline] Training disabled.")

        if not self.config.disable_eval:
            results = self.evaluate()
            return results
        else:
            logger.info("[Pipeline] Evaluation disabled.")
            return None
