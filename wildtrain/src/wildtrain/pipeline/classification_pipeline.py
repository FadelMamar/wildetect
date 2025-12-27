import os
from typing import Optional
from wildtrain.trainers.classification_trainer import ClassifierTrainer
from wildtrain.evaluators.classification import ClassificationEvaluator
from wildtrain.utils.logging import get_logger
from wildtrain.shared.models import ClassificationPipelineConfig, ClassificationConfig, ClassificationEvalConfig
from pathlib import Path

logger = get_logger(__name__)

class ClassificationPipeline:
    """
    Orchestrates the full classification pipeline: training, evaluation, and report saving.
    """
    def __init__(self, config_path: str):
        self.config = ClassificationPipelineConfig.from_yaml(config_path)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        self.best_model_path: Optional[str] = None

    def train(self):
        logger.info("[Pipeline] Starting classification training...")
        train_config = ClassificationConfig.from_yaml(self.config.train.config)
        trainer = ClassifierTrainer(train_config)
        trainer.run(debug=self.config.train.debug)
        self.best_model_path = trainer.best_model_path
        logger.info(f"[Pipeline] Best model path: {self.best_model_path}")
        logger.info("[Pipeline] Training completed.")

    def evaluate(self):
        logger.info("[Pipeline] Starting classification evaluation...")
        eval_config = ClassificationEvalConfig.from_yaml(self.config.eval.config)
        
        if self.best_model_path is not None:
            eval_config.classifier = self.best_model_path

        evaluator = ClassificationEvaluator(config=eval_config)
        results = evaluator.evaluate(debug=self.config.eval.debug, 
        save_path=os.path.join(self.config.results_dir, "eval_report.json"))
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
