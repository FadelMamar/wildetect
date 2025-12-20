import itertools
import os
from typing import Any, Dict, List
from omegaconf import OmegaConf, DictConfig, ListConfig
import optuna
from .classification_trainer import ClassifierTrainer
from abc import ABC, abstractmethod
from ..utils.logging import get_logger

logger = get_logger(__name__)
class Sweeper(ABC):
    
    @abstractmethod
    def __call__(self, trial: optuna.Trial) -> Any:
        pass

    @abstractmethod
    def run(self):
        pass

class ClassifierSweeper(Sweeper):
    def __init__(self, sweep_config_path: str,debug:bool=False):
        self.sweep_config_path = sweep_config_path
        self.sweep_cfg = DictConfig(OmegaConf.load(self.sweep_config_path))
        self.base_cfg = DictConfig(OmegaConf.load(self.sweep_cfg.base_config))
        self.counter = 0
        self.debug = debug
    def __call__(self,trial:optuna.Trial):
        self.counter += 1

        # Model params
        model_params = self.sweep_cfg.parameters.model
        self.base_cfg.model.backbone = trial.suggest_categorical("backbone", model_params.backbone)
        self.base_cfg.model.dropout = trial.suggest_categorical("dropout", model_params.dropout)
        
        # Train params
        train_params = self.sweep_cfg.parameters.train
        lr = trial.suggest_categorical("lr", train_params.lr)
        lrf = trial.suggest_categorical("lrf", train_params.lrf)
        label_smoothing = trial.suggest_categorical("label_smoothing", train_params.label_smoothing)
        weight_decay = trial.suggest_categorical("weight_decay", train_params.weight_decay)
        batch_size = trial.suggest_categorical("batch_size", train_params.batch_size)

        self.base_cfg.train.lr = lr
        self.base_cfg.train.lrf = lrf
        self.base_cfg.train.label_smoothing = label_smoothing
        self.base_cfg.train.weight_decay = weight_decay
        self.base_cfg.train.batch_size = batch_size

        self.base_cfg.mlflow.run_name = f"trial_{self.counter}_{self.base_cfg.model.backbone}"
        self.base_cfg.mlflow.experiment_name = self.sweep_cfg.sweep_name
        self.base_cfg.checkpoint.dirpath = f"checkpoints/classification_sweeps/{self.sweep_cfg.sweep_name}"

        logger.info(f"Running trial {self.counter} with params: {self.base_cfg}")
        
        # Train
        trainer = ClassifierTrainer(self.base_cfg)
        trainer.run(debug=self.debug)

        return trainer.best_model_score

    def run(self):
        
        study = optuna.create_study(
            direction="maximize",
            study_name=self.sweep_cfg.sweep_name,
            storage="sqlite:///{}.db".format(self.sweep_cfg.sweep_name),
            sampler=optuna.samplers.TPESampler(seed=self.sweep_cfg.seed),
            load_if_exists=True
        )

        study.optimize(self, n_trials=self.sweep_cfg.n_trials)
        
