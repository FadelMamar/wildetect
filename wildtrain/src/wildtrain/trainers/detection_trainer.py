from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
import mlflow
from ultralytics import YOLO
from ultralytics import settings
import os
import pandas as pd
from pathlib import Path

from ..utils.logging import ROOT, get_logger
from .base import ModelTrainer
from .yolo_utils import CustomYOLO
from ..utils.io import merge_data_cfg,remove_label_cache
from ..data.filters.algorithms import FilterDataCfg

logger = get_logger(__name__)



class UltralyticsDetectionTrainer(ModelTrainer):
    """
    Trainer class for object detection models using Ultralytics YOLO.
    This class handles training using parameters from a DictConfig (e.g., from yolo.yaml).
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model: Optional[YOLO] = None
        self.save_dir  = ROOT / "data" / "yolo_trainer"
        self.single_class_name="wildlife"

        if not self.config.dataset.load_as_single_class:
            raise ValueError("Not supported. Current pipeline only trains a localizer.")
        
        assert ((self.config.dataset.root_data_directory is not None) ^
                (self.config.dataset.data_cfg is not None)), "Either root_data_directory or data_cfg must be provided"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._set_data_cfg()

        self.filter = FilterDataCfg(data_config_yaml=self.config.dataset.data_cfg,
            keep_classes=self.config.dataset.keep_classes,
            discard_classes=self.config.dataset.discard_classes)
    
    def _set_data_cfg(self):
        assert (self.config.dataset.data_cfg is not None) ^ (self.config.dataset.root_data_directory is not None), "Either data_cfg or root_data_directory must be provided"
        
        # updating data_cfg
        if self.config.dataset.root_data_directory is not None:
            data_cfg = self.save_dir/f"{self.config.name}_merged.yaml"
            merge_data_cfg(root_data_directory=self.config.dataset.root_data_directory,
                                                        output_path=data_cfg,
                                                        force_merge=self.config.dataset.force_merge)
            self.config.dataset.data_cfg = data_cfg      
        
    def validate_config(self) -> None:
        if (
            self.config.model.architecture_file is None
            and self.config.model.weights is None
        ):
            raise ValueError("Either architecture_file or weights must be provided")

        if self.config.dataset.data_cfg is None:
            raise ValueError("data_cfg must be provided")        
            
    def pretrain(self,debug:bool=False):
        """
        Run pretraining phase for the model if enabled in configuration.
        """
        if not os.path.exists(self.config.pretraining.data_cfg):
            raise FileNotFoundError(f"Pretraining data config file not found: {self.config.pretraining.data_cfg}")
        
        logger.info("\n\n------------ Pretraining ----------\n")
        
        self.config.name += f"-PTR_freeze_{self.config.train.freeze}"
        self.config.train.epochs = self.config.train.ptr_epochs
        self.config.train.lr0 = self.config.train.ptr_lr0
        self.config.train.lrf = self.config.train.ptr_lrf
        self.config.train.freeze = self.config.train.ptr_freeze
        self.config.dataset.data_cfg = self.config.pretraining.data_cfg
        

        save_dir = self.config.pretraining.save_dir
        if save_dir is None:
            save_dir = ROOT / "data" / "pretraining"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        logger.info(f"Pretraining will be saved in {self.save_dir}")

        self._train(debug=debug)
    
    def curriculum_learning(self, debug:bool=False):
        """
        Run continual learning strategy for the model.

        Args:
            img_glob_pattern (str, optional): Glob pattern for images.
        """
        save_dir = self.config.curriculum.save_dir
        if save_dir is None:
            save_dir = ROOT / "data" / "curriculum_learning"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        logger.info(f"Continual Learning will be saved in {self.save_dir}")

        logger.info("\n\n------------ Continual Learning ----------\n")

        for flag in (self.config.curriculum.ratios, self.config.curriculum.epochs, self.config.curriculum.freeze):
            assert len(flag) == len(self.config.curriculum.lr0s), (
                f"All cl_* flags should match length. {len(flag)} != {len(self.config.curriculum.lr0s)}"
            )
                
        original_run_name = self.config.name
        for lr, ratio, num_epochs, freeze in zip(
            self.config.curriculum.lr0s, self.config.curriculum.ratios, self.config.curriculum.epochs, self.config.curriculum.freeze,
        ):
            # load best weights
            if self.best_model_path is not None:
                logger.info(f"Loading best model from {self.best_model_path}")
                self.config.model.weights = self.best_model_path

            cl_cfg_path = self.filter.get_data_cfg_paths_for_cl(
                ratio=ratio,
                split="train",
                cl_save_dir=self.save_dir,
                seed=self.config.train.seed,
            )
            self.config.name = f"{original_run_name}-cl_ratio-{ratio}_freeze-{freeze}"
            self.config.train.freeze = freeze
            self.config.train.lr0 = lr
            self.config.train.epochs = num_epochs
            self.config.dataset.data_cfg = cl_cfg_path

            self._train(debug=debug)

    def standard_training(self,debug:bool=False):
        """
        Run standard training phase for the model.
        """
        logger.info("\n\n------------ Standard Training ----------\n")
        self.config.dataset.data_cfg = self.filter.get_filtered_data_cfg(save_dir=self.save_dir)
        self._train(debug=debug)

    def get_model(self,):
        """Returns a customized detection model instance configured with specified config and weights."""

        # Load model
        if self.config.use_custom_yolo:
            model = CustomYOLO(model=self.config.model.architecture_file or self.config.model.weights,
            task="detect",
            **self.config.custom_yolo_kwargs,
            )
        else:
            model = YOLO(
                model=self.config.model.architecture_file or self.config.model.weights,
            task="detect",
        )

        logger.info(f"Loading model of type: {model.__class__.__name__}.")

        if self.config.model.weights is not None:
            model.load(self.config.model.weights)

        return model

    def run(self,debug:bool=False) -> None:

        if self.config.pretraining.data_cfg is not None:
            self.pretrain(debug=debug)
        
        if self.config.curriculum.enable:
            self.curriculum_learning(debug=debug)
        else:
            self.standard_training(debug=debug)
    
    def _train(self,debug:bool=False) -> None:
        """
        Run training phase for the model.
        """
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        
        settings.update({"mlflow": True})

        self.validate_config()

        # Load model
        self.model =  self.get_model()
        
        # Training parameters
        train_cfg = dict(self.config.train)

        if isinstance(self.config.dataset.data_cfg, list):
            data_cfg = os.path.join(self.save_dir, f"{self.config.name}_merged.yaml")
            merge_data_cfg(data_configs=self.config.dataset.data_cfg, 
            output_path=data_cfg,
            force_merge=self.config.dataset.force_merge)
        else:
            assert isinstance(self.config.dataset.data_cfg, (str,Path)), f"data_cfg must be str or path, got {type(self.config.dataset.data_cfg)}"
            data_cfg = self.config.dataset.data_cfg  

        remove_label_cache(data_cfg)      

        # Run training
        self.model.train(
            single_cls=self.config.dataset.load_as_single_class,
            data=data_cfg,
            name=self.config.name,
            project=self.config.project,
            time=2/60 if debug else None,
            save=True,
            **train_cfg,
        )
        
        # record path to the best model w.r.t mAP50
        self.best_model_path = self.model.trainer.best
