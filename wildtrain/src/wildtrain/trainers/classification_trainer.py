import traceback
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Any, Sequence
import mlflow
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
import torch
from wildata.pipeline import PathManager
from dotenv import load_dotenv
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Any, Optional, Tuple

from ..models.classifier import GenericClassifier
from ..data import ClassificationDataModule
from ..data.curriculum import CurriculumCallback
from ..utils.logging import ROOT, ENV_FILE
from ..utils.logging import get_logger
from ..utils.dvc_tracker import DVCTracker
from ..utils.transforms import create_transforms
from .base import ModelTrainer

logger = get_logger(__name__)


def track_dataset(cfg: DictConfig) -> None:
    """
    Track the dataset with DVC and log the dataset information to MLflow.
    """
    dvc_tracker = DVCTracker(ROOT / "dvc")
    path_manager = PathManager(cfg.dataset.root_data_directory)
    dataset_path = path_manager.get_framework_dir("roi").resolve()
    dataset_name = cfg.get("dataset_name", dataset_path.name)
    try:
        dvc_tracker.track_dataset_for_training(
            dataset_path, dataset_name, link_to_mlflow=True
        )
    except Exception:
        logger.warning(f"Failed to track dataset with DVC: {traceback.format_exc()}")


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        model: GenericClassifier,
        epochs: int = 20,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        lrf: float = 1e-2,
        weight_decay: float = 5e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = int(model.num_classes.item())
        self.label_to_class_map = model.label_to_class_map
        
        # Save model-specific hyperparameters for checkpoint restoration
        self.save_hyperparameters({
            'label_to_class_map': model.label_to_class_map,
            'backbone': model.backbone,
            'backbone_source': model.backbone_source,
            'input_size': model.input_size.item(),
            'dropout': model.dropout.item(),
            'freeze_backbone': model.freeze_backbone,
            'mean': model.mean.tolist(),
            'std': model.std.tolist(),
            'num_layers': model.num_layers.item(),
            'hidden_dim': model.hidden_dim.item(),
        })
        
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None

        # metrics
        cfg = {
            "task": "multiclass", 
            "num_classes": self.num_classes, 
            "average": None
        }
        self.accuracy = Accuracy(**cfg)
        self.precision = Precision(**cfg)
        self.recall = Recall(**cfg)
        self.f1score = F1Score(**cfg)
        self.ap = AUROC(**cfg)

        self.metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1score": self.f1score,
        }

        self.label_smoothing = label_smoothing            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        classes = y.cpu().flatten().tolist()
        weight = [
            len(classes) / torch.log2(torch.tensor(classes.count(i) + 2)).item() for i in range(self.num_classes)
        ]
        weight = torch.Tensor(weight).float().to(y.device)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.long().flatten(),  # Use flatten instead of squeeze(1)
            label_smoothing=self.label_smoothing,
            weight=weight,
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y = y.long().flatten()  # Use flatten instead of squeeze(1)

        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        for name, metric in self.metrics.items():
            metric.update(logits.softmax(dim=1), y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if self.mlflow_run_id is None:
            self.mlflow_run_id = getattr(self.logger, "run_id", None)
            self.mlflow_experiment_id = getattr(self.logger, "experiment_id", None)
            if self.mlflow_run_id is not None:
                logger.info(f"MLflow run_id: {self.mlflow_run_id}")
                logger.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
            else:
                logger.warning("No mlflow logger found")
                self.mlflow_run_id = "None"

        for name, metric in self.metrics.items():
            score = metric.compute().cpu()
            self.log(f"val_{name}", score.mean())
            for i, score in enumerate(score):
                cls_name = self.label_to_class_map.get(i, i)
                self.log(f"val_{name}_class_{cls_name}", score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [lr_scheduler]


class ClassifierTrainer(ModelTrainer):
    """
    Trainer class for image classification models.

    This class handles the training and evaluation of classification models
    using PyTorch Lightning and MLflow for experiment tracking.
    """

    def get_callbacks(self) -> tuple[list[Any], MLFlowLogger]:
        """
        Get the callbacks for the trainer.
        """
        assert ENV_FILE.exists(), "Environment file not found"
        load_dotenv(ENV_FILE, override=True)
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        if MLFLOW_TRACKING_URI is None:
            raise ValueError("MLFLOW_TRACKING_URI is not set")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.checkpoint.monitor,
            save_top_k=self.config.checkpoint.save_top_k,
            save_last=self.config.checkpoint.save_last,
            mode=self.config.checkpoint.mode,
            dirpath=ROOT / self.config.checkpoint.dirpath / self.config.mlflow.run_name,
            filename=self.config.checkpoint.filename,
            save_weights_only=self.config.checkpoint.save_weights_only,
            save_on_train_epoch_end=False
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        early_stopping = EarlyStopping(
            monitor=self.config.checkpoint.monitor,
            patience=self.config.checkpoint.patience,
            mode=self.config.checkpoint.mode,
            min_delta=self.config.checkpoint.min_delta,
        )
        mlflow_logger = MLFlowLogger(
            experiment_name=self.config.mlflow.experiment_name,
            run_name=self.config.mlflow.run_name,
            log_model=False,
            #checkpoint_path_prefix="classification",
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        
        # Create base callbacks list
        callbacks = [checkpoint_callback, early_stopping, lr_callback]
        
        return callbacks, mlflow_logger

    def log_model(self, model: ClassifierModule) -> None:
        if model.mlflow_run_id and self.best_model_path:
            with mlflow.start_run(run_id=model.mlflow_run_id):
                try:
                    mlflow.log_artifact(str(self.best_model_path), "checkpoint")
                    cfg_path = Path(self.best_model_path).with_name("config.yaml")
                    OmegaConf.save(self.config, cfg_path)
                    mlflow.log_artifact(str(cfg_path), "config")
                except Exception:
                    logger.error(f"Error logging best model: {traceback.format_exc()}")

                if self.config.get("track_dataset"):
                    track_dataset(self.config)

    def run(self, debug: bool = False) -> None:
        """
        Run image classification training or evaluation based on config.
        """
        # Set float32 matmul precision to medium
        if torch.cuda.is_available():   
            torch.set_float32_matmul_precision('medium')

        # DataModule - use the new classmethod for cleaner instantiation
        datamodule = ClassificationDataModule.from_dict_config(self.config)
        
        logger.info(f"Getting one batch of data to initialize lazy modules in classifier.")
        datamodule.setup(stage="fit")
        example_input, _ = next(iter(datamodule.train_dataloader()))

        # Model
        model_cfg = self.config.model
        if model_cfg.weights is not None:
            cls_model = GenericClassifier.load_from_checkpoint(model_cfg.weights)
            logger.info(f"Loaded model from checkpoint: {model_cfg.weights}")
        else:
            cls_model = GenericClassifier(
                backbone=model_cfg.backbone,
                pretrained=model_cfg.pretrained,
                backbone_source=model_cfg.backbone_source,
                label_to_class_map=datamodule.class_mapping,
                dropout=model_cfg.dropout,
                freeze_backbone=model_cfg.freeze_backbone,
                input_size=model_cfg.input_size,
                mean=model_cfg.mean,
                std=model_cfg.std,
                num_layers=model_cfg.num_layers,
                hidden_dim=model_cfg.hidden_dim,
            )
        model = ClassifierModule(
            epochs=self.config.train.epochs,
            model=cls_model,
            lr=self.config.train.lr,
            label_smoothing=self.config.train.label_smoothing,
            weight_decay=self.config.train.weight_decay,
            lrf=self.config.train.lrf,
        )
        model.example_input_array = example_input
        callbacks, mlflow_logger = self.get_callbacks()
        
        # Add curriculum callback if curriculum is enabled
        if datamodule.is_curriculum_enabled():
            logger.info("Curriculum learning is enabled. Adding CurriculumCallback.")
            curriculum_callback = CurriculumCallback(datamodule)
            callbacks.insert(0, curriculum_callback)  # Add at the beginning to ensure it runs early
        else:
            logger.info("Curriculum learning is disabled.")
        
        trainer = Trainer(
            max_epochs=self.config.train.epochs if not debug else 1,
            accelerator=self.config.train.accelerator,
            precision=self.config.train.precision,
            logger=mlflow_logger,
            limit_train_batches=3 if debug else None,
            check_val_every_n_epoch=self.config.train.val_check_interval,
            limit_val_batches=3 if debug else None,
            callbacks=callbacks,
            default_root_dir=ROOT / self.config.checkpoint.dirpath
            )

        trainer.fit(model, datamodule=datamodule)

        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score
    
        if self.config.mlflow.log_model:
            self.log_model(model)
