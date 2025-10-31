"""MLflow model registration utilities for WildTrain models."""

import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum

import numpy as np
import torch
import mlflow
from mlflow.models import infer_signature,ModelSignature
from ultralytics import YOLO
from pydantic import BaseModel, Field

from wildtrain.models.classifier import GenericClassifier
from wildtrain.models.detector import Detector
from wildtrain.utils.logging import get_logger, ROOT
from wildtrain.utils.io import save_yaml

import tomllib


from omegaconf import OmegaConf, DictConfig

# Disable torch XNNPACK for compatibility
os.environ["TORCH_XNNPACK_DISABLE"] = "1"

logger = get_logger(__name__)


class ModelTask(Enum):
    """Supported model tasks."""
    DETECT = "detect"
    CLASSIFY = "classify"
    SEGMENT = "segment"


class ModelType(Enum):
    """Supported model types."""
    DETECTOR = "detector"
    CLASSIFIER = "classifier"
    LOCALIZER = "localizer"


class ModelMetadata(BaseModel):
    """Metadata for model registration.
    
    This class provides structured metadata for MLflow model registration,
    including validation and default values.
    """
    batch: int = Field(default=8, gt=0, description="Batch size for inference")
    imgsz: int = Field(default=800, gt=0, description="Input image size")
    task: ModelTask = Field(default=ModelTask.DETECT, description="Model task type")
    num_classes: Optional[int] = Field(default=None, ge=2, description="Number of classes")
    model_type: ModelType = Field(description="Type of model (detector or classifier)")
    cls_export_format: Optional[str] = Field(default=None, description="Export format for the model")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow metadata."""
        return {
            "batch": self.batch,
            "imgsz": self.imgsz,
            "task": self.task.value,
            "num_classes": self.num_classes,
            "model_type": self.model_type.value,
            "cls_export_format": self.cls_export_format
        }


def normalize_path(model_path: Path) -> Path:
    """Normalize path for cross-platform compatibility."""
    resolved_path = Path(model_path).resolve()
    if platform.system().lower() != "windows":
        return Path(resolved_path.as_posix().replace("\\", "/"))
    return resolved_path

def read_dependencies():
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    dependencies = data["project"]["dependencies"]
    return dependencies

    
class ClassifierWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for classification models."""
    
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        """Handle serialization - exclude model objects."""
        state = self.__dict__.copy()
        # Remove any model references that can't be serialized
        #state.pop('model', None)
        #state.pop('artifacts', None)
        return state

    def __setstate__(self, state):
        """Handle deserialization."""
        self.__dict__.update(state)
        # Model will be loaded in load_context

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the TorchScript classification model from the artifacts."""
        model_path = Path(context.artifacts["classifier_ckpt"]).resolve()
        model_path = normalize_path(model_path)
        self.model = GenericClassifier.load_from_checkpoint(str(model_path))
        self.model.to_torchscript()
    
    def predict(self,context: mlflow.pyfunc.PythonModelContext,model_input):
        return self.model.predict(torch.tensor(model_input))

class DetectorWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Detector system."""
    
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        """Handle serialization - exclude model objects."""
        state = self.__dict__.copy()
        # Remove any model references that can't be serialized
        #state.pop('model', None)
        #state.pop('artifacts', None)
        return state

    def __setstate__(self, state):
        """Handle deserialization."""
        self.__dict__.update(state)
        # Model will be loaded in load_context

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load from the artifacts."""
        from wildtrain.cli.config_loader import ConfigLoader # avoid curcular import
        
        config = context.artifacts["config"]
        config = normalize_path(config)

        classifier_ckpt = context.artifacts.get("classifier_ckpt",None)
        if classifier_ckpt is not None:
            classifier_ckpt = normalize_path(classifier_ckpt)

        localizer_ckpt = context.artifacts["localizer_ckpt"]
        localizer_ckpt = normalize_path(localizer_ckpt)

        # Validate and load config
        ConfigLoader.load_detector_registration_config(config)
        config = OmegaConf.load(config)

        localizer_cfg = config.localizer.yolo or config.localizer.mmdet
        localizer_cfg.weights = localizer_ckpt

        # Check if the model can be loaded
        self.model =  Detector.from_config(localizer_config=localizer_cfg,
                                       classifier_ckpt=classifier_ckpt,
                                       classifier_export_kwargs=config.classifier.processing)
        #self.model.set_device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict(self,context: mlflow.pyfunc.PythonModelContext,model_input):
        """Predict from the model."""
        return self.model.predict(torch.tensor(model_input),return_as_dict=True)

def get_experiment_id(name: str) -> str:
    """Get or create an MLflow experiment ID.
    
    Args:
        name: MLflow experiment name
        
    Returns:
        Experiment ID
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created new experiment: {name} (ID: {exp_id})")
        return exp_id
    return exp.experiment_id

class ModelRegistrar:
    """Handles model registration to MLflow Model Registry."""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """Initialize the ModelRegistrar.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
    
    def register_detector(
        self,config_path:Path
    ) -> None:
        """Register a YOLO detection model to MLflow Model Registry.
        """
        from wildtrain.cli.config_loader import ConfigLoader # avoid curcular import
        # Validate and load config
        ConfigLoader.load_detector_registration_config(config_path)
        config = OmegaConf.load(config_path)

        assert (config.localizer.yolo is None) + (config.localizer.mmdet is None) == 1, f"Exactly one should be given"
        localizer_cfg = config.localizer.yolo or config.localizer.mmdet
        localizer_processing = config.localizer.processing

        self.tracking_uri = config.processing.mlflow_tracking_uri

        localizer_ckpt = self._export_localizer(
            model_path=Path(localizer_cfg.weights),
            export_format=localizer_processing.export_format,
            image_size=localizer_cfg.imgsz,
            batch_size=localizer_processing.batch_size,
            device=localizer_cfg.device,
            dynamic=localizer_processing.dynamic,
            task=localizer_cfg.task,
        )
        
        localizer_cfg.weights = str(localizer_ckpt)
        config_path = str(Path(localizer_cfg.weights).parent / "config.yaml")

        classifier_ckpt = config.classifier.weights
        artifacts = {"config":config_path,
                    "localizer_ckpt":str(localizer_ckpt)
                }      
        if classifier_ckpt is not None:
            artifacts["classifier_ckpt"] = str(classifier_ckpt)

        OmegaConf.save(config,config_path)
        
        # Check if the model can be loaded
        model =  Detector.from_config(localizer_config=localizer_cfg,
                                       classifier_ckpt=classifier_ckpt,
                                       classifier_export_kwargs=config.classifier.processing)
        if config.classifier.processing.export_format == "onnx":
            if model.classifier is not None:
                model.classifier.set_onnx_program(onnx_program=None)
        
        x = torch.rand(localizer_processing.batch_size,3,localizer_cfg.imgsz,localizer_cfg.imgsz)
        signature = infer_signature(x.cpu().numpy(), list(dict()))

        if getattr(localizer_cfg,"config",None) is not None:
            artifacts["mmdet_config"] = getattr(localizer_cfg,"config",None)
        
        # Create metadata using the new ModelMetadata class
        task = getattr(localizer_cfg,"task","detect")
        metadata = ModelMetadata(
            batch=localizer_processing.batch_size,
            imgsz=localizer_cfg.imgsz,
            task=ModelTask(task),
            model_type=ModelType.DETECTOR,
            cls_export_format=config.classifier.processing.export_format
        )
        
        self._register_model(
            metadata=metadata.to_dict(),
            artifacts=artifacts,
            name=config.processing.name,
            model=DetectorWrapper(),
            signature=signature, 
        )
        
        logger.info(f"Successfully registered detector model: {config.processing.name}")
        
    def register_classifier(
        self,
        weights: Union[str, Path],
        name: str = "classifier",
        export_format: str = "torchscript",
        batch_size: int = 8,
    ) -> None:
        """Register a classification model to MLflow Model Registry.
        """
        model_path = Path(weights)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        artifacts = {"classifier_ckpt":str(model_path),}
        
        # Load the model
        model = GenericClassifier.load_from_checkpoint(str(model_path))
        model = model.export(mode=export_format,
                             batch_size=batch_size,
                             output_path=Path(weights).with_suffix(f".{export_format}").as_posix()
                            )

        x = torch.rand(batch_size,3,model.input_size.item(),model.input_size.item())
        signature = infer_signature(x.cpu(), model.predict(x))
                
        metadata = ModelMetadata(
            batch=batch_size,
            imgsz=model.input_size.item(),
            task=ModelTask.CLASSIFY,
            num_classes=model.num_classes.item(),
            model_type=ModelType.CLASSIFIER,
            cls_export_format=export_format
        )
        
        self._register_model(
            metadata=metadata.to_dict(),
            name=name,
            model=ClassifierWrapper(),
            signature=signature,
            artifacts=artifacts,
        )
        
        logger.info(f"Successfully registered classifier model: {name}")
    
    def _export_localizer(
        self,
        model_path: Path,
        export_format: str,
        image_size: int,
        batch_size: int,
        device: str,
        dynamic: bool,
        task: str,
    ) -> Path:
        """Export the detector model in the specified format."""
        if export_format == "pt":
            return normalize_path(model_path)
        
        try:
            model = YOLO(model_path, task=task)
            model.export(
                format=export_format,
                imgsz=image_size,
                optimize=device == "cpu",
                nms=True,
                dynamic=dynamic,
                batch=batch_size,
                device=device,
            )
            
            if export_format == "openvino":
                export_path = model_path.with_name(f"{model_path.stem}_openvino_model")
            else:
                export_path = model_path.with_suffix(f".{export_format}")
            
            return normalize_path(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
        
    def _register_model(
        self,
        metadata: Dict[str, Any],
        name: str,
        model: Union[torch.nn.Module,mlflow.pyfunc.PythonModel],
        artifacts: Optional[Dict[str, str]] = None,
        signature: Optional[ModelSignature] = None,
    ) -> None:
        """Register a model to MLflow Model Registry."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        exp_id = get_experiment_id(name)
        
        with mlflow.start_run(experiment_id=exp_id):
            if isinstance(model,mlflow.pyfunc.PythonModel):
                mlflow.pyfunc.log_model(
                    python_model=model,
                    artifact_path="model",
                    pip_requirements=read_dependencies(),
                    registered_model_name=name,
                    signature=signature,
                    metadata=metadata,
                    artifacts=artifacts,
                )
            elif isinstance(model,torch.nn.Module):
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    pip_requirements=read_dependencies(),
                    registered_model_name=name,
                    signature=signature,
                    metadata=metadata,
                    artifacts=artifacts,
                )
            else:
                raise ValueError(f"Model type {type(model)} not supported")

