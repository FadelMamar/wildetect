import logging
import os
import traceback
from pathlib import Path
from time import time
from urllib.parse import quote, unquote
import pandas as pd
from dotenv import load_dotenv
from typing import Sequence

# from label_studio_ml.utils import get_local_path
from label_studio_tools.core.utils.io import get_local_path

from label_studio_sdk.client import LabelStudio
from PIL import Image

from ..data.tile import Tile
from ..registry import Detector
from ..config import PredictionConfig


logger = logging.getLogger(__name__)


class InferenceEngine(object):
    def __init__(self, config: PredictionConfig, detector: Detector, model_tag:str, roi_processor, dotenv_path:str=None):
        """
        Initialize the InferenceEngine with a prediction configuration.

        Args:
            config (PredictionConfig): Prediction configuration object.
        """
        self.config = config

        self.detector = None
        self.image_processor = None
        self.detection_processor = None
        self.model_tag = "None"

        # LS label config
        self.labelstudio_client: LabelStudio = None
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"
        self.detector = detector
        self.model_tag = model_tag
        self.roi_processor = roi_processor
        
        self.labelstudio_client = None
        try:
            self._set_ls_client(dotenv_path=dotenv_path)
        except Exception as e:
            logger.warning(f"Failed to set up Label Studio client: {e}")


    def _set_ls_client(self, dotenv_path: str):
        """
        Set up the Label Studio client using environment variables from a .env file.

        Args:
            dotenv_path (str): Path to the .env file containing API credentials.
        """
        # # Load environment variables
        load_dotenv(dotenv_path=dotenv_path)
        
        # # label studio client
        LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

        if LABEL_STUDIO_URL is None:
            raise ValueError("env variable LABEL_STUDIO_URL is not set.")
        if API_KEY is None:
            raise ValueError("env variable API_KEY is not set.")

        self.labelstudio_client = LabelStudio(
            base_url=LABEL_STUDIO_URL, api_key=API_KEY
        )

        return None


    def inference(
        self,
        tiles: list[Tile],
    ) -> list[Tile]:
        """
        Run multithreaded inference on a list of image paths or tiles.

        Args:
            images_paths (Sequence[str]): List of image paths to run inference on.
            tiles (list[Tile], optional): List of Tile objects to update with predictions.
            return_tiles (bool, optional): If True, return tiles with predictions.
            return_as_df (bool, optional): If True, return results as a DataFrame.

        Returns:
            Sequence[list[Detection]] | list[Tile] | pd.DataFrame: Detections, tiles, or DataFrame depending on arguments.
        """


        logger.info(f"Running inference on {len(tiles)} tiles.")

        detections = self.detector.predict(tiles=tiles)

        if len(detections) != len(tiles):
            raise ValueError(
                "Number of detections does not match number of images. {} != {}".format(
                    len(detections), len(tiles)
                )
            )

        for i, tile in enumerate(tiles):
            tile.set_predictions(detections[i])

        return tiles

    @classmethod
    def load_engine(
        cls,
        pred_config: PredictionConfig,
        roi_classifier_path: str = None,
        roi_cls_is_features: bool = True,
        roi_cls_label_map: dict = {0: "gt", 1: "tn"},
        roi_keep_classes: list = ["gt"],
        detection_label_map: dict = {0: "wildlife"},
        feature_extractor_path: str = "facebook/dinov2-with-registers-small",
        model_path: str = None,
        detection_model_type: str = "ultralytics",
        text_instruction: str = "detect wildlife species",
        mlflow_model_alias: str = "demo",
        mlflow_model_name: str = "labeler",
        set_ls_client: bool = False,
        buffer_size=24,
        timeout=60,
        dot_env_path: str = None,
    ) -> tuple:
        """
        Load and configure an InferenceEngine with all required models and processors.

        Args:
            pred_config (PredictionConfig): Prediction configuration.
            roi_classifier_path (str, optional): Path to ROI classifier checkpoint.
            roi_cls_is_features (bool, optional): Whether ROI classifier uses features.
            roi_cls_label_map (dict, optional): Label map for ROI classifier.
            roi_keep_classes (list, optional): Classes to keep after ROI classification.
            detection_label_map (dict, optional): Label map for detection model.
            feature_extractor_path (str, optional): Path or name of feature extractor.
            model_path (str, optional): Path to detection model weights.
            detection_model_type (str, optional): Type of detection model.
            text_instruction (str, optional): Instruction for detection model.
            mlflow_model_alias (str, optional): MLflow model alias.
            mlflow_model_name (str, optional): MLflow model name.
            set_ls_client (bool, optional): Whether to set up Label Studio client.
            dot_env_path (str, optional): Path to .env file for Label Studio.

        Returns:
            tuple: (InferenceEngine, feature_extractor)
        """
        if (model_path is None) and (pred_config.inference_service_url is None):
            logger.info(
                f"Loading model from mlflow name={mlflow_model_name}/alias={mlflow_model_alias} "
            )
            model, metadata = load_registered_model(
                alias=mlflow_model_alias,
                name=mlflow_model_name,
                mlflow_tracking_url="http://localhost:5000",
                load_unwrapped=True,
            )
            logger.info(f"model's metadata={metadata}")

            logger.info(f"{model.__class__.__name__} loaded successfully.")

            pred_config.batch_size = metadata.get("batch", pred_config.batch_size)
            pred_config.tilesize = metadata.get("tilesize", pred_config.tilesize)
            pred_config.imgsz = pred_config.tilesize

            detection_model = build_detector(
                detection_model_type=metadata["detection_model_type"],
                model_path=None,
                model=model,
                config=pred_config,
                text_instruction=text_instruction,
            )
        else:
            detection_model = build_detector(
                detection_model_type=detection_model_type,
                model_path=model_path,
                model=None,
                config=pred_config,
                text_instruction=text_instruction,
            )

        # build roi postprocessor
        feature_extractor = get_processor("feature_extractor")(
            hf_model_path=feature_extractor_path
        )

        roi_processor = None
        if roi_classifier_path is not None:
            model = ImageClassifier.load_from_checkpoint(
                roi_classifier_path,
                cls_is_features=roi_cls_is_features,
                map_location=pred_config.device,
            )

            roi_classifier = get_processor("classifier")(
                model,
                label_map=roi_cls_label_map,
                device=pred_config.device,
                feature_extractor=feature_extractor,
                imgsz=pred_config.cls_imgsz,
            )
            roi_processor = DetectionsPostprocessor(
                keep_classes=roi_keep_classes,
            )
            roi_processor.set_classifier(roi_classifier)

        # build object detection system
        detection_label_map = (
            getattr(detection_model, "names", None)
            or detection_label_map
            or {0: "wildlife"}
        )
        detector = ObjectDetectionSystem(
            config=pred_config,
            detection_label_map=detection_label_map,
            buffer_size=buffer_size,
            timeout=timeout,
        )
        detector.set_model(model=detection_model)
        detector.set_processor(roi_processor=roi_processor)

        engine = cls(config=pred_config)
        engine.set_detector(detector=detector, model_tag=mlflow_model_alias)

        if set_ls_client:
            engine.set_ls_client(dotenv_path=dot_env_path)

        return engine, feature_extractor

    @classmethod
    def from_yaml(cls, yaml_path: str) -> tuple:
        """
        Create an InferenceEngine from a YAML configuration file.

        This method provides a declarative way to configure and create an InferenceEngine
        by reading all parameters from a YAML file instead of passing them as function arguments.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Returns:
            tuple: (InferenceEngine, feature_extractor)

        Example:
            >>> engine, feature_extractor = InferenceEngine.from_yaml("configs/inference.yaml")
        """
        import yaml
        from pathlib import Path

        # Load YAML configuration
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract prediction configuration
        pred_config_dict = config.get("prediction", {})
        flight_specs_dict = config.get("flight_specs", {})

        # Create FlightSpecs object
        flight_specs = FlightSpecs(**flight_specs_dict)

        # Create PredictionConfig object
        pred_config = PredictionConfig(flight_specs=flight_specs, **pred_config_dict)

        # Extract other configurations
        roi_config = config.get("roi_classifier", {})
        detection_config = config.get("detection", {})
        feature_extractor_config = config.get("feature_extractor", {})
        mlflow_config = config.get("mlflow", {})
        label_studio_config = config.get("label_studio", {})
        system_config = config.get("system", {})

        # Call the original load_engine method with extracted parameters
        return cls.load_engine(
            pred_config=pred_config,
            roi_classifier_path=roi_config.get("path"),
            roi_cls_is_features=roi_config.get("is_features", True),
            roi_cls_label_map=roi_config.get("label_map", {0: "gt", 1: "tn"}),
            roi_keep_classes=roi_config.get("keep_classes", ["gt"]),
            detection_label_map=detection_config.get("label_map", {0: "wildlife"}),
            feature_extractor_path=feature_extractor_config.get(
                "path", "facebook/dinov2-with-registers-small"
            ),
            model_path=detection_config.get("model_path"),
            detection_model_type=detection_config.get("model_type", "ultralytics"),
            text_instruction=detection_config.get(
                "text_instruction", "detect wildlife species"
            ),
            mlflow_model_alias=mlflow_config.get("model_alias", "demo"),
            mlflow_model_name=mlflow_config.get("model_name", "labeler"),
            set_ls_client=label_studio_config.get("set_client", False),
            buffer_size=system_config.get("buffer_size", 24),
            timeout=system_config.get("timeout", 60),
            dot_env_path=label_studio_config.get("dot_env_path"),
        )
