import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import albumentations as A
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from ..data.detection import Detection


def get_processor(name: str) -> type:
    """Return the processor class based on the given name. 'feature_extractor', 'classifier', or 'roi_post'"""
    processor_map = {
        "feature_extractor": FeatureExtractor,
        "classifier": Classifier,
        "roi_post": RoIPostProcessor,
    }

    if name not in processor_map:
        raise NotImplementedError(
            f"Processor '{name}' is not implemented. Available: {list(processor_map.keys())}"
        )

    return processor_map[name]


class Processor(ABC):
    """Abstract base class for all processors."""

    def __init__(
        self,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the processor on the given arguments."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        pass


def check_images_sequences(images: Sequence[Image.Image]) -> None:
    """Assert that the input is a sequence of PIL Images.

    Args:
        images (Sequence[Image.Image]): Sequence of PIL Images to validate.

    Raises:
        AssertionError: If images is not a sequence or contains non-PIL Image objects.
    """
    if not isinstance(images, Sequence):
        raise AssertionError(f"Expected Sequence, got {type(images)}")

    errors = []
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            errors.append(f"Image at index {i} is not a PIL Image, got {type(img)}")

    if errors:
        raise AssertionError(f"Invalid images: {errors}")


class FeatureExtractor(Processor):
    """Feature extractor using a HuggingFace model."""

    def __init__(
        self,
        hf_model_path: str = "facebook/dinov2-with-registers-small",
    ):
        """Initialize the feature extractor with a HuggingFace model path.

        Args:
            hf_model_path (str): Path or name of the HuggingFace model.
        """
        super().__init__()

        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_model_path)
            self.extractor = AutoModel.from_pretrained(hf_model_path, device_map="auto")
            self.device = self.extractor.device
            self.logger.info(
                f"Feature extractor initialized with model: {hf_model_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {hf_model_path}: {e}")
            raise

    @torch.no_grad()
    def run(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        """Extract features from a sequence of images."""
        if not images:
            raise ValueError("Images sequence cannot be empty")

        try:
            # Convert numpy arrays to PIL Images

            # Process in batches if configured
            all_features = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]

                inputs = self.processor(images=batch_images, return_tensors="pt").to(
                    self.device
                )

                outputs = self.extractor(**inputs)
                features = (
                    outputs.pooler_output.cpu().reshape(len(batch_images), -1).numpy()
                )
                all_features.append(features)

            return np.vstack(all_features)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, "extractor"):
            del self.extractor
        if hasattr(self, "processor"):
            del self.processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class Classifier(Processor):
    """Image classifier using a PyTorch model."""

    def __init__(
        self,
        model_path: str,
        label_map: Dict[int, str],
        feature_extractor_path: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        device: str = "cpu",
    ):
        """Initialize the classifier.

        Args:
            model (torch.nn.Module): PyTorch model for classification.
            label_map (Dict[int, str]): Mapping from class indices to class names.
            feature_extractor_path (Optional[str]): Path to feature extractor model.
            imgsz (int): Image size for preprocessing.
            transform (Optional[A.Compose]): Albumentations transform for preprocessing.
        """
        super().__init__()

        if not isinstance(label_map, dict):
            raise ValueError("label_map must be a dictionary")

        self.model = torch.jit.load(model_path)
        self.label_map = label_map
        if feature_extractor_path:
            self.feature_extractor = FeatureExtractor(feature_extractor_path)
        else:
            self.feature_extractor = None

        # Move model to device
        self.device = device
        self.model = self.model.to(device)
        try:
            self.model.eval()
        except Exception:
            pass

        # Setup transform
        self.transform = transform

        self.logger.info(f"Classifier initialized with {len(label_map)} classes")

    def _pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to a numpy array.

        Args:
            image (Image.Image): PIL Image to convert.

        Returns:
            np.ndarray: Numpy array representation.
        """
        image = image.convert("RGB")
        return np.asarray(image)

    @torch.no_grad()
    def run(self, images: List[Image.Image], batch_size: int = 8) -> List[str]:
        """Classify a sequence of images.

        Args:
            images (Sequence[Image.Image]): List of PIL Images.

        Returns:
            List[str]: List of predicted class names.

        Raises:
            ValueError: If images is empty or invalid.
        """
        if not images:
            raise ValueError("Images sequence cannot be empty")

        check_images_sequences(images)

        try:
            # Preprocess images
            predictions = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : min(i + batch_size, len(images))]
                preprocessed = []
                for image in batch_images:
                    if self.transform is not None:
                        result = self.transform(image=self._pil_to_numpy(image))
                        preprocessed.append(result["image"])
                    else:
                        preprocessed.append(self._pil_to_numpy(image))

                # Use feature extractor if available
                if self.feature_extractor:
                    preprocessed = self.feature_extractor.run(preprocessed)
                    preprocessed = torch.Tensor(preprocessed).to(self.device)
                else:
                    # Convert to tensor format expected by the model
                    preprocessed = torch.stack(
                        [
                            torch.from_numpy(img.transpose(2, 0, 1)).float()
                            for img in preprocessed
                        ]
                    ).to(self.device)

                # Run inference
                probs = self.model(preprocessed).softmax(1)
                pred = probs.argmax(1).cpu().long().flatten().tolist()

                # Map predictions to class names
                predictions.extend(list(map(lambda x: self.label_map[x], pred)))

            return predictions

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            raise

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class RoIPostProcessor(Processor):
    """Post-process detections using a classifier to filter by class."""

    def __init__(
        self,
        model_path: str,
        label_map: Dict[int, str],
        feature_extractor_path: Optional[str] = None,
        roi_size: int = 96,
        transform: Optional[A.Compose] = None,
        device: str = "cpu",
        classifier: Optional[Classifier] = None,
        keep_classes: Optional[List[str]] = None,
    ):
        """Initialize the postprocessor.

        Args:
            keep_classes (Optional[List[str]]): List of class names to keep. Defaults to ["groundtruth"].
            config (Optional[ProcessorConfig]): Processor configuration.
        """
        super().__init__()
        self.box_size = roi_size

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = Classifier(
                model_path, label_map, feature_extractor_path, transform, device
            )
        self.keep = keep_classes or ["groundtruth"]
        self.logger.info(
            f"DetectionsPostprocessor initialized to keep classes: {self.keep}"
        )

    def run(
        self,
        detections: List[Detection],
        image: Image.Image,
        verbose: bool = False,
    ) -> List[Detection]:
        """Filter detections by running a classifier on cropped image regions.

        Args:
            detections (List[Detection]): List of detection objects.
            image (Image.Image): Source image.
            box_size (int): Size of the crop around each detection.
            verbose (Optional[bool]): Whether to show progress bar. Uses config if None.

        Returns:
            List[Detection]: Filtered detections.

        Raises:
            ValueError: If classifier is not set or image is invalid.
            AssertionError: If image is not a PIL Image.
        """
        if not isinstance(image, Image.Image):
            raise AssertionError("image must be a PIL Image")

        if not detections:
            return []

        self.logger.debug(f"Filtering {len(detections)} detections...")

        try:
            image = image.convert("RGB")
            img_width, img_height = image.size

            # Extract crops from detections
            crops = []
            valid_detections = []

            for det in detections:
                if not hasattr(det, "x_center") or not hasattr(det, "y_center"):
                    self.logger.warning(
                        f"Detection missing x_center/y_center coordinates: {det}"
                    )
                    continue

                x_center, y_center = det.x_center, det.y_center

                # Calculate crop bounds
                x1 = int(max(x_center - self.box_size // 2, 0))
                y1 = int(max(y_center - self.box_size // 2, 0))
                x2 = int(min(x_center + self.box_size // 2, img_width))
                y2 = int(min(y_center + self.box_size // 2, img_height))

                # Ensure minimum crop size
                if x2 - x1 < 10 or y2 - y1 < 10:
                    self.logger.warning(
                        f"Detection crop too small: {x1},{y1},{x2},{y2}"
                    )
                    continue

                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
                valid_detections.append(det)

            if not crops:
                self.logger.warning("No valid crops extracted from detections")
                return []

            # Classify crops
            if verbose:
                loader = tqdm(crops, desc="ROI based filtering")
                preds = self.classifier.run(list(loader))
            else:
                preds = self.classifier.run(crops)

            # Filter detections based on predictions
            filtered_detections = [
                det for det, pred in zip(valid_detections, preds) if pred in self.keep
            ]

            self.logger.info(
                f"Filtered {len(detections)} -> {len(filtered_detections)} detections"
            )
            return filtered_detections

        except Exception as e:
            self.logger.error(f"Detection postprocessing failed: {e}")
            raise
