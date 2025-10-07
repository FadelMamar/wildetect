import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from pathlib import Path

import cv2
import geopy
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms

from ..config import FlightSpecs
from ..gps.geographic_bounds import GeographicBounds
from ..gps.gps_service import GPSDetectionService, create_geographic_footprint
from ..gps.gps_utils import GPSUtils, get_gsd
from .detection import Detection
from .utils import get_image_dimensions, read_image

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Class representing an image tile."""

    image_path: Optional[str] = None
    image_data: Optional[Image.Image] = None

    id: Optional[str] = None

    width: Optional[int] = None
    height: Optional[int] = None

    x_offset: Optional[int] = None
    y_offset: Optional[int] = None

    parent_image: Optional[str] = None
    timestamp: Optional[str] = None

    tile_gps_loc: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    flight_specs: Optional["FlightSpecs"] = None

    geographic_footprint: Optional["GeographicBounds"] = None
    gsd: Optional[float] = None  # cm/px

    predictions: List[Detection] = field(default_factory=list)
    annotations: List[Detection] = field(default_factory=list)

    _pred_is_original: bool = False
    _annot_is_original: bool = False

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

        if self.image_data is not None:
            self.image_data = ImageOps.exif_transpose(self.image_data)

        if self.parent_image:
            try:
                self.timestamp = Image.open(self.parent_image)._getexif()[36867]
            except:
                pass

        elif self.image_path:
            try:
                self.timestamp = Image.open(self.image_path)._getexif()[36867]
            except:
                pass

        # Only open image if dimensions are not provided
        if self.width is None or self.height is None:
            if self.image_data is None:
                self.width, self.height = get_image_dimensions(self.image_path)
            else:
                self.width, self.height = self.image_data.size

        # GPS operations >>>>>
        if self.image_path is None and self.image_data is None:
            return None
        
        if Path(self.image_path).suffix.lower() == ".tif":
            logger.warning(f"Skipping GPS extraction for {self.image_path} as it is a TIFF file.")
            return None
            
        # GPS extraction
        try:
            self._extract_gps_coords()
            exif = self._extract_exif()

            if self.flight_specs is None:
                logger.warning("Flight specs are not provided.")
                return
            elif isinstance(self.flight_specs, FlightSpecs):
                pass
            else:
                raise ValueError(
                    f"Flight specs is either None or not a 'FlightSpecs' object. Found {type(self.flight_specs)}"
                )

            sensor_height = self.flight_specs.sensor_height
            if sensor_height is None:
                sensor_height = GPSUtils.SENSOR_HEIGHTS.get(exif["Model"])
                if sensor_height is None:
                    logger.debug("Sensor height not found. Please provide it.")

            # self.gsd = self.flight_specs.gsd
            if self.flight_specs is not None:
                self.gsd = get_gsd(
                    image_path=self.image_path,
                    image=self.image_data,
                    flight_specs=self.flight_specs,
                )
            try:
                self._set_geographic_footprint()
            except Exception as e:
                logger.warning(
                    f"Failed to set geographic footprint for {self.image_path}: {traceback.format_exc()}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize tile: {traceback.format_exc()}")

        return None

    def load_image_data(self) -> Image.Image:
        if self.image_data is not None:
            return self.image_data
        else:
            return read_image(self.image_path)

    @property
    def geo_box(self):
        if self.geographic_footprint is not None:
            return self.geographic_footprint.box
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(vars(self))
        d.pop("image_data")
        if self.geographic_footprint is not None:
            d["geographic_footprint"] = self.geographic_footprint.to_dict()
        d["geo_box"] = self.geo_box
        d["type"] = "Tile"
        d["flight_specs"] = vars(self.flight_specs)
        d["predictions"] = [det.to_dict() for det in self.predictions]
        d["annotations"] = [det.to_dict() for det in self.annotations]
        return d

    def geo_iou(self, other: "Tile") -> float:
        return self.geographic_footprint.overlap_ratio(other.geographic_footprint)

    def _extract_exif(self):
        exif = GPSUtils.get_exif(file_name=self.image_path, image=self.image_data)
        return exif

    def _set_geographic_footprint(self):
        self.geographic_footprint = create_geographic_footprint(
            x1=0,
            x2=self.width,
            y1=0,
            y2=self.height,
            lat_center_roi=self.latitude,
            long_center_roi=self.longitude,
            width_roi=self.width,
            height_roi=self.height,
            gsd=self.gsd,
        )

    def _extract_gps_coords(
        self,
    ) -> None:
        # assert self.image_path is not None, "Provide image_path field when defining a tile"
        image = None
        if self.image_path is None:
            image = self.image_data
            if image is None:
                raise ValueError(
                    "Image data is None. Please provide image_path or image_data."
                )

        coords = GPSUtils.get_gps_coord(
            file_name=self.image_path,
            image=image,
            return_as_decimal=True,
        )
        if coords is not None:
            self.latitude, self.longitude, self.altitude = coords[0]
            self.tile_gps_loc = str(
                geopy.Point(self.latitude, self.longitude, self.altitude / 1e3)
            )
        else:
            self.latitude, self.longitude, self.altitude = None, None, None
            logger.debug(f"Failed to extract GPS coordinates from {self.image_path}.")

        return None

    def set_offsets(self, x_offset: int, y_offset: int):
        self.y_offset = y_offset
        self.x_offset = x_offset

    def offset_detections(
        self,
    ):
        if self.x_offset is not None and self.y_offset is not None:
            if self._pred_is_original:
                logger.debug(
                    "Skipping - Predictions have already been mapped to the reference coordinates."
                )
            if self.predictions and (not self._pred_is_original):
                for det in self.predictions:
                    if det.is_empty:
                        continue
                    det.to_absolute_coords(self.x_offset, self.y_offset)
                self._pred_is_original = True

            if self.annotations and (not self._annot_is_original):
                if self._annot_is_original:
                    logger.debug(
                        "Skipping - Annotations have already been mapped to the reference coordinates."
                    )
                for det in self.annotations:
                    if det.is_empty:
                        continue
                    det.to_absolute_coords(self.x_offset, self.y_offset)
                self._annot_is_original = True
        else:
            logger.error("Failed...self.x_offset is None or self.y_offset is not None.")

    def _nms(self, threshold: float = 0.5):
        if len(self.predictions) < 2:
            return self.predictions

        bboxs = torch.Tensor([det.get_bbox() for det in self.predictions])
        scores = torch.Tensor([det.confidence for det in self.predictions])

        # get indices of examples to keep
        indx = nms(boxes=bboxs, scores=scores, iou_threshold=threshold)

        return [self.predictions[i] for i in indx.tolist()]

    def filter_detections(
        self,
        method: str = "nms",
        threshold: float = 0.5,
        clamp: bool = True,
        confidence_threshold: float = 0.0,
    ):
        assert method == "nms", "only nms is supported"

        if len(self.predictions) < 1:
            return

        if confidence_threshold > 0.0:
            self.predictions = [
                det
                for det in self.predictions
                if det.confidence >= confidence_threshold
            ]

        if clamp:
            for det in self.predictions:
                det.clamp_bbox(x_range=(0, self.width), y_range=(0, self.height))

        self.predictions = self._nms(threshold)

        return None

    def update_detection_gps(
        self, detection_type: Literal["predictions", "annotations"]
    ):
        """Update GPS information for all detections in this tile."""
        GPSDetectionService.update_detections_by_type(self, detection_type)

    def set_predictions(self, data: List[Detection], update_gps: bool = True) -> None:
        """Set predictions with proper validation."""
        if not isinstance(data, list):
            raise TypeError(f"Expected 'list' but received {type(data)}")

        if data:
            for det in data:
                if not isinstance(det, Detection):
                    raise TypeError(f"Expected Detection object, got {type(det)}")
                det.update_values_from_tile(self)
            self.predictions = data
        else:
            self.predictions = [Detection.empty(parent_image=self.image_path)]

        self.validate_detections(predictions=True, annotations=False)
        if update_gps:
            self.update_detection_gps(detection_type="predictions")

    def set_annotations(self, data: List[Detection], update_gps: bool = True) -> None:
        """Set annotations with proper validation."""
        if not isinstance(data, list):
            raise TypeError(f"Expected 'list' but received {type(data)}")

        if data:
            for det in data:
                if not isinstance(det, Detection):
                    raise TypeError(f"Expected Detection object, got {type(det)}")
                det.update_values_from_tile(self)
            self.annotations = data
        else:
            self.annotations = [Detection.empty(parent_image=self.image_path)]

        self.validate_detections(predictions=False, annotations=True)
        if update_gps:
            self.update_detection_gps(detection_type="annotations")

    def remove_predictions(self, indices: List[int]) -> None:
        """Remove predictions at specified indices."""
        for i in indices:
            self.predictions.pop(i)

    def add_detection(self, detection: Detection, is_annotation: bool = False) -> None:
        """Add a single detection to the tile."""
        if not isinstance(detection, Detection):
            raise TypeError(f"Expected Detection object, got {type(detection)}")

        self.validate_detection(detection)

        detection.update_values_from_tile(self)

        if is_annotation:
            if self.annotations is None:
                self.annotations = []
            self.annotations.append(detection)
        else:
            if self.predictions is None:
                self.predictions = []
            self.predictions.append(detection)

    def validate_detection(self, detection: Detection) -> None:
        """Validate a single detection."""
        if not detection.is_empty:
            if detection.x_center < 0 or detection.x_center >= self.width:
                raise ValueError(f"Detection: x_center out of bounds (0, {self.width})")
            if detection.y_center < 0 or detection.y_center >= self.height:
                raise ValueError(
                    f"Detection : y_center out of bounds (0, {self.height})"
                )

    def validate_detections(
        self, predictions: bool = True, annotations: bool = True
    ) -> None:
        """Validate all detections in this tile."""
        errors = []

        preds = getattr(self, "predictions", [])
        anns = getattr(self, "annotations", [])

        for i, det in enumerate(preds if predictions else []):
            if not det.is_empty:
                if det.x_center < 0 or det.x_center >= self.width:
                    errors.append(
                        f"Detection {i}: x_center {det.x_center} out of bounds{(0, self.width)}"
                    )
                if det.y_center < 0 or det.y_center >= self.height:
                    errors.append(
                        f"Detection {i}: y_center {det.y_center} out of bounds{(0, self.height)}"
                    )

        for i, det in enumerate(anns if annotations else []):
            if not det.is_empty:
                if det.x_center < 0 or det.x_center >= self.width:
                    errors.append(
                        f"Annotation {i}: x_center {det.x_center} out of bounds{(0, self.width)}"
                    )
                if det.y_center < 0 or det.y_center >= self.height:
                    errors.append(
                        f"Annotation {i}: y_center {det.y_center} out of bounds{(0, self.height)}"
                    )

        if len(errors) > 0:
            raise ValueError(f"Validation errors: {errors}")

        return None

    @classmethod
    def from_image_path(cls, image_path: str, **kwargs) -> "Tile":
        """Create tile from image path."""
        return cls(image_path=image_path, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tile":
        """Create tile from dictionary."""

        data = {k: v for k, v in data.items() if k in vars(cls)}
        if "image_path" not in data:
            data["image_path"] = None

        if "flight_specs" in data:
            flight_specs = data.pop("flight_specs")
            if isinstance(flight_specs, dict):
                data["flight_specs"] = FlightSpecs(**flight_specs)
            elif isinstance(flight_specs, FlightSpecs):
                data["flight_specs"] = flight_specs
            else:
                logger.error(
                    f"Invalid flight specs type: {type(flight_specs)}.Skipping..."
                )

        if "geographic_footprint" in data:
            geographic_footprint = data.pop("geographic_footprint")
            if isinstance(geographic_footprint, dict):
                data["geographic_footprint"] = GeographicBounds.from_dict(
                    geographic_footprint
                )
            elif isinstance(geographic_footprint, GeographicBounds):
                data["geographic_footprint"] = geographic_footprint
            else:
                logger.error(
                    f"Invalid geographic footprint type: {type(geographic_footprint)}.Skipping..."
                )

        if "predictions" in data:
            data["predictions"] = [
                Detection.from_dict(det) for det in data["predictions"]
            ]
        if "annotations" in data:
            data["annotations"] = [
                Detection.from_dict(det) for det in data["annotations"]
            ]

        # print(data)

        return cls(**data)

    @classmethod
    def from_image_data(cls, image_data: Image.Image, **kwargs) -> "Tile":
        """Create tile from image data."""
        return cls(image_data=image_data, image_path=None, **kwargs)

    def draw_detections(
        self,
        predictions: bool = True,
        annotations: bool = True,
        prediction_color: Tuple[int, int, int] = (0, 255, 0),  # Green for predictions
        annotation_color: Tuple[int, int, int] = (255, 0, 0),  # Red for annotations
        line_thickness: int = 2,
        font_scale: float = 0.5,
        show_confidence: bool = True,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw detection bounding boxes on an image.

        Args:
            image: Input image as numpy array. If None, uses tile's image data
            predictions: Whether to draw predictions
            annotations: Whether to draw annotations
            prediction_color: BGR color for prediction boxes (default: green)
            annotation_color: BGR color for annotation boxes (default: red)
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of the font for labels
            show_confidence: Whether to show confidence scores in labels
            save_path: Optional path to save the resulting image

        Returns:
            Image with detections drawn
        """

        # Load image if not provided
        pil_image = self.load_image_data()
        # Convert PIL image to numpy array (BGR format for OpenCV)
        image = np.array(pil_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = image.copy()

        # Draw predictions
        if predictions and self.predictions:
            for detection in self.predictions:
                if not detection.is_empty:
                    bbox = detection.bbox
                    class_name = detection.class_name
                    confidence = detection.confidence

                    # Draw bounding box
                    cv2.rectangle(
                        result,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        prediction_color,
                        line_thickness,
                    )

                    # Draw label
                    if show_confidence:
                        label = f"{class_name}: {confidence:.2f}"
                    else:
                        label = class_name

                    # Calculate text position
                    text_x = bbox[0]
                    text_y = bbox[1] - 10 if bbox[1] - 10 > 0 else bbox[1] + 20

                    # Draw text background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
                    )
                    cv2.rectangle(
                        result,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width, text_y + 5),
                        prediction_color,
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        result,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        line_thickness,
                    )

        # Draw annotations
        if annotations and self.annotations:
            for detection in self.annotations:
                if not detection.is_empty:
                    bbox = detection.bbox
                    class_name = detection.class_name
                    confidence = detection.confidence

                    # Draw bounding box
                    cv2.rectangle(
                        result,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        annotation_color,
                        line_thickness,
                    )

                    # Draw label
                    if show_confidence:
                        label = f"{class_name}: {confidence:.2f}"
                    else:
                        label = class_name

                    # Calculate text position
                    text_x = bbox[0]
                    text_y = bbox[1] - 10 if bbox[1] - 10 > 0 else bbox[1] + 20

                    # Draw text background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
                    )
                    cv2.rectangle(
                        result,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width, text_y + 5),
                        annotation_color,
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        result,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        line_thickness,
                    )

        # Save image if path is provided
        if save_path:
            cv2.imwrite(save_path, result)
            logger.info(f"Image with detections saved to: {save_path}")

        return result

    def draw_detections_with_legend(
        self,
        predictions: bool = True,
        annotations: bool = True,
        prediction_color: Tuple[int, int, int] = (0, 255, 0),
        annotation_color: Tuple[int, int, int] = (255, 0, 0),
        line_thickness: int = 2,
        font_scale: float = 0.5,
        show_confidence: bool = True,
        save_path: Optional[str] = None,
        legend_position: str = "top-right",
    ) -> np.ndarray:
        """Draw detections with a legend showing what each color represents."""

        # Draw the base image with detections
        result = self.draw_detections(
            predictions=predictions,
            annotations=annotations,
            prediction_color=prediction_color,
            annotation_color=annotation_color,
            line_thickness=line_thickness,
            font_scale=font_scale,
            show_confidence=show_confidence,
        )

        # Add legend
        legend_items = []
        if predictions and self.predictions:
            legend_items.append(("Predictions", prediction_color))
        if annotations and self.annotations:
            legend_items.append(("Annotations", annotation_color))

        if not legend_items:
            return result

        # Calculate legend position
        img_height, img_width = result.shape[:2]
        legend_width = 200
        legend_height = len(legend_items) * 30 + 20
        legend_x = 10
        legend_y = 10

        if legend_position == "top-right":
            legend_x = img_width - legend_width - 10
        elif legend_position == "bottom-left":
            legend_y = img_height - legend_height - 10
        elif legend_position == "bottom-right":
            legend_x = img_width - legend_width - 10
            legend_y = img_height - legend_height - 10

        # Draw legend background
        cv2.rectangle(
            result,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            result,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (255, 255, 255),
            2,
        )

        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 30

            # Draw color box
            cv2.rectangle(
                result,
                (legend_x + 10, y_pos - 10),
                (legend_x + 30, y_pos + 10),
                color,
                -1,
            )

            # Draw label
            cv2.putText(
                result,
                label,
                (legend_x + 40, y_pos + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        # Save image if path is provided
        if save_path:
            cv2.imwrite(save_path, result)
            logger.info(f"Image with detections and legend saved to: {save_path}")

        return result

    def draw_detections_to_pil(
        self,
        predictions: bool = True,
        annotations: bool = True,
        prediction_color: Tuple[int, int, int] = (0, 255, 0),
        annotation_color: Tuple[int, int, int] = (255, 0, 0),
        line_thickness: int = 2,
        font_scale: float = 0.5,
        show_confidence: bool = True,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """Draw detections and return as PIL Image."""

        # Draw detections
        result = self.draw_detections(
            predictions=predictions,
            annotations=annotations,
            prediction_color=prediction_color,
            annotation_color=annotation_color,
            line_thickness=line_thickness,
            font_scale=font_scale,
            show_confidence=show_confidence,
            save_path=save_path,
        )

        # Convert BGR to RGB for PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(result_rgb)

        return pil_image
