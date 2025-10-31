import json
import logging
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from tqdm import tqdm

from ..config import ROIConfig
from .base_adapter import BaseAdapter
from .utils import read_image

logger = logging.getLogger(__name__)


def extract_roi_from_image_bbox(
    image_path, bbox, roi_box_size=128, min_roi_size=32
) -> Optional[Image.Image]:
    """
    Utility to extract a ROI from an image given a bbox, with padding and resizing.
    Args:
        image_path: Path to the image file
        bbox: [x, y, w, h]
        roi_box_size: Output size (int)
        min_roi_size: Minimum ROI size (int)
        padding: Not used (for future extension)
    Returns:
        PIL.Image or None
    """
    img = read_image(image_path)
    x, y, w, h = bbox
    pad_x = roi_box_size // 2
    pad_y = roi_box_size // 2
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = int(x + w + pad_x)
    y2 = int(y + h + pad_y)

    if (x2 - x1) < min_roi_size or (y2 - y1) < min_roi_size:
        logger.warning(f"ROI size is too small: {x2 - x1}x{y2 - y1}")
        return None

    roi = img.crop((x1, y1, x2, y2))
    roi = roi.resize((roi_box_size, roi_box_size))

    return roi


class ROIAdapter(BaseAdapter):
    """
    Adapter for converting COCO annotation format to ROI classification format.

    This adapter extracts bounding boxes from object detection annotations
    and converts them into individual classification images with JSON labels.
    """

    def __init__(
        self,
        coco_data: Dict[str, Any],
        roi_callback: Optional[Callable] = None,
        random_roi_count: int = 1,
        roi_box_size: int = 128,
        min_roi_size: int = 32,
        sample_background: bool = True,
        background_class: str = "background",
        save_format: str = "jpg",
        quality: int = 95,
        dark_threshold: float = 0.5,
    ):
        """
        Initialize the ROI adapter.

        Args:
            coco_annotation_path: Path to COCO annotation file
            coco_data: COCO data dictionary
            roi_callback: Custom function to generate ROIs for unannotated images
            random_roi_count: Number of random ROIs to generate for unannotated images
            roi_padding: Extra padding around bounding boxes (as fraction of bbox size)
            min_roi_size: Minimum ROI size in pixels
            background_class: Class name for random ROIs
            save_format: Image format for ROI crops
            quality: JPEG quality (1-100)
        """
        super().__init__(coco_data)

        self.roi_callback = roi_callback
        self.random_roi_count = random_roi_count
        self.roi_box_size = roi_box_size
        self.min_roi_size = min_roi_size
        self.background_class = background_class
        self.save_format = save_format
        self.quality = quality
        self.sample_background = sample_background
        self.pad_roi = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=roi_box_size,
                    min_width=roi_box_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                ),
                A.CenterCrop(
                    height=roi_box_size,
                    width=roi_box_size,
                    p=1,
                ),
            ]
        )
        self.dark_threshold = dark_threshold

        self.class_mapping = self.create_class_mapping()

    @classmethod
    def from_config(cls, config: ROIConfig, coco_data: Dict[str, Any]):
        """
        Initialize the adapter from a configuration object.
        Args:
            config (ROIConfig): Configuration object with ROI settings.
            coco_data (Dict[str, Any]): COCO data dictionary.
        """
        assert isinstance(config, ROIConfig), "config must be an instance of ROIConfig"
        return cls(
            coco_data=coco_data,
            random_roi_count=config.random_roi_count,
            roi_box_size=config.roi_box_size,
            min_roi_size=config.min_roi_size,
            background_class=config.background_class,
            save_format=config.save_format,
            quality=config.quality,
            sample_background=config.sample_background,
            dark_threshold=config.dark_threshold,
        )

    def create_class_mapping(
        self,
    ):
        categories = self.coco_data.get("categories", [])
        # Create class mapping
        class_mapping = {cat["id"]: cat["name"] for cat in categories}
        class_mapping[max(class_mapping.keys()) + 1] = self.background_class
        return class_mapping

    def convert(self) -> Dict[str, Any]:
        """
        Convert the loaded COCO annotation to ROI format for the specified split.

        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').

        Returns:
            Dict[str, Any]: ROI data containing:
                - 'roi_images': List of ROI image information
                - 'roi_labels': List of ROI label information
                - 'class_mapping': Mapping of class IDs to names
                - 'statistics': Processing statistics
        """
        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])

        # Group annotations by image
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        roi_images = []
        roi_labels = []
        roi_counter = 0
        statistics = {
            "total_images": len(images),
            "annotated_images": 0,
            "unannotated_images": 0,
            "total_rois": 0,
            "rois_from_annotations": 0,
            "rois_from_callback": 0,
            "rois_from_random": 0,
        }

        for image in tqdm(images, desc="Mining ROIs"):
            image_id = image["id"]

            # Get annotations for this image
            image_annotations = annotations_by_image.get(image_id, [])

            if image_annotations:
                # Extract ROIs from annotations
                statistics["annotated_images"] += 1
                image_rois = self._extract_rois_from_annotations(
                    image_annotations, image, roi_counter
                )
                roi_images.extend(image_rois["roi_images"])
                roi_labels.extend(image_rois["roi_labels"])
                roi_counter = image_rois["next_counter"]
                statistics["rois_from_annotations"] += len(image_rois["roi_images"])
                statistics["total_rois"] += len(image_rois["roi_images"])
            elif self.sample_background:
                # Generate ROIs for unannotated image
                statistics["unannotated_images"] += 1
                image_rois = self._generate_rois_for_unannotated_image(
                    image, roi_counter
                )
                roi_images.extend(image_rois["roi_images"])
                roi_labels.extend(image_rois["roi_labels"])
                roi_counter = image_rois["next_counter"]
                statistics["rois_from_callback"] += image_rois["callback_rois"]
                statistics["rois_from_random"] += image_rois["random_rois"]
                statistics["total_rois"] += len(image_rois["roi_images"])

        print(statistics)

        return {
            "roi_images": roi_images,
            "roi_labels": roi_labels,
            "class_mapping": self.class_mapping,
            "statistics": statistics,
        }

    def save(
        self,
        roi_data: Dict[str, Any],
        output_labels_dir: Union[str, Path],
        output_images_dir: Union[str, Path],
        draw_original_bboxes: bool = False,
    ) -> None:
        """
        Save the ROI-formatted data to the output directory.

        Args:
            roi_data (Dict[str, Any]): ROI data from convert method
            output_path (Optional[str]): Directory to save the ROI data
        """
        if not roi_data or not roi_data.get("roi_images"):
            logger.warning("No ROI data to save")
            return

        # Create output directories
        labels_dir = Path(output_labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Save class mapping
        self._save_class_mapping(roi_data["class_mapping"], labels_dir)

        # Save statistics
        self._save_statistics(roi_data["statistics"], labels_dir)

        # Save ROI images
        image_paths = self._save_roi_images(
            roi_data["roi_images"],
            output_images_dir,
            draw_original_bboxes=draw_original_bboxes,
        )

        # Save ROI labels as JSON
        self._save_roi_labels_json(roi_data["roi_labels"], labels_dir, image_paths)

        logger.info(
            f"Saved {len(roi_data['roi_images'])} ROI images to {output_images_dir}"
        )

    def _extract_rois_from_annotations(
        self, annotations: List[Dict], image: Dict, start_counter: int
    ) -> Dict[str, Any]:
        """
        Extract ROIs from existing annotations.

        Args:
            annotations: List of annotations for the image
            image: Image information
            start_counter: Starting counter for ROI IDs

        Returns:
            Dictionary with ROI images, labels, and next counter
        """
        roi_images = []
        roi_labels = []
        counter = start_counter

        image_path = image["file_name"]
        width = image["width"]
        height = image["height"]

        # Load image for cropping
        try:
            img = read_image(image_path)
            width, height = img.size
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            return {"roi_images": [], "roi_labels": [], "next_counter": counter}

        for ann in annotations:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            category_id = ann.get("category_id")

            # Apply padding
            pad_x = self.roi_box_size // 2
            pad_y = self.roi_box_size // 2

            # Calculate padded coordinates
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(width, int(x + w + pad_x))
            y2 = min(height, int(y + h + pad_y))

            # Check minimum size
            if (x2 - x1) < self.min_roi_size or (y2 - y1) < self.min_roi_size:
                continue

            # Generate ROI filename
            roi_filename = (
                f"{Path(image_path).stem}_roi_{counter:06d}.{self.save_format}"
            )

            # Create ROI image info
            roi_image_info = {
                "roi_id": counter,
                "roi_filename": roi_filename,
                "original_image_path": image_path,
                "original_image_id": image["id"],
                "bbox": [x1, y1, x2, y2],
                "original_bbox": [x, y, x + w, y + h],
                "width": x2 - x1,
                "height": y2 - y1,
            }

            # Create ROI label info
            roi_label_info = {
                "roi_id": counter,
                "class_id": category_id,
                "class_name": self._get_class_name(category_id),
                "original_annotation_id": ann.get("id"),
                "file_name": roi_filename,
            }

            roi_images.append(roi_image_info)
            roi_labels.append(roi_label_info)
            counter += 1

        return {
            "roi_images": roi_images,
            "roi_labels": roi_labels,
            "next_counter": counter,
        }

    def _generate_rois_for_unannotated_image(
        self, image: Dict, start_counter: int
    ) -> Dict[str, Any]:
        """
        Generate ROIs for unannotated image using callback or random generation.

        Args:
            image: Image information
            start_counter: Starting counter for ROI IDs

        Returns:
            Dictionary with ROI images, labels, and statistics
        """
        roi_images = []
        roi_labels = []
        counter = start_counter
        callback_rois = 0
        random_rois = 0

        image_path = image["file_name"]

        # Try callback first
        if self.roi_callback:
            try:
                # Load image for callback
                img = read_image(image_path)
                callback_bboxes = self.roi_callback(img)
                for bbox_info in callback_bboxes:
                    bbox = bbox_info.get("bbox")
                    class_name = bbox_info.get("class", self.background_class)

                    if bbox:
                        roi_result = self._create_roi_from_bbox(
                            image, bbox, class_name, counter
                        )
                        if roi_result:
                            roi_images.append(roi_result["roi_image"])
                            roi_labels.append(roi_result["roi_label"])
                            counter = roi_result["next_counter"]
                            callback_rois += 1
            except Exception:
                logger.warning(
                    f"Error in ROI callback for {image_path}: {traceback.format_exc()}"
                )

        # Generate random ROIs if needed
        remaining_rois = self.random_roi_count - callback_rois
        if remaining_rois > 0:
            for _ in range(remaining_rois):
                roi_result = self._create_random_roi(image, counter)
                if roi_result:
                    roi_images.append(roi_result["roi_image"])
                    roi_labels.append(roi_result["roi_label"])
                    counter = roi_result["next_counter"]
                    random_rois += 1

        return {
            "roi_images": roi_images,
            "roi_labels": roi_labels,
            "next_counter": counter,
            "callback_rois": callback_rois,
            "random_rois": random_rois,
        }

    def _create_roi_from_bbox(
        self, image: Dict, bbox: List[int], class_name: str, counter: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create ROI from bounding box coordinates.

        Args:
            image: Image information
            bbox: Bounding box [x, y, w, h]
            class_name: Class name for the ROI
            counter: ROI counter

        Returns:
            Dictionary with ROI image, label, and next counter, or None if failed
        """
        roi = extract_roi_from_image_bbox(
            image["file_name"], bbox, self.roi_box_size, self.min_roi_size
        )
        if roi is None:
            return None
        x, y, w, h = bbox
        width = image["width"]
        height = image["height"]
        pad_x = self.roi_box_size // 2
        pad_y = self.roi_box_size // 2
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + pad_x)
        y2 = min(height, y + pad_y)
        roi_box = list(map(int, [x1, y1, x2, y2]))
        roi_filename = f"{class_name}_roi_{counter:06d}.{self.save_format}"
        roi_image_info = {
            "roi_id": counter,
            "roi_filename": roi_filename,
            "original_image_path": image["file_name"],
            "original_image_id": image["id"],
            "bbox": roi_box,
            "original_bbox": None,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        roi_label_info = {
            "roi_id": counter,
            "class_name": class_name,
            "class_id": self._get_class_id(class_name),
            "file_name": roi_filename,
        }
        return {
            "roi_image": roi_image_info,
            "roi_label": roi_label_info,
            "next_counter": counter + 1,
        }

    def _create_random_roi(
        self, image: Dict, counter: int, max_attempts: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Create a random ROI for unannotated image.

        Args:
            image: Image information
            counter: ROI counter

        Returns:
            Dictionary with ROI image, label, and next counter, or None if failed
        """
        width = image["width"]
        height = image["height"]

        w = self.roi_box_size
        h = self.roi_box_size
        x = random.randint(w // 2, width - w // 2)
        y = random.randint(h // 2, height - h // 2)

        bbox = [x, y, w, h]
        roi_result = None
        for _ in range(max_attempts):
            roi_result = self._create_roi_from_bbox(
                image, bbox, self.background_class, counter
            )
            if roi_result is None:
                continue
            bbox = roi_result["roi_image"]["bbox"]
            path = roi_result["roi_image"]["original_image_path"]
            cropped_img = self.crop_image(path, bbox)
            if not self.is_image_dark(cropped_img):
                return roi_result

        return roi_result

    def is_image_dark(self, image: np.ndarray) -> bool:
        return (
            np.isclose(image, 0.0, atol=10).sum() / image.size
        ) > self.dark_threshold

    def crop_image(
        self,
        image_path: Union[str, Path],
        bbox: Tuple[float, float, float, float],
        original_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        img = read_image(image_path)
        if original_bbox:
            img = sv.TriangleAnnotator().annotate(
                scene=img.copy(),
                detections=sv.Detections(
                    xyxy=np.array(original_bbox).reshape(1, 4),
                    class_id=np.array([0]),
                ),
            )
        cropped_img = img.crop(bbox)
        cropped_img = np.array(cropped_img)
        return cropped_img

    def _save_roi_images(
        self,
        roi_images: List[Dict],
        output_dir: Union[Path, str],
        draw_original_bboxes: bool = False,
    ) -> List[str]:
        """Save ROI images to disk."""

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        def _save_one_image(roi_info: Dict) -> Tuple[bool, str, str]:
            try:
                roi_path = str(output_dir / roi_info["roi_filename"])
                cropped_img = self.crop_image(
                    roi_info["original_image_path"],
                    roi_info["bbox"],
                    roi_info.get("original_bbox") if draw_original_bboxes else None,
                )
                if not self.is_image_dark(cropped_img):
                    if cropped_img.size != (self.roi_box_size, self.roi_box_size):
                        cropped_img = self.pad_roi(image=cropped_img)["image"]
                    Image.fromarray(cropped_img).save(roi_path)
                    return True, "Success", os.path.basename(roi_path)
                else:
                    return False, "Dark image", "None"
            except Exception as e:
                return (
                    False,
                    f"Error saving ROI image {roi_info['roi_filename']}: {e}",
                    "None",
                )

        failures = 0
        errors = []
        paths = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            for success, error_msg, path in tqdm(
                executor.map(_save_one_image, roi_images),
                desc="Saving ROI images",
                total=len(roi_images),
            ):
                if not success:
                    failures += 1
                    errors.append(error_msg)
                else:
                    paths.append(path)
        if failures > 0:
            logger.warning(f"Failed to save {failures}/{len(roi_images)} ROI images")
            logger.warning(f"Reasons: {set(errors)}")
        return paths

    def _save_roi_labels_json(
        self,
        labels_data: List[Dict[str, Any]],
        output_dir: Path,
        save_image_paths: List[str],
    ) -> None:
        """Save ROI labels as JSON file, only for images that have been saved."""
        labels_file = output_dir / "roi_labels.json"

        # Filter labels_data to only include entries whose file_name is in save_image_paths
        filtered_labels = [
            label for label in labels_data if label.get("file_name") in save_image_paths
        ]

        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(filtered_labels, f, indent=2, ensure_ascii=False)

    def _save_class_mapping(
        self, class_mapping: Dict[int, str], output_dir: Path
    ) -> None:
        """Save class mapping to separate file."""
        mapping_file = output_dir / "class_mapping.json"

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    def _save_statistics(self, statistics: Dict[str, Any], output_dir: Path) -> None:
        """Save processing statistics to separate file."""
        stats_file = output_dir / "statistics.json"

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

    def _get_class_name(self, class_id: Optional[int]) -> str:
        """Get class name from class ID."""
        if class_id is None:
            return "unknown"
        categories = self.coco_data.get("categories", [])
        for cat in categories:
            if cat["id"] == class_id:
                return cat["name"]
        return "unknown"

    def _get_class_id(self, class_name: str) -> int:
        """Get class ID from class name."""
        for id_, cat in self.class_mapping.items():
            if cat == class_name:
                return id_
        raise ValueError(f"Class name {class_name} not found in categories")
