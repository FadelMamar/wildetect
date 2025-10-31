import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class YOLOAdapter(BaseAdapter):
    """
    Adapter for converting COCO annotation format to YOLO format.
    """

    def __init__(
        self,
        coco_data: Dict[str, Any],
    ):
        super().__init__(coco_data)

    def convert(
        self,
    ) -> Dict[str, List[str]]:
        """
        Convert the loaded COCO annotation to YOLO format for the specified split.
        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').
        Returns:
            Dict[str, List[str]]: Mapping from image file names to lists of YOLO label lines.
        """
        images = self.coco_data.get("images", [])
        image_id_to_image = {img["id"]: img for img in images}
        image_labels = {img["file_name"]: [] for img in images}
        annotations = self.coco_data.get("annotations", [])

        for ann in annotations:
            img = image_id_to_image.get(ann["image_id"])
            if img is not None:
                width, height = img["width"], img["height"]
                yolo_line = self._annotation_to_yolo_line(ann, width, height)
                if yolo_line:
                    image_labels[img["file_name"]].append(yolo_line)
        return image_labels

    # --- Private utility methods ---
    def _annotation_to_yolo_line(
        self, ann: Dict[str, Any], width: int, height: int
    ) -> str:
        # YOLO format: class x_center y_center w h (all normalized)
        if "bbox" not in ann or not ann["bbox"]:
            return ""
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height
        class_id = ann["category_id"]

        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
