import csv
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from .base_converter import BaseConverter


def clip_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
    tolerance: int = 5,
    verbose: bool = False,
) -> Tuple[List[float], bool, bool]:
    """
    Clip bounding box coordinates to ensure they stay within image boundaries.

    Args:
        bbox: Bounding box as [x, y, width, height]
        image_width: Width of the image
        image_height: Height of the image
        tolerance: Number of pixels to allow outside boundaries before clipping
        verbose: Whether to print debug information

    Returns:
        Tuple of (clipped_bbox, was_clipped, is_valid) where:
        - clipped_bbox: The clipped bounding box coordinates
        - was_clipped: True if the bbox was clipped (within tolerance but outside boundaries)
        - is_valid: True if the bbox is valid (within tolerance zone)
    """
    if len(bbox) < 4:
        return bbox, False, True

    x, y, w, h = bbox

    # Check if bbox is within tolerance zone (valid)
    is_valid = (
        x >= -tolerance
        and y >= -tolerance
        and x + w <= image_width + tolerance
        and y + h <= image_height + tolerance
    )

    # Check if bbox exceeds boundaries beyond tolerance (invalid)
    is_invalid = (
        x < -tolerance
        or y < -tolerance
        or x + w > image_width + tolerance
        or y + h > image_height + tolerance
    )

    was_clipped = False

    if is_invalid:
        if verbose:
            print(
                f"Invalid bbox [{x}, {y}, {w}, {h}] for image {image_width}x{image_height} (outside tolerance)"
            )
        # For invalid boxes, return original bbox but mark as invalid
        return bbox, False, False

    # Check if bbox needs clipping (within tolerance but outside boundaries)
    if not is_valid:
        was_clipped = True
        if verbose:
            print(
                f"Clipping bbox [{x}, {y}, {w}, {h}] for image {image_width}x{image_height}"
            )

    # Clip coordinates to image boundaries
    x_clipped = max(0, min(x, image_width))
    y_clipped = max(0, min(y, image_height))
    w_clipped = min(w, image_width - x_clipped)
    h_clipped = min(h, image_height - y_clipped)

    # Ensure width and height are positive
    w_clipped = max(0, w_clipped)
    h_clipped = max(0, h_clipped)

    clipped_bbox = [x_clipped, y_clipped, w_clipped, h_clipped]

    clipped_bbox = list(map(math.floor, clipped_bbox))

    if was_clipped and verbose:
        print(f"Clipped bbox to [{x_clipped}, {y_clipped}, {w_clipped}, {h_clipped}]")

    return clipped_bbox, was_clipped, is_valid


class COCOToCSVConverter(BaseConverter):
    """
    Converter from COCO annotation format to CSV format.
    """

    def __init__(self):
        """
        Initialize the COCO to CSV converter.
        """
        super().__init__()

    def convert(
        self,
        coco_file_path: str,
        output_csv_path: str,
        filter_invalid_annotations: bool = False,
        coco_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convert COCO annotation file to CSV format.

        Args:
            coco_file_path: Path to the COCO JSON annotation file
            output_csv_path: Path where the CSV file will be saved
            filter_invalid_annotations: If True, filter out invalid annotations instead of raising errors
            coco_data: Optional pre-loaded COCO data dictionary

        The CSV will contain the following columns:
        - image_id: Unique identifier for the image
        - file_name: Path to the image file
        - width, height: Image dimensions
        - category_id: Category ID (9999 for negative images)
        - category_name: Category name (empty for negative images)
        - bbox_x, bbox_y, bbox_width, bbox_height: Bounding box coordinates (clipped to image boundaries)
        - area: Annotation area
        - annotation_id: Unique annotation ID (empty for negative images)
        - has_annotations: Boolean flag indicating if image has annotations
        """
        if coco_data is None:
            self.logger.info(f"Reading COCO annotation file: {coco_file_path}")
            with open(coco_file_path, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
        else:
            assert isinstance(coco_data, dict), "coco_data must be a dictionary"

        self._validate_coco_annotation(
            coco_file_path=None,
            coco_annotation=coco_data,
            filter_invalid_annotations=filter_invalid_annotations,
        )

        # Extract data
        images = coco_data.get("images", []) if coco_data else []
        annotations = coco_data.get("annotations", []) if coco_data else []
        categories = coco_data.get("categories", []) if coco_data else []

        # Create category mapping
        category_map = {cat["id"]: cat["name"] for cat in categories}

        # Create annotation mapping by image_id
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        # Prepare CSV data
        csv_rows = []
        clipped_count = 0

        for image in images:
            image_id = image["id"]
            image_width = image["width"]
            image_height = image["height"]
            image_annotations = annotations_by_image.get(image_id, [])

            if not image_annotations:
                # Negative image - no annotations
                csv_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "width": image_width,
                        "height": image_height,
                        "category_id": 9999,
                        "category_name": "",
                        "bbox_x": "",
                        "bbox_y": "",
                        "bbox_width": "",
                        "bbox_height": "",
                        "area": "",
                        "annotation_id": "",
                        "has_annotations": False,
                    }
                )
            else:
                # Image with annotations
                tolerance = 5
                for ann in image_annotations:
                    bbox = ann.get("bbox", [])

                    # Clip bounding box if necessary
                    if bbox and len(bbox) >= 4:
                        clipped_bbox, was_clipped, is_valid = clip_bbox(
                            bbox,
                            image_width,
                            image_height,
                            verbose=False,
                            tolerance=tolerance,
                        )
                        if was_clipped:
                            clipped_count += 1
                        if not is_valid:
                            # Skip invalid annotations
                            raise ValueError(
                                f"Invalid bbox [{bbox}] for image {image_width}x{image_height}, with tolerance {tolerance}"
                            )
                        bbox = clipped_bbox

                    # Calculate area from clipped bbox
                    area = bbox[2] * bbox[3] if len(bbox) >= 4 else ann.get("area", "")

                    csv_rows.append(
                        {
                            "image_id": image_id,
                            "file_name": image["file_name"],
                            "width": image_width,
                            "height": image_height,
                            "category_id": ann["category_id"],
                            "category_name": category_map.get(ann["category_id"], ""),
                            "bbox_x": bbox[0] if len(bbox) >= 4 else "",
                            "bbox_y": bbox[1] if len(bbox) >= 4 else "",
                            "bbox_width": bbox[2] if len(bbox) >= 4 else "",
                            "bbox_height": bbox[3] if len(bbox) >= 4 else "",
                            "area": area,
                            "annotation_id": ann["id"],
                            "has_annotations": True,
                        }
                    )

        # Write CSV file
        self.logger.info(f"Writing CSV file: {output_csv_path}")
        fieldnames = [
            "image_id",
            "file_name",
            "width",
            "height",
            "category_id",
            "category_name",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "area",
            "annotation_id",
            "has_annotations",
        ]

        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        # Log summary
        total_images = len(images)
        total_annotations = len(annotations)
        negative_images = sum(
            1 for img in images if img["id"] not in annotations_by_image
        )

        self.logger.info(f"Conversion completed:")
        self.logger.info(f"  - Total images: {total_images}")
        self.logger.info(f"  - Total annotations: {total_annotations}")
        self.logger.info(f"  - Negative images: {negative_images}")
        self.logger.info(f"  - Bounding boxes clipped: {clipped_count}")
        self.logger.info(f"  - CSV rows written: {len(csv_rows)}")
