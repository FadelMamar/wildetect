import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..converters.labelstudio_converter import LabelstudioConverter
from ..converters.yolo_to_master import YOLOToMasterConverter
from ..logging_config import get_logger
from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator

logger = get_logger(__name__)


class Loader:
    def __init__(self):
        self.split_name = "NOT_SET"
        self.dotenv_path: Optional[str] = None
        self.ls_xml_config: Optional[str] = None
        self.ls_parse_config: bool = False

    def _load_json(self, annotation_path: str) -> Dict[str, Any]:
        with open(annotation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        bbox_tolerance: int,
        split_name: str,
        dotenv_path: Optional[str] = None,
        ls_xml_config: Optional[str] = None,
        ls_parse_config: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split name: {split_name}")

        self.split_name = split_name
        self.dotenv_path = dotenv_path
        self.ls_xml_config = ls_xml_config
        self.ls_parse_config = ls_parse_config

        dataset_info, split_data = self._load_and_validate_dataset(
            source_path=source_path,
            source_format=source_format,
            dataset_name=dataset_name,
            bbox_tolerance=bbox_tolerance,
        )

        return dataset_info, split_data

    def _load_coco_to_split_format(
        self,
        coco_data: Dict[str, Any],
        dataset_name: str,
        image_dir: Path,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load COCO annotation file and convert to split-based format.

        Args:
            coco_annotation_path: Path to COCO annotation file
            dataset_name: Name of the dataset

        Returns:
            Tuple of (dataset_info, split_data)
        """
        # Extract dataset info
        dataset_info = {
            "name": dataset_name,
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": coco_data.get("categories", []),
        }

        # Group images and annotations by split
        split_data = {}
        images_by_split = {}
        annotations_by_split = {}

        # Determine split for each image (simple logic - can be improved)
        warnings = []
        for image in coco_data.get("images", []):
            if not Path(image["file_name"]).is_absolute():
                path = image_dir / self.split_name / Path(image["file_name"]).name
                if not path.exists():
                    msg = f"The expected format {path} does not exist. Skipping image."
                    logger.debug(msg)
                    warnings.append(msg)
                    continue
            else:
                path = image["file_name"]

            image["file_name"] = str(Path(path).resolve())
            if self.split_name not in images_by_split:
                images_by_split[self.split_name] = []
                annotations_by_split[self.split_name] = []
            images_by_split[self.split_name].append(image)

        if len(warnings) > 0:
            logger.warning(
                f"Failed to load {len(warnings)}/{len(coco_data.get('images', []))} images. Set logging level to debug to see."
            )
            logger.info(
                "Make sure the format follows : annotation_file_path/../images/split_name/file_name"
            )
            logger.error("Some warnings:")
            logger.error(warnings[:5])
        # Group annotations by split
        for annotation in coco_data.get("annotations", []):
            image_id = annotation["image_id"]
            # Find which split this annotation belongs to
            for split, images in images_by_split.items():
                if any(img["id"] == image_id for img in images):
                    annotations_by_split[split].append(annotation)
                    break

        # Create split data
        for split in images_by_split.keys():
            split_data[split] = {
                "images": images_by_split[split],
                "annotations": annotations_by_split[split],
                "categories": coco_data.get("categories", []),
            }

        return dataset_info, split_data

    def _check_if_all_images_are_absolute(self, coco_data: Dict[str, Any]) -> bool:
        num_images_is_absolute = 0
        for image in coco_data.get("images", []):
            if Path(image["file_name"]).is_absolute():
                num_images_is_absolute += 1
        return num_images_is_absolute == len(coco_data.get("images", []))

    def _load_and_validate_dataset(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        bbox_tolerance: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load and validate dataset based on source format.

        Returns:
            Tuple of (dataset_info, split_data) or (None, None) if validation fails
        """
        if source_format == "coco":
            # For COCO, source_path should be the annotation file path
            validator = COCOValidator(source_path)
            is_valid, errors, warnings = validator.validate(
                bbox_tolerance=bbox_tolerance
            )
            if not is_valid:
                raise ValueError("COCO validation failed")

            # Load COCO data and convert to split-based structure
            coco_data = self._load_json(source_path)
            image_dir = Path(source_path).parents[1] / "images"

            if len(coco_data.get("images", [])) == 0:
                raise ValueError("No images found in COCO data")

            if not image_dir.exists():
                if not self._check_if_all_images_are_absolute(coco_data):
                    raise ValueError(
                        f"Expected {image_dir} does not exist. Loading will fail as not all image paths are absolute."
                    )

            dataset_info, split_data = self._load_coco_to_split_format(
                coco_data=coco_data,
                dataset_name=dataset_name,
                image_dir=image_dir,
            )

        elif source_format == "yolo":
            # For YOLO, source_path should be the data.yaml file path
            validator = YOLOValidator(source_path)
            is_valid, errors, warnings = validator.validate()
            if not is_valid:
                raise ValueError(f"YOLO validation failed {errors}")

            # Create converter and convert YOLO to COCO format
            converter = YOLOToMasterConverter(source_path)
            dataset_info, split_data = converter.convert(
                dataset_name, task_type="detection", filter_invalid_annotations=False
            )
            split_data = {self.split_name: split_data[self.split_name]}

        elif source_format == "ls":
            converter = LabelstudioConverter(dotenv_path=self.dotenv_path)
            dataset_info, coco_data = converter.convert(
                input_file=source_path,
                dataset_name=dataset_name,
                parse_ls_config=self.ls_parse_config,
                ls_xml_config=self.ls_xml_config,
            )
            split_data = {self.split_name: coco_data}
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        return dataset_info, split_data
