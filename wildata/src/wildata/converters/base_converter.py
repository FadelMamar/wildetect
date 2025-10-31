import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..validators.coco_validator import COCOValidator


class BaseConverter(ABC):
    """
    Abstract base class for annotation format converters.
    Defines the interface for loading, converting, and saving annotations.
    """

    def __init__(
        self,
    ):
        self.logger = logging.getLogger(__name__)
        pass

    @abstractmethod
    def convert(
        self, dataset_name: str
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Convert source format to COCO format with split-based organization.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Tuple of (dataset_info, split_data) where split_data is a dict
            mapping split names to COCO format data
        """
        pass

    def save_coco_annotation(self, coco_data: Dict[str, Any], output_path: str) -> None:
        """
        Save the COCO annotation dictionary to a JSON file.
        Args:
            coco_data (Dict[str, Any]): The COCO annotation data.
            output_path (str): Path to save the output JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

    def _validate_coco_annotation(
        self,
        coco_file_path: Optional[str] = None,
        coco_annotation: Optional[Dict[str, Any]] = None,
        filter_invalid_annotations: bool = False,
    ) -> None:
        """
        Validate the COCO annotation using COCOValidator.
        Args:
            coco_annotation (Dict[str, Any]): The COCO annotation to validate.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Raises:
            ValueError: If validation fails.
        """
        COCOValidator(
            coco_file_path=coco_file_path or "",
            filter_invalid_annotations=filter_invalid_annotations,
            coco_data=coco_annotation,
        ).validate()
