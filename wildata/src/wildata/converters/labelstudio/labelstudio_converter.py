import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

from ..base_converter import BaseConverter
from .labelstudio_parser import LabelStudioParser


class LabelstudioConverter(BaseConverter):
    """
    Converter for Label Studio JSON annotations to COCO format.
    Inherits from BaseConverter and implements the required interface.

    COCO conversion uses :class:`LabelStudioParser` and supports
    ``rectanglelabels`` (bbox) exports. Optional ``ls_xml_config`` supplies
    category IDs (XML ``<Label value=\"...\"/>`` order, 1-based) for COCO
    ``categories``; otherwise IDs follow sorted labels in the export.
    """

    def __init__(
        self, dotenv_path: Optional[str] = None, client: Optional[LabelStudio] = None
    ):
        """
        Initialize the Label Studio converter.

        Args:
            label_studio_url: Label Studio server URL (optional, can be set via env var)
            api_key: Label Studio API key (optional, can be set via env var)
        """
        super().__init__()

        self.dotenv_path = dotenv_path

        if dotenv_path is not None:
            load_dotenv(dotenv_path)

        label_studio_url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        try:
            if client is None:
                self.ls_client = LabelStudio(base_url=label_studio_url, api_key=api_key)
            else:
                self.ls_client = client
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Label Studio client: {e}. "
                "Ensure LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY are set correctly."
            )
            self.ls_client = None

    def convert(
        self,
        input_file: str,
        dataset_name: str,
        ls_xml_config: Optional[str] = None,
        parse_ls_config: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert Label Studio JSON annotations to COCO format.
        """
        category_mapping: Optional[Dict[str, int]] = None
        if ls_xml_config is not None:
            if not os.path.isfile(ls_xml_config):
                raise FileNotFoundError(
                    f"Label Studio XML config not found: {ls_xml_config}"
                )            
        parsed_config = None
        if parse_ls_config:
            parsed_config = self.get_ls_parsed_config(ls_json_path=input_file)
        
        category_mapping = self.get_category_mapping(ls_xml_config=ls_xml_config,parsed_config=parsed_config)

        coco_data = self.convert_ls_json_to_coco(
            input_file=input_file,
            category_mapping=category_mapping,
            out_file_name=None,
        )

        # Create dataset info
        dataset_info = {
            "name": dataset_name,
            "description": "Dataset converted from Label Studio format",
            "version": "1.0",
            "contributor": "Label Studio Converter",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
            "classes": coco_data.get("categories", []),
        }

        return dataset_info, coco_data

    def convert_ls_json_to_coco(
        self,
        input_file: str,
        category_mapping: Optional[Dict[str, int]] = None,
        out_file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Converts LS json annotations to coco format via :meth:`LabelStudioParser.to_coco_format`.

        Args:
            input_file: path to LS json annotation file
            category_mapping: optional label name -> COCO category id (e.g. from XML)
            out_file_name: if not None, it will save the converted annotations

        Returns:
            dict: annotations in coco format
        """
        parser = LabelStudioParser.from_file(input_file)
        coco_annotations = parser.to_coco_format(category_mapping=category_mapping)

        if out_file_name is not None:
            self.save_coco_annotation(coco_annotations, out_file_name)

        return coco_annotations

    def get_ls_parsed_config(
        self,
        ls_json_path: str,
    ) -> Union[Dict[str, Any], None]:
        """
        Get parsed configuration from Label Studio project.

        Args:
            ls_json_path: Path to Label Studio JSON file
            ls_client: Label Studio client instance
            dotenv_path: Path to .env file

        Returns:
            Parsed Label Studio configuration
        """

        with open(ls_json_path, "r", encoding="utf-8") as f:
            ls_annotation = json.load(fp=f)

        ids = set([annot["project"] for annot in ls_annotation])
        if len(ids) != 1:
            raise ValueError(
                f"Annotations come from different projects. Project ids are: {ids}"
            )

        project_id = ids.pop()

        if self.ls_client is None:
            self.logger.warning(
                "No Label Studio client available, returning empty config"
            )
            return {}

        parsed_config = self.ls_client.projects.get(id=project_id).parsed_label_config
        return parsed_config

    def get_category_mapping(self, ls_xml_config: Optional[str]=None, parsed_config:Optional[dict]=None) -> Dict[str, int]:
        """
        Build label name -> COCO category id (1-based, stable) from a Label Studio
        labeling config XML. Uses ``<Label value=\"...\"/>`` elements in document order.
        """

        assert (ls_xml_config is None) ^ (parsed_config is None), "Exactly one should be provided."

        if parsed_config:
            labels = parsed_config['detections']['labels']
            mapping = dict()
            for i,label in enumerate(labels):
                mapping[label] = i+1
            return mapping

        tree = ET.parse(ls_xml_config)
        root = tree.getroot()
        labels: List[str] = []
        seen: set[str] = set()
        for el in root.iter():
            local = el.tag.split("}", 1)[-1]
            if local != "Label":
                continue
            raw = el.attrib.get("value")
            if raw is None or not str(raw).strip():
                continue
            name = str(raw).strip()
            if name not in seen:
                seen.add(name)
                labels.append(name)
        if not labels:
            self.logger.warning(
                "No <Label value=\"...\"/> entries found in %s; "
                "category IDs will follow the export default",
                ls_xml_config,
            )
        return {name: i + 1 for i, name in enumerate(labels)}