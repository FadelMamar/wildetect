import io
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dotenv import load_dotenv
from label_studio_converter import Converter
from label_studio_sdk.client import LabelStudio

from ..config import ROOT
from .base_converter import BaseConverter


class LabelstudioConverter(BaseConverter):
    """
    Converter for Label Studio JSON annotations to COCO format.
    Inherits from BaseConverter and implements the required interface.
    """

    def __init__(self, dotenv_path: Optional[str] = None, client: Optional[LabelStudio] = None):
        """
        Initialize the Label Studio converter.

        Args:
            label_studio_url: Label Studio server URL (optional, can be set via env var)
            api_key: Label Studio API key (optional, can be set via env var)
        """
        super().__init__()

        self.dotenv_path = dotenv_path or ROOT/".env"

        if dotenv_path is not None:
            load_dotenv(dotenv_path)

        label_studio_url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        try:
            self.ls_client = client or LabelStudio(base_url=label_studio_url, api_key=api_key)
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

        parsed_config = None
        if parse_ls_config:
            parsed_config = self.get_ls_parsed_config(
                ls_json_path=input_file,
            )

        coco_data = self._convert_ls_json_to_coco(
            input_file=input_file,
            out_file_name=None,
            parsed_config=parsed_config,
            ls_xml_config=ls_xml_config,
        )

        # Create dataset info
        dataset_info = {
            "name": dataset_name,
            "description": f"Dataset converted from Label Studio format",
            "version": "1.0",
            "contributor": "Label Studio Converter",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
            "classes": coco_data.get("categories", []),
        }

        return dataset_info, coco_data

    def _convert_ls_json_to_coco(
        self,
        input_file: str,
        ls_xml_config: Optional[str] = None,
        out_file_name: Optional[str] = None,
        parsed_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Converts LS json annotations to coco format

        Args:
            input_file: path to LS json annotation file
            ls_xml_config: path to Label Studio XML config file
            out_file_name: if not None, it will save the converted annotations
            parsed_config: parsed Label Studio configuration

        Returns:
            dict: annotations in coco format
        """

        # Load converter
        config_str = None
        assert not all(
            [parsed_config is not None, ls_xml_config is not None]
        ), "Either parsed_config or ls_xml_config must be provided"

        if parsed_config is not None:
            config_str = parsed_config

        else:
            with open(ls_xml_config, encoding="utf-8") as f:
                config_str = f.read()

        handler = Converter(
            config=config_str, project_dir=None, download_resources=False
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            handler.convert_to_coco(
                input_data=input_file,
                output_dir=tmp_dir,
                output_image_dir=os.path.join(tmp_dir, "images"),
                is_dir=False,
            )
            # Load and update image paths
            coco_json_path = os.path.join(tmp_dir, "result.json")

            with open(coco_json_path, "r", encoding="utf-8") as f:
                coco_annotations = json.load(f)

            if out_file_name is not None:
                self.save_coco_annotation(coco_annotations, out_file_name)

        return coco_annotations

    def get_ls_parsed_config(
        self, ls_json_path: str,
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
            raise ValueError(f"Annotations come from different projects. Project ids are: {ids}")

        project_id = ids.pop()

        if self.ls_client is None:
            self.logger.warning(
                "No Label Studio client available, returning empty config"
            )
            return {}

        project = self.ls_client.projects.get(id=project_id)
        parsed_config = project.parsed_label_config

        return parsed_config
