"""Configuration loader for CLI with Pydantic validation."""

import json
from pathlib import Path
from typing import Dict, Any, Type, TypeVar, Union, Optional
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import yaml
import traceback
from enum import Enum

from ..shared.models import (
    ClassificationConfig,
    DetectionConfig,
    DetectionVisualizationConfig,
    ClassificationPipelineConfig,
    DetectionPipelineConfig,
    ClassificationEvalConfig,
    DetectionEvalConfig,
    ClassificationVisualizationConfig,
    LocalizerRegistrationConfig,
    ClassifierRegistrationConfig,
    DetectorRegistrationConfig,
)
from ..shared.config_types import ConfigType
from ..shared.validation import ConfigValidationError, ConfigFileNotFoundError, ConfigParseError

console = Console()

T = TypeVar('T', bound=BaseModel)

ROOT = Path(__file__).parents[3]


class ConfigLoader:
    """Configuration loader with Pydantic validation."""
    
    @staticmethod
    def load_and_validate(
        config_path: Path,
        config_class: Type[T],
        config_name: str = "configuration"
    ) -> T:
        """
        Load YAML configuration file and validate against Pydantic model.
        
        Args:
            config_path: Path to the configuration file
            config_class: Pydantic model class to validate against
            config_name: Name of the configuration for error messages
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigFileNotFoundError: If config file doesn't exist
            ConfigParseError: If config file cannot be parsed
            ConfigValidationError: If config fails validation
        """
        # Check if file exists
        if not config_path.exists():
            raise ConfigFileNotFoundError(f"{config_name} file not found: {config_path}")
        
        try:
            # Load YAML using OmegaConf
            cfg = OmegaConf.load(config_path)
            
            # Convert to dict
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            
            # Validate against Pydantic model
            validated_config = config_class(**config_dict)
            
            console.print(f"[bold green]✓[/bold green] {config_name} loaded and validated successfully")
            
            return validated_config
            
        except Exception as e:
            if isinstance(e, (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError)):
                raise
            
            if isinstance(e, ValidationError):
                # Format Pydantic validation errors
                error_messages = []
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    error_messages.append(f"  • {field_path}: {error['msg']}")  # type: ignore
                
                error_text = f"Configuration validation failed:\n" + "\n".join(error_messages)
                console.print(Panel(error_text, title="❌ Validation Error", border_style="red"))
                raise ConfigValidationError(f"{config_name} validation failed: {str(e)}")
            
            raise ConfigParseError(f"Failed to parse {config_name} file: {traceback.format_exc()}")
    
    @staticmethod
    def load_classification_config(config_path: Path) -> ClassificationConfig:
        """Load and validate classification configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            ClassificationConfig, 
            "Classification configuration"
        )
    
    @staticmethod
    def load_classification_eval_config(config_path: Path) -> ClassificationEvalConfig:
        """Load and validate classification evaluation configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            ClassificationEvalConfig, 
            "Classification evaluation configuration"
        )
    
    @staticmethod
    def load_detection_eval_config(config_path: Path) -> DetectionEvalConfig:
        """Load and validate detection evaluation configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            DetectionEvalConfig, 
            "Detection evaluation configuration"
        )
    
    @staticmethod
    def load_detection_config(config_path: Path) -> DetectionConfig:
        """Load and validate detection configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            DetectionConfig, 
            "Detection configuration"
        )
    
    @staticmethod
    def load_detection_visualization_config(config_path: Path) -> DetectionVisualizationConfig:
        """Load and validate detection visualization configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            DetectionVisualizationConfig, 
            "Detection visualization configuration"
        )
    
    @staticmethod
    def load_classification_visualization_config(config_path: Path) -> ClassificationVisualizationConfig:
        """Load and validate classification visualization configuration."""
        return ConfigLoader.load_and_validate(
            config_path, 
            ClassificationVisualizationConfig, 
            "Classification visualization configuration"
        )
    
    @staticmethod
    def load_localizer_registration_config(config_path: Path) -> LocalizerRegistrationConfig:
        """Load and validate detector registration configuration."""
        return ConfigLoader.load_and_validate(
            config_path,
            LocalizerRegistrationConfig,
            "Localizer registration configuration"
        )
    
    @staticmethod
    def load_classifier_registration_config(config_path: Path) -> ClassifierRegistrationConfig:
        """Load and validate classifier registration configuration."""
        return ConfigLoader.load_and_validate(
            config_path,
            ClassifierRegistrationConfig,
            "Classifier registration configuration"
        )
    
    @staticmethod
    def load_detector_registration_config(config_path: Path) -> DetectorRegistrationConfig:
        """Load and validate Detector registration configuration."""
        return ConfigLoader.load_and_validate(
            config_path,
            DetectorRegistrationConfig,
            "Model registration configuration"
        )
    
    @staticmethod
    def load_pipeline_config(
        config_path: Path, 
        pipeline_type: str = "classification"
    ) -> Union[ClassificationPipelineConfig, DetectionPipelineConfig]:
        """Load and validate pipeline configuration."""
        if pipeline_type == "classification":
            return ConfigLoader.load_and_validate(
                config_path, 
                ClassificationPipelineConfig, 
                "Classification pipeline configuration"
            )
        elif pipeline_type == "detection":
            return ConfigLoader.load_and_validate(
                config_path, 
                DetectionPipelineConfig, 
                "Detection pipeline configuration"
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    @staticmethod
    def validate_config_file(config_path: Path, config_type: ConfigType) -> bool:
        """Validate a configuration file using the appropriate Pydantic model."""
        try:
            if config_type == ConfigType.CLASSIFICATION:
                ConfigLoader.load_classification_config(config_path)
            elif config_type == ConfigType.DETECTION:
                ConfigLoader.load_detection_config(config_path)
            elif config_type == ConfigType.CLASSIFICATION_EVAL:
                ConfigLoader.load_classification_eval_config(config_path)
            elif config_type == ConfigType.DETECTION_EVAL:
                ConfigLoader.load_detection_eval_config(config_path)
            elif config_type == ConfigType.CLASSIFICATION_VISUALIZATION:
                ConfigLoader.load_classification_visualization_config(config_path)
            elif config_type == ConfigType.DETECTION_VISUALIZATION:
                ConfigLoader.load_detection_visualization_config(config_path)
            elif config_type == ConfigType.CLASSIFICATION_PIPELINE:
                ConfigLoader.load_pipeline_config(config_path, "classification")
            elif config_type == ConfigType.DETECTION_PIPELINE:
                ConfigLoader.load_pipeline_config(config_path, "detection")
            elif config_type == ConfigType.DETECTOR_REGISTRATION:
                ConfigLoader.load_localizer_registration_config(config_path)
            elif config_type == ConfigType.CLASSIFIER_REGISTRATION:
                ConfigLoader.load_classifier_registration_config(config_path)
            elif config_type == ConfigType.MODEL_REGISTRATION:
                ConfigLoader.load_model_registration_config(config_path)
            else:
                raise ValueError(f"Unknown config type: {config_type}")
            return True
        except Exception as e:
            raise ConfigValidationError(f"{config_type} configuration validation failed: {str(e)}")
    
    @staticmethod
    def generate_schema(config_class: Type[BaseModel]) -> Dict[str, Any]:
        """Generate JSON schema for a configuration class."""
        return config_class.model_json_schema()
        
    @staticmethod
    def save_schema(config_class: Type[T], output_path: Path, save_format: str = "json") -> None:
        """Save schema to file."""
        schema = ConfigLoader.generate_schema(config_class)
        with open(output_path, 'w', encoding='utf-8') as f:
            if save_format == "json":
                json.dump(schema, f, indent=2)
            elif save_format == "yaml":
                yaml.dump(schema, f, indent=2)
            else:
                raise ValueError(f"Unknown format: {save_format}. Valid formats: json, yaml")
        console.print(f"[bold green]✓[/bold green] Schema saved to: {output_path}")
    
    @staticmethod
    def generate_default_config(config_type: ConfigType) -> str:
        """Generate a default YAML configuration template for the given config type."""

        assert isinstance(config_type, ConfigType), f"config_type must be a ConfigType enum, got: {config_type}"
        
        # Map config types to their corresponding Pydantic model classes
        config_class_map = {
            ConfigType.CLASSIFICATION: ClassificationConfig,
            ConfigType.DETECTION: DetectionConfig,
            ConfigType.CLASSIFICATION_EVAL: ClassificationEvalConfig,
            ConfigType.DETECTION_EVAL: DetectionEvalConfig,
            ConfigType.CLASSIFICATION_VISUALIZATION: ClassificationVisualizationConfig,
            ConfigType.DETECTION_VISUALIZATION: DetectionVisualizationConfig,
            ConfigType.CLASSIFICATION_PIPELINE: ClassificationPipelineConfig,
            ConfigType.DETECTION_PIPELINE: DetectionPipelineConfig,
            ConfigType.DETECTOR_REGISTRATION: LocalizerRegistrationConfig,
            ConfigType.CLASSIFIER_REGISTRATION: ClassifierRegistrationConfig,
            ConfigType.MODEL_REGISTRATION: DetectorRegistrationConfig,
        }
        
        if config_type not in config_class_map:
            raise ValueError(f"Unknown config type: {config_type}. Valid types: {list(config_class_map.keys())}")
        
        # Get the Pydantic model class
        config_class = config_class_map[config_type]
        
        # Generate schema from the model
        schema = ConfigLoader.generate_schema(config_class)
        
        # Convert schema to a template with example values
        template = ConfigLoader._schema_to_template(schema)
        
        return str(yaml.dump(template, default_flow_style=False, indent=2, sort_keys=False))
    
    @staticmethod
    def _schema_to_template(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a JSON schema to a YAML template with example values."""
        template = {}
        
        if "properties" not in schema:
            return template
        
        # Get the $defs section for resolving references
        defs = schema.get("$defs", {})
        
        for prop_name, prop_schema in schema["properties"].items():
            if prop_name == "model_config":  # Skip internal Pydantic config
                continue
                
            template[prop_name] = ConfigLoader._get_example_value(prop_schema, defs)
        
        return template
    
    @staticmethod
    def _get_example_value(schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None) -> Any:
        """Get an example value for a schema property."""
        prop_type = schema.get("type", "string")
        
        # Handle $ref references
        if "$ref" in schema:
            ref_path = schema["$ref"]
            if ref_path.startswith("#/$defs/") and defs:
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    return ConfigLoader._get_example_value(defs[ref_name], defs)
            # If we can't resolve the ref, fall back to default
            return "example_string"
        
        if "enum" in schema:
            # Use first enum value as example
            return schema["enum"][0]
        
        if prop_type == "object" and "properties" in schema:
            # Recursively build nested object
            nested_obj = {}
            for nested_prop_name, nested_prop_schema in schema["properties"].items():
                if nested_prop_name == "model_config":
                    continue
                nested_obj[nested_prop_name] = ConfigLoader._get_example_value(nested_prop_schema, defs)
            return nested_obj
        
        if prop_type == "array" and "items" in schema:
            # Create array with single example item
            example_item = ConfigLoader._get_example_value(schema["items"], defs)
            return [example_item]
        
        # Handle different types with sensible defaults
        if prop_type == "string":
            if "description" in schema:
                # Use description as hint for example value
                desc = schema["description"].lower()
                if "path" in desc or "file" in desc or "directory" in desc:
                    return "path/to/file"
                elif "device" in desc:
                    return "cpu"
                elif "name" in desc:
                    return "example_name"
                elif "field" in desc:
                    return "example_field"
                else:
                    return "example_string"
            else:
                return "example_string"
        
        elif prop_type == "integer":
            if "minimum" in schema:
                min_val = schema["minimum"]
                if min_val <= 0:
                    return 1
                elif min_val <= 10:
                    return min_val
                else:
                    return max(min_val, 1)
            elif "default" in schema:
                return schema["default"]
            else:
                return 1
        
        elif prop_type == "number":
            if "minimum" in schema:
                min_val = schema["minimum"]
                if min_val <= 0:
                    return 0.1
                else:
                    return max(min_val, 0.1)
            elif "default" in schema:
                return schema["default"]
            else:
                return 0.1
        
        elif prop_type == "boolean":
            if "default" in schema:
                return schema["default"]
            else:
                return False
        
        elif prop_type == "null":
            return None
        
        else:
            return "example_value"
    
