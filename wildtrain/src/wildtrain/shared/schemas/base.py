"""Base configuration classes and enums for all schemas."""

from pathlib import Path
from typing import Any
from pydantic import BaseModel
import yaml
from enum import StrEnum


class BaseConfig(BaseModel):
    """Base configuration class with common fields."""
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Create config from YAML file."""
        if not Path(yaml_path).exists():
            raise ValueError(f"YAML file does not exist: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


class SweepObjectiveTypes(StrEnum):
    """Types of benchmark objective."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAP = "map"
    MAP_50 = "map_50"
    MAP_50_95 = "map_50_95"
    FITNESS = "fitness"


class SweepDirectionTypes(StrEnum):
    """Types of benchmark direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
