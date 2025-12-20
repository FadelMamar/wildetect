"""Configuration endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pathlib import Path

from ..models.requests import ConfigValidationRequest
from ..models.responses import (
    ConfigValidationResponse,
)
from ..dependencies import get_logger
from ...shared.config_types import ConfigType
from ...shared.validation import validate_config_file

logger = get_logger("config")

router = APIRouter()

@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a configuration file."""
    
    try:
        logger.info(f"Validating configuration file: {request.config_path}")
        
        validate_config_file(Path(request.config_path), ConfigType(request.config_type))
        is_valid = True
        errors = []
        warnings = []
               
        return ConfigValidationResponse(
            success=True,
            message="Configuration validation completed",
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            config_type=request.config_type
        )
        
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/types")
async def get_config_types() -> Dict[str, Any]:
    """Get available configuration types."""
    
    config_types = [
        {
            "name": ConfigType.CLASSIFICATION.value,
            "description": "Classification training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION.value,
            "description": "Detection training configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.CLASSIFICATION_EVAL.value,
            "description": "Classification evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION_EVAL.value,
            "description": "Detection evaluation configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.CLASSIFICATION_VISUALIZATION.value,
            "description": "Classification visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION_VISUALIZATION.value,
            "description": "Detection visualization configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.CLASSIFICATION_PIPELINE.value,
            "description": "Classification pipeline configuration",
            "file_extension": ".yaml"
        },
        {
            "name": ConfigType.DETECTION_PIPELINE.value,
            "description": "Detection pipeline configuration",
            "file_extension": ".yaml"
        }
    ]
    
    return {
        "success": True,
        "message": "Configuration types retrieved",
        "config_types": config_types,
        "total_types": len(config_types)
    }
