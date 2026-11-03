"""Configuration endpoints for the WildTrain API."""

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ...shared.config_types import ConfigType
from ..dependencies import get_logger
from ..models.requests import ConfigValidationRequest
from ..models.responses import (
    ConfigValidationResponse,
)

logger = get_logger("config")

router = APIRouter()


@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a configuration file."""

    kind = request.config_type
    harness_kinds = {
        "experiment",
        "eval",
        "visualization",
        "registration",
    }
    retired = {item.value for item in ConfigType}
    try:
        logger.info(f"Validating configuration file: {request.config_path}")
        if kind == "pipeline":
            raise HTTPException(
                status_code=422,
                detail=(
                    "Pipeline configs have been removed. Set eval.enabled: true "
                    "on an experiment and use POST /experiments."
                ),
            )
        if kind in harness_kinds:
            from ...harness.ops import validate_kind

            validate_kind(Path(request.config_path), kind)
        elif kind in retired:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Config type {kind!r} validated the old trainer schema. "
                    "Use experiment, eval, visualization, or registration."
                ),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown config type {kind!r}. "
                    "Use experiment, eval, visualization, or registration."
                ),
            )
        return ConfigValidationResponse(
            success=True,
            message="Configuration validation completed",
            is_valid=True,
            errors=[],
            warnings=[],
            config_type=request.config_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_config_types() -> Dict[str, Any]:
    """Get available configuration types."""

    config_types = [
        {
            "name": "experiment",
            "description": "Classification or detection experiment",
            "file_extension": ".yaml",
        },
        {
            "name": "eval",
            "description": "Standalone evaluation document",
            "file_extension": ".yaml",
        },
        {
            "name": "visualization",
            "description": "FiftyOne visualization document",
            "file_extension": ".yaml",
        },
        {
            "name": "registration",
            "description": "MLflow registration document",
            "file_extension": ".yaml",
        },
    ]

    return {
        "success": True,
        "message": "Configuration types retrieved",
        "config_types": config_types,
        "total_types": len(config_types),
    }
