"""Visualization endpoints for the WildTrain API."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ...harness.ops import visualization_from_mapping
from ..dependencies import get_logger
from ..models.requests import ExperimentTrainingRequest
from ..models.responses import VisualizationResponse
from ..services.visualization_service import VisualizationService
from ..utils.background_tasks import create_background_job, job_manager

logger = get_logger("visualization")

router = APIRouter()


@router.post("/", response_model=VisualizationResponse)
async def visualize_document(
    request: ExperimentTrainingRequest,
) -> VisualizationResponse:
    """Visualize the task named in the document."""

    try:
        visualization_from_mapping(request.config)
        job_id = create_background_job(
            task_func=_visualize_document_task,
            config=request.config,
            metadata={"task_type": "visualization"},
        )
        return VisualizationResponse(
            success=True,
            message="Visualization job created",
            job_id=job_id,
            status="pending",
            progress=0.0,
        )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to create visualization job: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _visualize_document_task(config: Dict[str, Any]) -> None:
    from ...harness.runner import default_registry

    document = visualization_from_mapping(config)
    default_registry().get(document.task).visualize(document)


@router.get("/datasets")
async def get_fiftyone_datasets() -> Dict[str, Any]:
    """Get list of available FiftyOne datasets."""
    try:
        datasets_info = VisualizationService.get_fiftyone_datasets()
        return {
            "success": True,
            "message": f"Retrieved {datasets_info['total_count']} FiftyOne datasets",
            **datasets_info,
        }
    except Exception as e:
        logger.error(f"Failed to get FiftyOne datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a specific FiftyOne dataset."""
    try:
        dataset_info = VisualizationService.get_dataset_info(dataset_name)
        return {
            "success": True,
            "message": f"Retrieved dataset info for {dataset_name}",
            **dataset_info,
        }
    except Exception as e:
        logger.error(f"Failed to get dataset info for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_visualization_status(job_id: str) -> Dict[str, Any]:
    """Get visualization job status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Calculate duration if job is completed
    duration = None
    if job.completed_at and job.started_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return {
        "success": True,
        "message": f"Visualization job {job_id} status retrieved",
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "logs": job.logs,
        "metadata": job.metadata,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "duration": duration,
    }


