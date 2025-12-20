"""Visualization endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from ..models.requests import (
    ClassificationVisualizationRequest,
    DetectionVisualizationRequest
)
from ..models.responses import (
    VisualizationResponse
)
from ..utils.background_tasks import job_manager, create_background_job, JobStatus
from ..dependencies import get_logger
from ..services.visualization_service import VisualizationService

logger = get_logger("visualization")

router = APIRouter()


@router.post("/classifier", response_model=VisualizationResponse)
async def visualize_classifier_predictions(request: ClassificationVisualizationRequest) -> VisualizationResponse:
    """Visualize classifier predictions."""

    try:
        # Create background job for visualization
        job_id = create_background_job(
            task_func=_visualize_classifier_task,
            config=request.config,
            metadata={
                "task_type": "classification_visualization",
                "model_type": "classifier",
                "config": request.config.model_dump()
            }
        )

        return VisualizationResponse(
            success=True,
            message="Classification visualization job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create classification visualization job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detector", response_model=VisualizationResponse)
async def visualize_detector_predictions(request: DetectionVisualizationRequest) -> VisualizationResponse:
    """Visualize detector predictions."""

    try:
        # Create background job for visualization
        job_id = create_background_job(
            task_func=_visualize_detector_task,
            config=request.config,
            metadata={
                "task_type": "detection_visualization",
                "model_type": "detector",
                "config": request.config.model_dump()
            }
        )

        return VisualizationResponse(
            success=True,
            message="Detection visualization job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create detection visualization job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def get_fiftyone_datasets() -> Dict[str, Any]:
    """Get list of available FiftyOne datasets."""
    try:
        datasets_info = VisualizationService.get_fiftyone_datasets()
        return {
            "success": True,
            "message": f"Retrieved {datasets_info['total_count']} FiftyOne datasets",
            **datasets_info
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
            **dataset_info
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
        "duration": duration
    }


# Background task functions
def _visualize_classifier_task(config: Any) -> None:
    """Background task for classifier visualization using actual CLI integration."""
    logger.info("Starting classifier visualization task with CLI integration")
    
    try:
        # Use the visualization service to run actual CLI visualization
        VisualizationService.visualize_classifier_predictions(config)
        logger.info("Classifier visualization completed successfully")
        return
    except Exception as e:
        logger.error(f"Classifier visualization failed: {e}")
        raise


def _visualize_detector_task(config: Any) -> None:
    """Background task for detector visualization using actual CLI integration."""
    logger.info("Starting detector visualization task with CLI integration")
    
    try:
        # Use the visualization service to run actual CLI visualization
        VisualizationService.visualize_detector_predictions(config)
        logger.info("Detector visualization completed successfully")
        return
    except Exception as e:
        logger.error(f"Detector visualization failed: {e}")
        raise
