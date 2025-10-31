"""Evaluation endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

from ..models.requests import (
    ClassificationEvalRequest,
    DetectionEvalRequest
)
from ..models.responses import (
    EvaluationResponse
)
from ..utils.background_tasks import job_manager, create_background_job, JobStatus
from ..utils.error_handling import JobNotFoundError
from ..dependencies import get_logger
from ..services.evaluation_service import EvaluationService

logger = get_logger("evaluation")

router = APIRouter()


@router.post("/classifier", response_model=EvaluationResponse)
async def evaluate_classifier(request: ClassificationEvalRequest) -> EvaluationResponse:
    """Evaluate a classification model."""

    try:
        # Create background job for evaluation
        config = request.config
        job_id = create_background_job(
            task_func=_evaluate_classifier_task,
            config=config,            
            debug=request.debug,
            metadata={
                "task_type": "classification_evaluation",
                "model_type": "classifier",
                "config": config.model_dump()
            }
        )

        return EvaluationResponse(
            success=True,
            message="Classification evaluation job created",
            metrics=dict(),
            results_path=None,
            confusion_matrix=None,
            class_metrics=None,
            job_id=job_id
        )

    except Exception as e:
        logger.error(f"Failed to create classification evaluation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detector", response_model=EvaluationResponse)
async def evaluate_detector(request: DetectionEvalRequest) -> EvaluationResponse:
    """Evaluate a detection model."""

    try:
        # Create background job for evaluation
        config = request.config
        job_id = create_background_job(
            task_func=_evaluate_detector_task,
            config=config,
            debug=request.debug,
            model_type=request.model_type,
            metadata={
                "task_type": "detection_evaluation",
                "model_type": "detector",
                "config": config.model_dump()
            }
        )

        return EvaluationResponse(
            success=True,
            message="Detection evaluation job created",
            metrics=dict(),
            results_path=None,
            confusion_matrix=None,
            class_metrics=None,
            job_id=job_id
        )

    except Exception as e:
        logger.error(f"Failed to create detection evaluation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_evaluation_status(job_id: str) -> Dict[str, Any]:
    """Get evaluation job status."""

    job = job_manager.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)

    # Calculate duration if job is completed
    duration = None
    if job.completed_at and job.started_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return {
        "success": True,
        "message": f"Evaluation job {job_id} status retrieved",
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


@router.get("/jobs")
async def list_evaluation_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """List evaluation jobs."""

    jobs = job_manager.get_jobs()

    # Filter by status if specified
    if status:
        try:
            status_enum = JobStatus(status)
            jobs = [job for job in jobs if job.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    # Apply pagination
    total_count = len(jobs)
    jobs = jobs[offset:offset + limit]

    # Convert to dict format
    job_dicts = []
    for job in jobs:
        job_dicts.append({
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "progress": job.progress,
            "metadata": job.metadata
        })

    return {
        "success": True,
        "message": f"Retrieved {len(job_dicts)} evaluation jobs",
        "jobs": job_dicts,
        "total_count": total_count,
        "filtered_count": len(job_dicts)
    }


# Background task functions
def _evaluate_classifier_task(config: Any,debug: bool) -> Dict[str, Any]:
    """Background task for classifier evaluation using actual CLI integration."""
    logger.info("Starting classifier evaluation task with CLI integration")
    
    try:
        # Use the evaluation service to run actual CLI evaluation
        results = EvaluationService.evaluate_classifier(config,debug)
        logger.info("Classifier evaluation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Classifier evaluation failed: {e}")
        raise


def _evaluate_detector_task(config: Any,model_type: str,debug: bool) -> Dict[str, Any]:
    """Background task for detector evaluation using actual CLI integration."""
    logger.info("Starting detector evaluation task with CLI integration")
    
    try:
        # Use the evaluation service to run actual CLI evaluation
        results = EvaluationService.evaluate_detector(config,model_type,debug)
        logger.info("Detector evaluation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Detector evaluation failed: {e}")
        raise
