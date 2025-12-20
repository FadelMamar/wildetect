"""Training endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from ..models.requests import (
    ClassificationTrainingRequest,
    DetectionTrainingRequest,
    JobStatusRequest,
    JobCancelRequest
)
from ..models.responses import (
    TrainingResponse,
    JobDetailResponse,
    JobListResponse
)
from ..utils.background_tasks import job_manager, create_background_job, JobStatus
from ..utils.error_handling import JobNotFoundError, JobExecutionError
from ..dependencies import get_settings, get_logger
from ..services.training_service import TrainingService

logger = get_logger("training")

router = APIRouter()


@router.post("/classifier", response_model=TrainingResponse)
async def train_classifier(request: ClassificationTrainingRequest) -> TrainingResponse:
    """Train a classification model."""

    try:
        # Create background job for training
        job_id = create_background_job(
            task_func=_train_classifier_task,
            config=request.config,
            metadata={
                "task_type": "classification_training",
                "model_type": "classifier",
                "config": request.config
            }
        )

        return TrainingResponse(
            success=True,
            message="Classification training job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create classification training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detector", response_model=TrainingResponse)
async def train_detector(request: DetectionTrainingRequest) -> TrainingResponse:
    """Train a detection model."""

    try:
        # Create background job for training
        job_id = create_background_job(
            task_func=_train_detector_task,
            config=request.config,
            metadata={
                "task_type": "detection_training",
                "model_type": "detector",
                "config": request.config
            }
        )

        return TrainingResponse(
            success=True,
            message="Detection training job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create detection training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobDetailResponse)
async def get_training_status(job_id: str) -> JobDetailResponse:
    """Get training job status."""

    job = job_manager.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)

    # Calculate duration if job is completed
    duration = None
    if job.completed_at and job.started_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return JobDetailResponse(
        success=True,
        message=f"Job {job_id} status retrieved",
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        logs=job.logs,
        metadata=job.metadata,
        result_files=[],  # TODO: Get actual result files
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration=duration
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> JobListResponse:
    """List training jobs."""

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

    return JobListResponse(
        success=True,
        message=f"Retrieved {len(job_dicts)} training jobs",
        jobs=job_dicts,
        total_count=total_count,
        filtered_count=len(job_dicts)
    )


@router.post("/cancel", response_model=JobDetailResponse)
async def cancel_training_job(request: JobCancelRequest) -> JobDetailResponse:
    """Cancel a training job."""

    success = job_manager.cancel_job(request.job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not cancel job {request.job_id}. Job may not exist or may not be cancellable."
        )

    job = job_manager.get_job(request.job_id)
    if not job:
        raise JobNotFoundError(request.job_id)

    return JobDetailResponse(
        success=True,
        message=f"Job {request.job_id} cancelled successfully",
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        logs=job.logs,
        metadata=job.metadata,
        result_files=[],
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


# Background task functions
def _train_classifier_task(config: Any) -> None:
    """Background task for classifier training using actual CLI integration."""
    logger.info("Starting classifier training task with CLI integration")
    
    try:
        # Use the training service to run actual CLI training
        TrainingService.train_classifier(config)
        logger.info("Classifier training completed successfully")
        return
    except Exception as e:
        logger.error(f"Classifier training failed: {e}")
        raise


def _train_detector_task(config: Any) -> None:
    """Background task for detector training using actual CLI integration."""
    logger.info("Starting detector training task with CLI integration")
    
    try:
        # Use the training service to run actual CLI training
        TrainingService.train_detector(config)
        logger.info("Detector training completed successfully")
        return
    except Exception as e:
        logger.error(f"Detector training failed: {e}")
        raise

