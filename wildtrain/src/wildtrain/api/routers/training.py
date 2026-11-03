"""Training endpoints for the WildTrain API."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ..dependencies import get_logger
from ..models.requests import ExperimentTrainingRequest, JobCancelRequest
from ..models.responses import JobDetailResponse, JobListResponse, TrainingResponse
from ..services.training_service import TrainingService
from ..utils.background_tasks import JobStatus, create_background_job, job_manager
from ..utils.error_handling import JobNotFoundError

logger = get_logger("training")

router = APIRouter()


async def train_experiment(request: ExperimentTrainingRequest) -> TrainingResponse:
    """Train the task named in the experiment document.

    Mounted only at ``POST /experiments``.
    """

    try:
        job_id = create_background_job(
            task_func=_train_experiment_task,
            config=request.config,
            debug=request.debug,
            metadata={
                "task_type": "experiment",
                "config": request.config,
            },
        )
        return TrainingResponse(
            success=True,
            message="Experiment training job created",
            job_id=job_id,
            status="pending",
            progress=0.0,
        )
    except Exception as e:
        logger.error(f"Failed to create experiment training job: {e}")
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
        duration=duration,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_training_jobs(
    status: Optional[str] = None, limit: int = 50, offset: int = 0
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
    jobs = jobs[offset : offset + limit]

    # Convert to dict format
    job_dicts = []
    for job in jobs:
        job_dicts.append(
            {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "progress": job.progress,
                "metadata": job.metadata,
            }
        )

    return JobListResponse(
        success=True,
        message=f"Retrieved {len(job_dicts)} training jobs",
        jobs=job_dicts,
        total_count=total_count,
        filtered_count=len(job_dicts),
    )


@router.post("/cancel", response_model=JobDetailResponse)
async def cancel_training_job(request: JobCancelRequest) -> JobDetailResponse:
    """Cancel a training job."""

    success = job_manager.cancel_job(request.job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not cancel job {request.job_id}. Job may not exist or may not be cancellable.",
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
        completed_at=job.completed_at,
    )


# Background task functions
def _train_experiment_task(config: Any, debug: bool = False) -> None:
    """Background task that dispatches through the harness registry."""
    logger.info("Starting experiment training task")
    try:
        TrainingService.train_experiment(config, debug=debug)
        logger.info("Experiment training completed successfully")
    except Exception as e:
        logger.error(f"Experiment training failed: {e}")
        raise


