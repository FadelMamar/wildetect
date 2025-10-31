"""Pipeline endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

from ..models.requests import (
    ClassificationPipelineRequest,
    DetectionPipelineRequest,
    JobStatusRequest,
    JobCancelRequest
)
from ..models.responses import (
    PipelineResponse,
    JobDetailResponse,
    JobListResponse
)
from ..utils.background_tasks import job_manager, create_background_job, JobStatus
from ..utils.error_handling import JobNotFoundError
from ..dependencies import get_logger
from ..services.pipeline_service import PipelineService

logger = get_logger("pipeline")

router = APIRouter()


@router.post("/classification", response_model=PipelineResponse)
async def run_classification_pipeline(request: ClassificationPipelineRequest) -> PipelineResponse:
    """Run classification pipeline."""

    try:

        # Create background job for pipeline
        job_id = create_background_job(
            task_func=_run_classification_pipeline_task,
            config=request.config,
            debug=request.debug,
            verbose=request.verbose,
            metadata={
                "task_type": "classification_pipeline",
                "pipeline_type": "classification",
                "config": request.config.model_dump()
            }
        )

        return PipelineResponse(
            success=True,
            message="Classification pipeline job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create classification pipeline job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detection", response_model=PipelineResponse)
async def run_detection_pipeline(request: DetectionPipelineRequest) -> PipelineResponse:
    """Run detection pipeline."""

    try:
        
        # Create background job for pipeline
        job_id = create_background_job(
            task_func=_run_detection_pipeline_task,
            config=request.config,
            debug=request.debug,
            verbose=request.verbose,
            metadata={
                "task_type": "detection_pipeline",
                "pipeline_type": "detection",
                "config": request.config.model_dump()
            }
        )

        return PipelineResponse(
            success=True,
            message="Detection pipeline job created",
            job_id=job_id,
            status="pending",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to create detection pipeline job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobDetailResponse)
async def get_pipeline_status(job_id: str) -> JobDetailResponse:
    """Get pipeline job status."""

    job = job_manager.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)

    # Calculate duration if job is completed
    duration = None
    if job.completed_at and job.started_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return JobDetailResponse(
        success=True,
        message=f"Pipeline job {job_id} status retrieved",
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
async def list_pipeline_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> JobListResponse:
    """List pipeline jobs."""

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
        message=f"Retrieved {len(job_dicts)} pipeline jobs",
        jobs=job_dicts,
        total_count=total_count,
        filtered_count=len(job_dicts)
    )


@router.post("/cancel", response_model=JobDetailResponse)
async def cancel_pipeline_job(request: JobCancelRequest) -> JobDetailResponse:
    """Cancel a pipeline job."""

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
        message=f"Pipeline job {request.job_id} cancelled successfully",
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
def _run_classification_pipeline_task(config: Any, debug: bool, verbose: bool) -> Dict[str, Any]:
    """Background task for classification pipeline using actual CLI integration."""
    logger.info("Starting classification pipeline task with CLI integration")
    
    try:
        # Use the pipeline service to run actual CLI pipeline
        results = PipelineService.run_classification_pipeline(config)
        logger.info("Classification pipeline completed successfully")
        return results
    except Exception as e:
        logger.error(f"Classification pipeline failed: {e}")
        raise


def _run_detection_pipeline_task(config: Any, debug: bool, verbose: bool) -> Dict[str, Any]:
    """Background task for detection pipeline using actual CLI integration."""
    logger.info("Starting detection pipeline task with CLI integration")
    
    try:
        # Use the pipeline service to run actual CLI pipeline
        results = PipelineService.run_detection_pipeline(config)
        logger.info("Detection pipeline completed successfully")
        return results
    except Exception as e:
        logger.error(f"Detection pipeline failed: {e}")
        raise

