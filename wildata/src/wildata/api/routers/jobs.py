"""
Job management endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import verify_token
from ..exceptions import JobError, NotFoundError
from ..models.jobs import BackgroundJob, JobStatus
from ..models.responses import JobStatusResponse
from ..services.job_queue import get_job_queue

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}", response_model=JobStatusResponse, operation_id="get_job_status"
)
async def get_job_status(job_id: str, user=Depends(verify_token)):
    """Get the status of a background job."""
    try:
        job_queue = get_job_queue()
        job = await job_queue.get_job(job_id)

        if not job:
            raise NotFoundError(f"Job {job_id} not found", resource_type="job")

        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=job.result,
            job_type=job.job_type,
        )
    except JobError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/", response_model=List[JobStatusResponse], operation_id="list_jobs")
async def list_jobs(
    status_filter: JobStatus = None,
    limit: int = 50,
    offset: int = 0,
    user=Depends(verify_token),
):
    """List all background jobs with optional filtering."""
    try:
        job_queue = get_job_queue()
        jobs = await job_queue.list_jobs(
            status_filter=status_filter,
            limit=limit,
            offset=offset,
            user_id=user.get("user_id"),
        )

        return [
            JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                result=job.result,
                job_type=job.job_type,
            )
            for job in jobs
        ]
    except JobError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/{job_id}", operation_id="cancel_job")
async def cancel_job(job_id: str, user=Depends(verify_token)):
    """Cancel a background job."""
    try:
        job_queue = get_job_queue()
        success = await job_queue.cancel_job(job_id, user.get("user_id"))

        if not success:
            raise NotFoundError(
                f"Job {job_id} not found or cannot be cancelled", resource_type="job"
            )

        return {"message": f"Job {job_id} cancelled successfully"}
    except JobError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
