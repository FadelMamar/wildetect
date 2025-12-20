"""
Background job queue implementation.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..exceptions import JobError
from ..models.jobs import BackgroundJob, JobResult, JobStatus

# Global job queue instance
_job_queue: Optional["JobQueue"] = None


class JobQueue:
    """Background job queue implementation."""

    def __init__(self):
        self._jobs: Dict[str, BackgroundJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(
        self, job_type: str, parameters: Dict, user_id: Optional[str] = None
    ) -> BackgroundJob:
        """Create a new background job."""
        async with self._lock:
            job_id = str(uuid.uuid4())
            job = BackgroundJob(
                job_id=job_id,
                job_type=job_type,
                parameters=parameters,
                user_id=user_id,
                status=JobStatus.PENDING,
            )
            self._jobs[job_id] = job
            return job

    async def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        """Get a job by ID."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        result: Optional[JobResult] = None,
    ) -> bool:
        """Update job status and progress."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            job.status = status
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result

            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.now()

            return True

    async def list_jobs(
        self,
        status_filter: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
    ) -> List[BackgroundJob]:
        """List jobs with optional filtering."""
        async with self._lock:
            jobs = list(self._jobs.values())

            # Filter by user if specified
            if user_id:
                jobs = [job for job in jobs if job.user_id == user_id]

            # Filter by status if specified
            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)

            # Apply pagination
            return jobs[offset : offset + limit]

    async def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            # Check if user can cancel this job
            if user_id and job.user_id != user_id:
                return False

            # Only allow cancellation of pending or running jobs
            if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
                return False

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.result = JobResult(
                success=False,
                message="Job cancelled by user",
                error="Job was cancelled",
            )

            return True

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed/cancelled jobs."""
        cutoff_time = datetime.now().replace(hour=datetime.now().hour - max_age_hours)

        async with self._lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                if (
                    job.status
                    in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                    and job.completed_at
                    and job.completed_at < cutoff_time
                ):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self._jobs[job_id]

            return len(jobs_to_remove)


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
