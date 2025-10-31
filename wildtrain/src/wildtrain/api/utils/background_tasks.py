"""Background task management for the WildTrain API."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Coroutine
from enum import Enum
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..dependencies import get_settings, get_job_logger

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Job information and metadata."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

#TODO replace with dramatiq https://dramatiq.io/motivation.html#compared-to 
class JobManager:
    """Manages background jobs and their status."""
    
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._settings = get_settings()
    
    def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        self._jobs[job_id] = job_info
        logger.info(f"Created job {job_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information by ID."""
        return self._jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, **kwargs) -> None:
        """Update job status and optional fields."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._jobs[job_id]
        job.status = status
        
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.now()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = datetime.now()
        
        # Update other fields
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        logger.info(f"Updated job {job_id} status to {status}")
    
    def add_job_log(self, job_id: str, message: str, level: str = "INFO") -> None:
        """Add a log message to a job."""
        if job_id not in self._jobs:
            return
        
        job = self._jobs[job_id]
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        job.logs.append(log_entry)
        
        # Also log to the job-specific logger
        job_logger = get_job_logger(job_id)
        if level == "ERROR":
            job_logger.error(message)
        elif level == "WARNING":
            job_logger.warning(message)
        else:
            job_logger.info(message)
    
    def get_jobs(self, status: Optional[JobStatus] = None) -> list:
        """Get all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [job for job in jobs if job.status == status]
        return jobs
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> None:
        """Clean up old completed/failed jobs."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        for job_id, job in self._jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.completed_at and job.completed_at < cutoff):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still pending or running."""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self.add_job_log(job_id, "Job cancelled by user", "INFO")
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False


# Global job manager instance
job_manager = JobManager()


async def run_background_task(
    task_func: Callable,
    job_id: str,
    *args,
    **kwargs
) -> None:
    """Run a task in the background with job tracking."""
    
    job_manager.update_job_status(job_id, JobStatus.RUNNING)
    job_manager.add_job_log(job_id, "Starting background task", "INFO")
    
    try:
        # Run the task
        if asyncio.iscoroutinefunction(task_func):
            result = await task_func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            # Create a wrapper function to handle keyword arguments
            def run_sync_task():
                return task_func(*args, **kwargs)
            result = await loop.run_in_executor(None, run_sync_task)
        
        # Update job with success
        job_manager.update_job_status(
            job_id, 
            JobStatus.COMPLETED,
            result=result,
            progress=1.0
        )
        job_manager.add_job_log(job_id, "Background task completed successfully", "INFO")
        
    except Exception as e:
        # Update job with error
        error_msg = str(e)
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_msg
        )
        job_manager.add_job_log(job_id, f"Background task failed: {error_msg}", "ERROR")
        logger.error(f"Background task failed for job {job_id}: {error_msg}", exc_info=True)
        raise


def create_background_job(
    task_func: Callable,
    *args,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Create a background job and return the job ID."""
    job_id = job_manager.create_job(metadata)
    
    # Schedule the background task
    asyncio.create_task(run_background_task(task_func, job_id, *args, **kwargs))
    
    return job_id
