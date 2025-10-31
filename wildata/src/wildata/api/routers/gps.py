"""
GPS update endpoints.
"""

import asyncio
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..dependencies import get_background_task_semaphore, verify_token
from ..exceptions import DatasetError, NotFoundError, ValidationError
from ..models.requests import (
    UpdateGPSRequest,
)
from ..models.responses import (
    UpdateGPSResponse,
)
from ..services.job_queue import get_job_queue
from ..services.task_handlers import (
    handle_update_gps,
)

router = APIRouter(prefix="/gps", tags=["gps"])


@router.post(
    "/update", response_model=UpdateGPSResponse, operation_id="update_images_gps"
)
async def update_gps_from_csv(
    request: UpdateGPSRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Update GPS data from CSV file."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="update_gps",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_update_gps():
            async with semaphore:
                await handle_update_gps(job.job_id, request)

        background_tasks.add_task(run_update_gps)

        return UpdateGPSResponse(
            success=True,
            job_id=job.job_id,
            message="GPS update started in background",
            updated_images_count=0,  # Will be updated by the handler
            output_dir=request.output_dir,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start GPS update: {str(e)}",
        )
