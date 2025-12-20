"""
Visualization endpoints.
"""

import asyncio
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..dependencies import get_background_task_semaphore, verify_token
from ..exceptions import DatasetError, NotFoundError, ValidationError
from ..models.requests import (
    VisualizeRequest,
)
from ..models.responses import (
    VisualizationResponse,
)
from ..services.job_queue import get_job_queue
from ..services.task_handlers import (
    handle_visualize_classification,
    handle_visualize_detection,
)

router = APIRouter(prefix="/visualize", tags=["visualization"])


@router.post(
    "/classification",
    response_model=VisualizationResponse,
    operation_id="visualize_classification_dataset",
)
async def visualize_classification_dataset(
    request: VisualizeRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Visualize a classification dataset."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="visualize_classification",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_visualize_classification():
            async with semaphore:
                await handle_visualize_classification(job.job_id, request)

        background_tasks.add_task(run_visualize_classification)

        return VisualizationResponse(
            success=True,
            job_id=job.job_id,
            message="Classification visualization started in background",
            dataset_name=request.dataset_name,
            split=request.split,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start classification visualization: {str(e)}",
        )


@router.post(
    "/detection",
    response_model=VisualizationResponse,
    operation_id="visualize_detection_dataset",
)
async def visualize_detection_dataset(
    request: VisualizeRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Visualize a detection dataset."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="visualize_detection",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_visualize_detection():
            async with semaphore:
                await handle_visualize_detection(job.job_id, request)

        background_tasks.add_task(run_visualize_detection)

        return VisualizationResponse(
            success=True,
            job_id=job.job_id,
            message="Detection visualization started in background",
            dataset_name=request.dataset_name,
            split=request.split,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start detection visualization: {str(e)}",
        )
