"""
ROI dataset management endpoints.
"""

import asyncio
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..dependencies import get_background_task_semaphore, verify_token
from ..exceptions import DatasetError, NotFoundError, ValidationError
from ..models.requests import (
    BulkCreateROIRequest,
    CreateROIRequest,
)
from ..models.responses import (
    BulkCreateROIResponse,
    CreateROIResponse,
    DatasetInfo,
)
from ..services.job_queue import get_job_queue
from ..services.task_handlers import (
    handle_bulk_create_roi,
    handle_create_roi,
)

router = APIRouter(prefix="/roi", tags=["roi"])


@router.post(
    "/create", response_model=CreateROIResponse, operation_id="create_roi_dataset"
)
async def create_roi_dataset(
    request: CreateROIRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Create a single ROI dataset."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="create_roi",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_create_roi():
            async with semaphore:
                await handle_create_roi(job.job_id, request)

        background_tasks.add_task(run_create_roi)

        return CreateROIResponse(
            success=True,
            dataset_name=request.dataset_name,
            job_id=job.job_id,
            message="ROI dataset creation started in background",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ROI creation: {str(e)}",
        )


@router.post(
    "/create/bulk",
    response_model=BulkCreateROIResponse,
    operation_id="bulk_create_roi_dataset",
)
async def bulk_create_roi_datasets(
    request: BulkCreateROIRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Bulk create multiple ROI datasets."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="bulk_create_roi",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_bulk_create_roi():
            async with semaphore:
                await handle_bulk_create_roi(job.job_id, request)

        background_tasks.add_task(run_bulk_create_roi)

        return BulkCreateROIResponse(
            success=True,
            job_id=job.job_id,
            total_datasets=len(request.source_paths),
            message="Bulk ROI creation started in background",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk ROI creation: {str(e)}",
        )


@router.get("/{dataset_name}", operation_id="get_roi_dataset_info")
async def get_roi_dataset_info(
    dataset_name: str, root: str = "data", user=Depends(verify_token)
):
    """Get information about a specific ROI dataset."""
    try:
        from ...pipeline import DataPipeline

        pipeline = DataPipeline(root=root, split_name="train")
        datasets = pipeline.list_datasets()

        # Find the specific ROI dataset
        dataset = None
        for d in datasets:
            if (
                d.get("dataset_name") == dataset_name
                and "roi" in d.get("dataset_name", "").lower()
            ):
                dataset = d
                break

        if not dataset:
            raise NotFoundError(
                f"ROI dataset '{dataset_name}' not found", resource_type="roi_dataset"
            )

        return {
            "dataset_name": dataset.get("dataset_name"),
            "total_images": dataset.get("total_images"),
            "total_annotations": dataset.get("total_annotations"),
            "splits": dataset.get("splits"),
            "root_directory": root,
            "dataset_type": "roi",
        }

    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ROI dataset info: {str(e)}",
        )
