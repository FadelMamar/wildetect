"""
Dataset management endpoints.
"""

import asyncio
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..dependencies import get_background_task_semaphore, verify_token
from ..exceptions import DatasetError, NotFoundError, ValidationError
from ..models.requests import (
    BulkImportRequest,
    ImportDatasetRequest,
)
from ..models.responses import (
    BulkImportResponse,
    DatasetInfo,
    DatasetListResponse,
    ImportDatasetResponse,
)
from ..services.job_queue import get_job_queue
from ..services.task_handlers import (
    handle_bulk_import,
    handle_import_dataset,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/import", response_model=ImportDatasetResponse, operation_id="import_dataset"
)
async def import_dataset(
    request: ImportDatasetRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Import a single dataset."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="import_dataset",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_import():
            async with semaphore:
                await handle_import_dataset(job.job_id, request)

        background_tasks.add_task(run_import)

        return ImportDatasetResponse(
            success=True,
            dataset_name=request.dataset_name,
            job_id=job.job_id,
            message="Dataset import started in background",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start import: {str(e)}",
        )


@router.post(
    "/import/bulk",
    response_model=BulkImportResponse,
    operation_id="bulk_import_dataset",
)
async def bulk_import_datasets(
    request: BulkImportRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
    semaphore=Depends(get_background_task_semaphore),
):
    """Bulk import multiple datasets."""
    try:
        # Create background job
        job_queue = get_job_queue()
        job = await job_queue.create_job(
            job_type="bulk_import",
            parameters=request.model_dump(),
            user_id=user.get("user_id"),
        )

        # Add background task
        async def run_bulk_import():
            async with semaphore:
                await handle_bulk_import(job.job_id, request)

        background_tasks.add_task(run_bulk_import)

        return BulkImportResponse(
            success=True,
            job_id=job.job_id,
            total_datasets=len(request.source_paths),
            message="Bulk import started in background",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk import: {str(e)}",
        )


@router.get("/", response_model=DatasetListResponse, operation_id="list_datasets")
async def list_datasets(root: str = "data", user=Depends(verify_token)):
    """List all available datasets."""
    try:
        from ...pipeline import DataPipeline

        pipeline = DataPipeline(root=root, split_name="train")
        datasets = pipeline.list_datasets()

        dataset_infos = []
        for dataset in datasets:
            dataset_info = DatasetInfo(
                dataset_name=dataset.get("dataset_name", "Unknown"),
                total_images=dataset.get("total_images"),
                total_annotations=dataset.get("total_annotations"),
                splits=dataset.get("splits"),
            )
            dataset_infos.append(dataset_info)

        return DatasetListResponse(
            datasets=dataset_infos, total_count=len(dataset_infos), root_directory=root
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}",
        )


@router.get("/{dataset_name}", operation_id="get_dataset_info")
async def get_dataset_info(
    dataset_name: str, root: str = "data", user=Depends(verify_token)
):
    """Get information about a specific dataset."""
    try:
        from ...pipeline import DataPipeline

        pipeline = DataPipeline(root=root, split_name="train")
        datasets = pipeline.list_datasets()

        # Find the specific dataset
        dataset = None
        for d in datasets:
            if d.get("dataset_name") == dataset_name:
                dataset = d
                break

        if not dataset:
            raise NotFoundError(
                f"Dataset '{dataset_name}' not found", resource_type="dataset"
            )

        return {
            "dataset_name": dataset.get("dataset_name"),
            "total_images": dataset.get("total_images"),
            "total_annotations": dataset.get("total_annotations"),
            "splits": dataset.get("splits"),
            "root_directory": root,
        }

    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset info: {str(e)}",
        )
