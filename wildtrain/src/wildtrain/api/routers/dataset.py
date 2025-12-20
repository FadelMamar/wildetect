"""Dataset endpoints for the WildTrain API."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..models.requests import DatasetStatsRequest
from ..models.responses import DatasetStatsResponse
from ..dependencies import get_logger
from ..services.dataset_service import DatasetService

logger = get_logger("dataset")

router = APIRouter()


@router.post("/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(request: DatasetStatsRequest) -> DatasetStatsResponse:
    """Get dataset statistics."""
    
    try:
        logger.info(f"Computing dataset statistics for {request.data_dir}")
        
        # Use the dataset service to run actual CLI dataset stats
        results = DatasetService.get_dataset_stats(
            data_dir=request.data_dir,
            split=request.split,
            output_file=request.output_file
        )
        
        return DatasetStatsResponse(
            success=True,
            message="Dataset statistics computed successfully",
            stats=results["stats"],
            class_distribution=results.get("class_distribution", {}),
            mean=results["stats"]["mean"],
            std=results["stats"]["std"],
            total_samples=results.get("total_samples", 0),
            split_info=results["split_info"]
        )
        
    except Exception as e:
        logger.error(f"Failed to compute dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/splits/{dataset_path:path}")
async def get_dataset_splits(dataset_path: str) -> Dict[str, Any]:
    """Get available dataset splits."""

    try:
        data_dir = Path(dataset_path)
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_path}")

        logger.info(f"Getting dataset splits for {dataset_path}")

        # Use the dataset service to get splits
        splits_info = DatasetService.get_dataset_splits(data_dir)

        return {
            "success": True,
            "message": f"Dataset splits retrieved for {dataset_path}",
            "dataset_path": dataset_path,
            "splits": splits_info["splits"],
            "total_splits": splits_info["total_splits"]
        }

    except Exception as e:
        logger.error(f"Failed to get dataset splits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

