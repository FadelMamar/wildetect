"""
Test configuration and fixtures for API tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from wildata.api.main import app
from wildata.api.models.jobs import JobResult, JobStatus
from wildata.api.models.requests import (
    BulkCreateROIRequest,
    BulkImportRequest,
    CreateROIRequest,
    ImportDatasetRequest,
    UpdateGPSRequest,
    VisualizeRequest,
)
from wildata.api.models.responses import (
    BulkCreateROIResponse,
    BulkImportResponse,
    CreateROIResponse,
    DatasetInfo,
    ImportDatasetResponse,
    JobStatusResponse,
    UpdateGPSResponse,
)
from wildata.config import ROOT


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_job_id():
    """Provide a mock job ID for testing."""
    return "test-job-12345"


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        test_dir = Path(temp_dir)
        (test_dir / "dataset1").mkdir()
        (test_dir / "dataset2").mkdir()
        (test_dir / "images").mkdir()
        (test_dir / "gps.csv").touch()
        (test_dir / "output").mkdir()

        yield str(test_dir)


@pytest.fixture
def mock_job():
    """Create a mock job with proper attributes."""
    from datetime import datetime

    job = Mock()
    job.job_id = "test-job-12345"
    job.status = JobStatus.COMPLETED
    job.progress = 100.0
    job.created_at = datetime.now()
    job.started_at = datetime.now()
    job.completed_at = datetime.now()
    job.job_type = "import_dataset"
    job.result = JobResult(
        success=True,
        message="Job completed successfully",
        data={"dataset_name": "test_dataset"},
    )
    return job


@pytest.fixture
def mock_jobs():
    """Create a list of mock jobs."""
    from datetime import datetime

    jobs = []
    for i in range(2):
        job = Mock()
        job.job_id = f"job{i+1}"
        job.status = JobStatus.COMPLETED if i == 0 else JobStatus.RUNNING
        job.progress = 100.0 if i == 0 else 50.0
        job.created_at = datetime.now()
        job.started_at = datetime.now()
        job.completed_at = datetime.now() if i == 0 else None
        job.job_type = "import_dataset" if i == 0 else "bulk_import"
        job.result = JobResult(
            success=True,
            message="Job completed successfully",
            data={"dataset_name": f"test_dataset_{i+1}"},
        )
        jobs.append(job)
    return jobs


# Test data for request models using real paths from config files
@pytest.fixture
def valid_import_request_data():
    """Valid import dataset request data using real paths."""
    return ImportDatasetRequest(
        source_path=r"D:\workspace\savmap\coco\annotations\train.json",
        source_format="coco",
        dataset_name="savmap",
        root=str(ROOT / "data" / "api-testing-dataset"),
        split_name="train",
        processing_mode="batch",
        track_with_dvc=False,
        bbox_tolerance=5,
        disable_roi=True,
    )


@pytest.fixture
def valid_bulk_import_request_data():
    """Valid bulk import request data using real paths."""
    return BulkImportRequest(
        source_paths=[
            r"D:\workspace\savmap\coco\annotations",  # directory
        ],
        source_format="coco",
        root=str(ROOT / "data" / "api-testing-dataset"),
        split_name="train",
        processing_mode="batch",
        track_with_dvc=False,
        bbox_tolerance=5,
        disable_roi=True,
    )


@pytest.fixture
def valid_create_roi_request_data():
    """Valid create ROI request data using real paths."""
    return CreateROIRequest(
        source_path=r"D:\workspace\data\general_dataset\tiled-data\coco-dataset\annotations\annotations_val.json",
        source_format="coco",
        dataset_name="general_dataset",
        root=str(ROOT / "data" / "api-testing-dataset"),
        split_name="val",
        bbox_tolerance=5,
        roi_config={
            "random_roi_count": 1,
            "roi_box_size": 384,
            "min_roi_size": 32,
            "dark_threshold": 0.7,
            "background_class": "background",
            "save_format": "jpg",
            "quality": 95,
            "sample_background": True,
        },
        draw_original_bboxes=False,
    )


@pytest.fixture
def valid_bulk_create_roi_request_data():
    """Valid bulk create ROI request data using real paths."""
    return BulkCreateROIRequest(
        source_paths=[
            r"D:\workspace\data\general_dataset\tiled-data\coco-dataset\annotations\annotations_train.json",
            r"D:\workspace\savmap\coco\annotations\train.json",
        ],
        source_format="ls",
        root=str(ROOT / "data" / "api-testing-dataset"),
        split_name="train",
        bbox_tolerance=5,
        roi_config={
            "random_roi_count": 1,
            "roi_box_size": 384,
            "min_roi_size": 32,
            "dark_threshold": 0.7,
            "background_class": "background",
            "save_format": "jpg",
            "sample_background": True,
            "quality": 95,
        },
        draw_original_bboxes=False,
    )


@pytest.fixture
def valid_update_gps_request_data(temp_test_dir):
    """Valid update GPS request data using real paths."""
    return UpdateGPSRequest(
        image_folder=r"D:\workspace\data\savmap_dataset_v2\images_splits",
        csv_path=str(ROOT / "examples" / "mock_csv.csv"),
        output_dir=str(ROOT / "data" / "savmap_dataset_v2_splits_with_gps"),
        skip_rows=0,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
    )


@pytest.fixture
def valid_export_request_data():
    """Valid export dataset request data."""
    return ExportDatasetRequest(
        dataset_name="savmap",
        target_format="coco",
        target_path=str(ROOT / "data" / "api-testing-dataset-export"),
        root=str(ROOT / "data" / "api-testing-dataset"),
    )


@pytest.fixture
def valid_visualize_request_data():
    """Valid visualize request data using real paths."""
    return VisualizeRequest(
        dataset_name="savmap",
        root_data_directory=str(ROOT / "data" / "api-testing-dataset"),
        split="train",
        load_as_single_class=False,
        background_class_name="background",
        single_class_name="wildlife",
        keep_classes=None,
        discard_classes=None,
    )


# Mock patches for common dependencies
@pytest.fixture(autouse=True)
def mock_common_dependencies():
    """Mock common dependencies used across tests."""
    with patch("wildata.api.routers.datasets.verify_token") as mock_verify_token, patch(
        "wildata.api.routers.jobs.verify_token"
    ) as mock_verify_token_jobs, patch(
        "wildata.api.routers.roi.verify_token"
    ) as mock_verify_token_roi, patch(
        "wildata.api.routers.gps.verify_token"
    ) as mock_verify_token_gps, patch(
        "wildata.api.routers.visualization.verify_token"
    ) as mock_verify_token_viz:
        # Mock token verification
        mock_verify_token.return_value = {
            "user_id": "anonymous",
            "username": "anonymous",
        }
        mock_verify_token_jobs.return_value = {
            "user_id": "anonymous",
            "username": "anonymous",
        }
        mock_verify_token_roi.return_value = {
            "user_id": "anonymous",
            "username": "anonymous",
        }
        mock_verify_token_gps.return_value = {
            "user_id": "anonymous",
            "username": "anonymous",
        }
        mock_verify_token_viz.return_value = {
            "user_id": "anonymous",
            "username": "anonymous",
        }

        yield {
            "verify_token": mock_verify_token,
        }


# Helper functions for creating valid response models
def create_job_status_response(
    job_id: str, status: JobStatus = JobStatus.COMPLETED
) -> JobStatusResponse:
    """Create a valid JobStatusResponse."""
    from datetime import datetime

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        progress=100.0,
        created_at=datetime.now(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
        job_type="import_dataset",
        result=JobResult(
            success=True,
            message="Job completed successfully",
            data={"dataset_name": "test_dataset"},
        ),
    )


def create_import_dataset_response(
    job_id: str = "test-job-123"
) -> ImportDatasetResponse:
    """Create a valid ImportDatasetResponse."""
    return ImportDatasetResponse(
        success=True,
        dataset_name="test_dataset",
        job_id=job_id,
        message="Dataset import job started",
    )


def create_bulk_import_response(job_id: str = "test-job-123") -> BulkImportResponse:
    """Create a valid BulkImportResponse."""
    return BulkImportResponse(
        success=True,
        job_id=job_id,
        total_datasets=2,
        message="Bulk import job started",
    )


def create_create_roi_response(job_id: str = "test-job-123") -> CreateROIResponse:
    """Create a valid CreateROIResponse."""
    return CreateROIResponse(
        success=True,
        dataset_name="test_roi_dataset",
        job_id=job_id,
        message="ROI dataset creation job started",
    )


def create_bulk_create_roi_response(
    job_id: str = "test-job-123"
) -> BulkCreateROIResponse:
    """Create a valid BulkCreateROIResponse."""
    return BulkCreateROIResponse(
        success=True,
        job_id=job_id,
        total_datasets=2,
        message="Bulk ROI creation job started",
    )


def create_export_dataset_response(
    job_id: str = "test-job-123"
) -> ExportDatasetResponse:
    """Create a valid ExportDatasetResponse."""
    return ExportDatasetResponse(
        success=True,
        dataset_name="test_dataset",
        target_format="coco",
        target_path="/path/to/export",
        job_id=job_id,
        message="Dataset export job started",
    )


def create_update_gps_response(job_id: str = "test-job-123") -> UpdateGPSResponse:
    """Create a valid UpdateGPSResponse."""
    return UpdateGPSResponse(
        success=True,
        job_id=job_id,
        message="GPS update job started",
        updated_images_count=10,
        output_dir="/path/to/output",
    )


def create_dataset_info(name: str = "test_dataset") -> DatasetInfo:
    """Create a valid DatasetInfo."""
    from datetime import datetime

    return DatasetInfo(
        dataset_name=name,
        total_images=100,
        total_annotations=500,
        splits=["train", "val"],
        created_at=datetime.now(),
        last_modified=datetime.now(),
    )
