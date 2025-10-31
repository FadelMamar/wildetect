"""
Tests for API endpoints.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from wildata.api.models.jobs import JobResult, JobStatus
from wildata.api.models.responses import (
    BulkCreateROIResponse,
    BulkImportResponse,
    CreateROIResponse,
    DatasetInfo,
    ImportDatasetResponse,
    JobStatusResponse,
    UpdateGPSResponse,
)


# Helper functions for creating valid response models
def create_job_status_response(
    job_id: str, status: JobStatus = JobStatus.COMPLETED
) -> JobStatusResponse:
    """Create a valid JobStatusResponse."""
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
    return DatasetInfo(
        dataset_name=name,
        total_images=100,
        total_annotations=500,
        splits=["train", "val"],
        created_at=datetime.now(),
        last_modified=datetime.now(),
    )


class TestDatasetEndpoints:
    """Test dataset-related endpoints."""

    def test_import_dataset_endpoint(self, test_client, valid_import_request_data):
        """Test import dataset endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_import_request_data.model_dump()
        response = test_client.post("/api/v1/datasets/import", json=request_data)

        # The import might complete synchronously, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["dataset_name"] == "savmap"
        assert "job_id" in data

    def test_bulk_import_dataset_endpoint(
        self, test_client, valid_bulk_import_request_data
    ):
        """Test bulk import dataset endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_bulk_import_request_data.model_dump()
        response = test_client.post("/api/v1/datasets/import/bulk", json=request_data)

        print(response.json())

        # The bulk import might complete synchronously, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["total_datasets"] == 1
        assert "job_id" in data

    def test_get_dataset_info_endpoint(self, test_client):
        """Test get dataset info endpoint."""
        response = test_client.get("/api/v1/datasets/test_dataset")
        # This might return 404 if dataset doesn't exist, which is expected
        assert response.status_code in [200, 404]


class TestROIEndpoints:
    """Test ROI-related endpoints."""

    def test_create_roi_endpoint(self, test_client, valid_create_roi_request_data):
        """Test create ROI endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_create_roi_request_data.model_dump()
        response = test_client.post("/api/v1/roi/create", json=request_data)
        # ROI creation might be sync, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["dataset_name"] == "general_dataset"
        assert "job_id" in data

    def test_bulk_create_roi_endpoint(
        self, test_client, valid_bulk_create_roi_request_data
    ):
        """Test bulk create ROI endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_bulk_create_roi_request_data.model_dump()
        response = test_client.post("/api/v1/roi/create/bulk", json=request_data)
        # Bulk ROI creation might be sync, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["total_datasets"] == 2
        assert "job_id" in data

    def test_get_roi_dataset_info_endpoint(self, test_client):
        """Test get ROI dataset info endpoint."""
        response = test_client.get("/api/v1/roi/test_roi_dataset")
        # This might return 404 if dataset doesn't exist, which is expected
        assert response.status_code in [200, 404]


class TestGPSEndpoints:
    """Test GPS-related endpoints."""

    def test_update_gps_endpoint(self, test_client, valid_update_gps_request_data):
        """Test update GPS endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_update_gps_request_data.model_dump()
        response = test_client.post("/api/v1/gps/update", json=request_data)

        # GPS update might complete synchronously, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "output_dir" in data


class TestVisualizationEndpoints:
    """Test visualization endpoints."""

    def test_visualize_classification_endpoint(
        self, test_client, valid_visualize_request_data
    ):
        """Test visualize classification endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_visualize_request_data.model_dump()
        response = test_client.post(
            "/api/v1/visualize/classification", json=request_data
        )
        # Visualization might be sync, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["dataset_name"] == "savmap"  # Match the fixture dataset name
        assert "job_id" in data

    def test_visualize_detection_endpoint(
        self, test_client, valid_visualize_request_data
    ):
        """Test visualize detection endpoint."""
        # Convert Pydantic model to dict for API call
        request_data = valid_visualize_request_data.model_dump()
        response = test_client.post("/api/v1/visualize/detection", json=request_data)
        # Visualization might be sync, so check for either 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["success"] is True
        assert data["dataset_name"] == "savmap"  # Match the fixture dataset name
        assert "job_id" in data


class TestJobEndpoints:
    """Test job-related endpoints."""

    def test_get_job_status_endpoint(self, test_client, mock_job_id: str):
        """Test get job status endpoint."""
        response = test_client.get(f"/api/v1/jobs/{mock_job_id}")
        # This might return 404 if job doesn't exist, which is expected
        assert response.status_code in [200, 404]

    def test_list_jobs_endpoint(self, test_client):
        """Test list jobs endpoint."""
        response = test_client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        # Should return a list (might be empty)
        assert isinstance(data, list)


class TestErrorHandling:
    """Test error handling."""

    def test_job_not_found(self, test_client, mock_job_id: str):
        """Test handling of job not found."""
        response = test_client.get(f"/api/v1/jobs/{mock_job_id}")
        assert response.status_code == 404
        data = response.json()
        assert "error_code" in data

    def test_invalid_request_data(self, test_client):
        """Test handling of invalid request data."""
        invalid_data = {
            "source_path": "/nonexistent/path",
            "source_format": "invalid_format",
            "dataset_name": "test_dataset",
        }

        response = test_client.post("/api/v1/datasets/import", json=invalid_data)
        assert response.status_code == 422

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        incomplete_data = {
            "source_format": "coco",
            # Missing source_path and dataset_name
        }

        response = test_client.post("/api/v1/datasets/import", json=incomplete_data)
        assert response.status_code == 422


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
