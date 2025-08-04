"""
Tests for the WildDetect API integration.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from wildetect.api.main import app

# Create test client
client = TestClient(app)


def test_api_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "WildDetect API"
    assert "version" in data
    assert "endpoints" in data


def test_api_info():
    """Test the info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "pytorch_version" in data
    assert "cuda_available" in data
    assert "dependencies" in data
    assert isinstance(data["dependencies"], dict)


def test_api_upload_no_files():
    """Test upload endpoint with no files."""
    response = client.post("/upload", files=[])
    assert response.status_code == 422  # Validation error


def test_api_detect_request():
    """Test detection endpoint with valid request."""
    detection_request = {
        "confidence": 0.3,
        "device": "auto",
        "batch_size": 16,
        "tile_size": 800,
        "output": "test_results",
        "pipeline_type": "single",
    }

    response = client.post("/detect", json=detection_request)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "running"


def test_api_census_request():
    """Test census endpoint with valid request."""
    census_request = {
        "campaign_id": "test_campaign",
        "confidence": 0.3,
        "device": "auto",
        "batch_size": 16,
        "tile_size": 800,
        "output": "test_census_results",
        "pilot_name": "Test Pilot",
        "target_species": ["deer", "elk"],
        "export_to_fiftyone": True,
        "create_map": True,
        "pipeline_type": "single",
    }

    response = client.post("/census", json=census_request)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "running"
    assert "campaign_id" in data
    assert data["campaign_id"] == "test_campaign"


def test_api_job_status():
    """Test job status endpoint."""
    # First create a job
    detection_request = {
        "confidence": 0.3,
        "device": "auto",
        "batch_size": 16,
        "tile_size": 800,
        "output": "test_results",
        "pipeline_type": "single",
    }

    response = client.post("/detect", json=detection_request)
    job_id = response.json()["job_id"]

    # Check job status
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data
    assert "progress" in data


def test_api_job_status_not_found():
    """Test job status endpoint with non-existent job."""
    response = client.get("/jobs/nonexistent")
    assert response.status_code == 404


def test_api_visualize_invalid_path():
    """Test visualization endpoint with invalid results path."""
    response = client.post(
        "/visualize",
        data={"results_path": "nonexistent.json"},
        json={"show_confidence": True},
    )
    assert response.status_code == 404


def test_api_analyze_invalid_path():
    """Test analysis endpoint with invalid results path."""
    response = client.post(
        "/analyze", data={"results_path": "nonexistent.json"}, json={"create_map": True}
    )
    assert response.status_code == 404


def test_api_fiftyone_launch():
    """Test FiftyOne launch endpoint."""
    with patch(
        "wildetect.core.visualization.fiftyone_manager.FiftyOneManager.launch_app"
    ) as mock_launch:
        mock_launch.return_value = None
        response = client.get("/fiftyone/launch")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"]


def test_api_fiftyone_dataset_info():
    """Test FiftyOne dataset info endpoint."""
    with patch(
        "wildetect.core.visualization.fiftyone_manager.FiftyOneManager"
    ) as mock_manager:
        mock_instance = MagicMock()
        mock_instance.get_dataset_info.return_value = {
            "name": "test_dataset",
            "num_samples": 100,
            "fields": ["filepath", "predictions"],
        }
        mock_instance.get_annotation_stats.return_value = {
            "annotated_samples": 50,
            "total_detections": 200,
            "annotation_rate": 50.0,
        }
        mock_manager.return_value = mock_instance

        response = client.get("/fiftyone/datasets/test_dataset")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_info" in data
        assert "annotation_stats" in data


def test_api_export_dataset():
    """Test FiftyOne export endpoint."""
    with patch(
        "wildetect.core.visualization.fiftyone_manager.FiftyOneManager"
    ) as mock_manager:
        mock_instance = MagicMock()
        mock_instance.dataset = MagicMock()
        mock_manager.return_value = mock_instance

        response = client.post(
            "/fiftyone/export/test_dataset",
            data={"export_format": "coco", "export_path": "test_export"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "export_path" in data


def test_api_export_dataset_no_dataset():
    """Test FiftyOne export endpoint with no dataset."""
    with patch(
        "wildetect.core.visualization.fiftyone_manager.FiftyOneManager"
    ) as mock_manager:
        mock_instance = MagicMock()
        mock_instance.dataset = None
        mock_manager.return_value = mock_instance

        response = client.post(
            "/fiftyone/export/test_dataset", data={"export_format": "coco"}
        )
        assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
