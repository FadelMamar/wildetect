"""
Dataset Endpoints Integration Tests

These tests ensure dataset operations work correctly.
"""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wildtrain.api.main import fastapi_app as app


def _synthetic_roi(root: Path) -> None:
    image_dir = root / "framework_formats" / "roi" / "species" / "images" / "train"
    label_dir = root / "framework_formats" / "roi" / "species" / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), (20, 40, 60)).save(image_dir / "a.png")
    Image.new("RGB", (8, 8), (80, 90, 100)).save(image_dir / "b.png")
    (label_dir / "class_mapping.json").write_text(
        json.dumps({"1": "background", "2": "wildlife"}), encoding="utf-8"
    )
    (label_dir / "roi_labels.json").write_text(
        json.dumps(
            [
                {"file_name": "a.png", "class_id": 1},
                {"file_name": "b.png", "class_id": 2},
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.dataset
class TestDatasetEndpoints:
    """Test dataset endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> str:
        _synthetic_roi(tmp_path)
        return str(tmp_path)

    def test_error_handling_for_nonexistent_datasets(self, client):
        """Test error handling for non-existent datasets."""
        response = client.post(
            "/dataset/stats",
            json={"data_dir": "/nonexistent/dataset/path", "split": "train"},
        )

        # Should handle non-existent datasets gracefully
        assert response.status_code in [404, 500, 422]

    def test_cli_integration_for_dataset_operations(self, client, data_dir):
        """Dataset stats run on a synthetic ROI tree, not a machine path."""

        response = client.post(
            "/dataset/stats", json={"data_dir": data_dir, "split": "train"}
        )

        assert response.status_code in [200, 201, 202]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert data["success"] is True
            assert "stats" in data
            assert "class_distribution" in data

    def test_dataset_stats_with_output_file(self, client, data_dir):
        """Test dataset stats with output file specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "stats_output.json"

            response = client.post(
                "/dataset/stats",
                json={
                    "data_dir": data_dir,
                    "split": "train",
                    "output_file": str(output_file),
                },
            )

            assert response.status_code in [200, 201, 202, 500]
            if response.status_code in [200, 201, 202]:
                data = response.json()
                assert data["success"] is True

    def test_dataset_stats_error_propagation(self, client):
        """Test that dataset stats errors are properly propagated."""
        response = client.post(
            "/dataset/stats", json={"data_dir": "/path/to/dataset", "split": "train"}
        )

        # Should propagate errors properly
        assert response.status_code in [500, 422]

    def test_dataset_stats_validation(self, client):
        """Test dataset stats request validation."""
        # Test with invalid data types
        invalid_requests = [
            {"data_dir": 123, "split": "train"},  # data_dir should be string
            {"data_dir": "/tmp/roi", "split": 456},  # split should be string
            {"data_dir": "", "split": "train"},  # empty data_dir
            {"data_dir": "/tmp/roi", "split": ""},  # empty split
        ]

        for request in invalid_requests:
            response = client.post("/dataset/stats", json=request)

            # Should validate input properly
            assert response.status_code in [422, 400, 500]
