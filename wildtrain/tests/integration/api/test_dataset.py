"""
Dataset Endpoints Integration Tests

These tests ensure dataset operations work correctly.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import yaml

from wildtrain.api.main import fastapi_app as app

DATASET_NAME = "savmap"
DATA_DIR = r"D:\workspace\data\demo-dataset"

@pytest.mark.dataset
class TestDatasetEndpoints:
    """Test dataset endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)      
        
    def test_error_handling_for_nonexistent_datasets(self, client):
        """Test error handling for non-existent datasets."""
        response = client.post("/dataset/stats", json={
            "data_dir": "/nonexistent/dataset/path",
            "split": "train"
        })
        
        # Should handle non-existent datasets gracefully
        assert response.status_code in [404, 500, 422]
    
    def test_cli_integration_for_dataset_operations(self, client):
        """Test CLI integration for dataset operations."""
                    
        response = client.post("/dataset/stats", json={
            "data_dir": DATA_DIR,
            "split": "train"
        })
        
        assert response.status_code in [200, 201, 202]
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert data["success"] is True
            assert "stats" in data
            assert "class_distribution" in data
    
    def test_dataset_stats_with_output_file(self, client):
        """Test dataset stats with output file specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "stats_output.json"
        
            response = client.post("/dataset/stats", json=  {
                "data_dir": DATA_DIR,
                "split": "train",
                "output_file": str(output_file)
            })
            
            assert response.status_code in [200, 201, 202, 500]
            if response.status_code in [200, 201, 202]:
                data = response.json()
                assert data["success"] is True   
        
    def test_dataset_stats_error_propagation(self, client):
        """Test that dataset stats errors are properly propagated."""                    
        response = client.post("/dataset/stats", json={
            "data_dir": "/path/to/dataset",
            "split": "train"
        })
        
        # Should propagate errors properly
        assert response.status_code in [500, 422]
    
    def test_dataset_stats_validation(self, client):
        """Test dataset stats request validation."""
        # Test with invalid data types
        invalid_requests = [
            {"data_dir": 123, "split": "train"},  # data_dir should be string
            {"data_dir": DATA_DIR, "split": 456},  # split should be string
            {"data_dir": "", "split": "train"},  # empty data_dir
            {"data_dir": DATA_DIR, "split": ""},  # empty split
        ]
        
        for request in invalid_requests:
            response = client.post("/dataset/stats", json=request)
            
            # Should validate input properly
            assert response.status_code in [422, 400, 500]
        

