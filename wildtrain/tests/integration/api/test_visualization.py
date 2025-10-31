"""
Visualization endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.visualization
class TestVisualizationEndpoints:
    """Test visualization endpoints."""
    
    def test_visualize_classifier_with_real_config(self, client, classification_visualization_config):
        """Test POST /visualization/classifier endpoint with real config."""
        response = client.post("/visualization/classifier", json={
            "config": classification_visualization_config,
        })
        
        assert response.status_code in [200, 201, 202], f"Response: {response.json()}"
        data = response.json()
        assert "success" in data
        assert "job_id" in data
        assert "status" in data
    
    def test_visualize_detector_with_real_config(self, client, detection_visualization_config):
        """Test POST /visualization/detector endpoint with real config."""
        response = client.post("/visualization/detector", json={
            "config": detection_visualization_config,
        })
        
        assert response.status_code in [200, 201, 202], f"Response: {response.json()}"
        data = response.json()
        assert "success" in data or "job_id" in data
        if "job_id" in data:
            assert "message" in data
        
    def test_list_visualization_datasets(self, client):
        """Test GET /visualization/datasets endpoint."""
        response = client.get("/visualization/datasets")
        
        assert response.status_code in [200,], f"Response: {response.json()}"
        if response.status_code == 200:
            data = response.json()
            assert "datasets" in data or isinstance(data, list)
    
    def test_get_dataset_info(self, client):
        """Test GET /visualization/datasets/{dataset_name} endpoint."""
        dataset_name = "savmap-train"
        response = client.get(f"/visualization/datasets/{dataset_name}")
        
        assert response.status_code in [200,], f"Response: {response.json()}"
        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "info" in data
        else:
            print(response.json())
    
    def test_get_visualization_status(self, client):
        """Test GET /visualization/status/{job_id} endpoint."""
        job_id = "test-viz-job-123"
        response = client.get(f"/visualization/status/{job_id}")
        
        assert response.status_code in [200, 404], f"Response: {response.json()}"
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_invalid_visualization_config(self, client):
        """Test visualization with invalid config."""
        response = client.post("/visualization/classifier", json={
            "config": {"invalid": "config"},
            "debug": True,
            "verbose": False,
            
        })
        
        assert response.status_code in [422, 400], f"Response: {response.json()}"
    
