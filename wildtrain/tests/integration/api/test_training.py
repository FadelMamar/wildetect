"""
Training endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.training
class TestTrainingEndpoints:
    """Test training endpoints."""
    
    def test_train_classifier_with_real_config(self, client, classification_config):
        """Test POST /training/classifier endpoint with real config."""
        classification_config["train"]["epochs"] = 1
        response = client.post("/training/classifier", json={
            "config": classification_config,
        })
        
        assert response.status_code in [200, 201, 202]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data
    
    def test_train_detector_with_real_config(self, client, detection_config):
        """Test POST /training/detector endpoint with real config."""
        detection_config["train"]["epochs"] = 1
        print("detection_config:",detection_config)
        response = client.post("/training/detector", json={
            "config": detection_config,            
        })
        
        assert response.status_code in [200, 201, 202]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data
        