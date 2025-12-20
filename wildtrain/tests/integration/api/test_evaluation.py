"""
Evaluation endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.evaluation
class TestEvaluationEndpoints:
    """Test evaluation endpoints."""
    
    def test_evaluate_classifier_with_real_config(self, client, classification_eval_config):
        """Test POST /evaluation/classifier endpoint with real config."""
        response = client.post("/evaluation/classifier", json={
            "config": classification_eval_config,     
            "debug": True,
        })
        
        assert response.status_code in [200, 201, 202]  # 422 for validation errors
        data = response.json()
        assert "success" in data
        assert "job_id" in data
    
    def test_evaluate_detector_with_real_config(self, client, detection_eval_config):
        """Test POST /evaluation/detector endpoint with real config."""
        response = client.post("/evaluation/detector", json={
            "config": detection_eval_config,
            "debug": True,
            "model_type": "yolo",
        })
        
        assert response.status_code in [200, 201, 202]  # 422 for validation errors
        data = response.json()
        assert "success" in data or "job_id" in data  # Check for either success or job_id
        if "job_id" in data:
            assert "message" in data  # Should have a message field
        
    def test_get_evaluation_status(self, client):
        """Test GET /evaluation/status/{job_id} endpoint."""
        job_id = "test-eval-job-123"
        response = client.get(f"/evaluation/status/{job_id}")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_list_evaluation_jobs(self, client):
        """Test GET /evaluation/jobs endpoint."""
        response = client.get("/evaluation/jobs")
        
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data or isinstance(data, list)
        
    def test_invalid_evaluation_config(self, client):
        """Test evaluation with invalid config."""
        response = client.post("/evaluation/classifier", json={
            "config": {"invalid": "config"},            
        })
        assert response.status_code in [422, 400]  # Validation error

