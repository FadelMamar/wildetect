"""
Pipeline Endpoints Integration Tests

These tests ensure pipeline operations work correctly.
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

@pytest.mark.critical
@pytest.mark.pipeline
class TestPipelineEndpoints:
    """Test pipeline endpoints."""
    
    
    def test_post_pipeline_classification_runs_full_pipeline(self, client, classification_pipeline_config):
        """Test POST /pipeline/classification runs full pipeline."""        
        response = client.post("/pipeline/classification", json={
            "config": classification_pipeline_config
        })
        
        assert response.status_code in [200, 201, 202,]
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
    
    def test_post_pipeline_detection_runs_full_pipeline(self, client, detection_pipeline_config):
        """Test POST /pipeline/detection runs full pipeline."""
        
        response = client.post("/pipeline/detection", json={
            "config": detection_pipeline_config
        })
        
        assert response.status_code in [200, 201, 202]
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
