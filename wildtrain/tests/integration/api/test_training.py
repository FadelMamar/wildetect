"""
Training endpoint integration tests.
"""

import pytest


@pytest.mark.critical
@pytest.mark.training
class TestTrainingEndpoints:
    """Test training endpoints."""

    def test_train_experiment_with_real_config(self, client, classification_config):
        """Test POST /experiments with the shipped classification document."""
        classification_config["train"]["epochs"] = 1
        response = client.post(
            "/experiments",
            json={
                "config": classification_config,
            },
        )

        assert response.status_code in [200, 201, 202]  # 422 for validation errors
        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "success" in data
            assert "job_id" in data
            assert "status" in data

    def test_removed_training_routes(self, client, detection_config):
        """Old per-task training routes are gone."""
        response = client.post(
            "/training/detector",
            json={
                "config": detection_config,
            },
        )
        assert response.status_code == 404
        response = client.post(
            "/training/classifier",
            json={"config": detection_config},
        )
        assert response.status_code == 404
        gone = client.post(
            "/training/experiments",
            json={"config": detection_config},
        )
        assert gone.status_code == 404
