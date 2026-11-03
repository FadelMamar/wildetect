"""Train-then-eval is POST /experiments with eval.enabled."""

import pytest


@pytest.mark.critical
@pytest.mark.pipeline
class TestPipelineEndpoints:
    """The old pipeline routes are gone. Experiments own train-then-eval."""

    def test_post_pipeline_classification_runs_full_pipeline(
        self, client, classification_config
    ):
        """An experiment with eval.enabled creates one training job."""
        classification_config["eval"]["enabled"] = True
        response = client.post(
            "/experiments",
            json={"config": classification_config, "debug": True},
        )

        assert response.status_code in [200, 201, 202]
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        removed = client.post(
            "/pipeline/classification",
            json={"config": classification_config},
        )
        assert removed.status_code == 404

    def test_post_pipeline_detection_runs_full_pipeline(
        self, client, detection_config
    ):
        """Detection uses the same POST /experiments route."""
        detection_config["eval"]["enabled"] = True
        response = client.post(
            "/experiments",
            json={"config": detection_config, "debug": True},
        )

        assert response.status_code in [200, 201, 202]
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        removed = client.post(
            "/pipeline/detection",
            json={"config": detection_config},
        )
        assert removed.status_code == 404
