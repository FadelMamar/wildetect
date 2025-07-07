"""
Tests for CLI-UI Integration using subprocesses.

This module tests the integration between CLI and UI functionality using subprocess calls.
"""

import json
import os
import random
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from wildetect.cli_ui_integration import CLIUIIntegration

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"


def load_image_path():
    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path


class TestCLIUIIntegration:
    """Test CLI-UI integration functionality using subprocesses."""

    def setup_method(self):
        """Set up test fixtures."""
        self.integration = CLIUIIntegration()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integration_initialization(self):
        """Test that the integration can be initialized."""
        assert self.integration is not None
        assert hasattr(self.integration, "_run_cli_command")
        assert hasattr(self.integration, "run_detection_ui")
        assert hasattr(self.integration, "run_census_ui")
        assert hasattr(self.integration, "analyze_results_ui")
        assert hasattr(self.integration, "visualize_results_ui")
        assert hasattr(self.integration, "get_system_info_ui")
        assert hasattr(self.integration, "clear_results_ui")

    @patch("subprocess.Popen")
    def test_run_cli_command_success(self, mock_popen):
        """Test successful CLI command execution."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "stderr")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = self.integration._run_cli_command("detect", ["image.jpg"])

        assert result["success"] is True
        assert result["stdout"] == "stdout"
        assert result["stderr"] == "stderr"
        assert result["return_code"] == 0

    @patch("subprocess.Popen")
    def test_run_cli_command_failure(self, mock_popen):
        """Test failed CLI command execution."""
        # Mock failed subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 1
        mock_process.communicate.return_value = ("stdout", "error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        result = self.integration._run_cli_command("detect", ["image.jpg"])

        assert result["success"] is False
        assert "error message" in result["error"]
        assert result["return_code"] == 1

    @patch("subprocess.Popen")
    def test_system_info_ui(self, mock_popen):
        """Test system information retrieval using subprocess."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = (
            "Component | Status | Details\nPyTorch | ✓ | Version 2.0.0\nCUDA | ✓ | Available",
            "",
        )
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        system_info = self.integration.get_system_info_ui()

        assert system_info["success"] is True
        assert "system_info" in system_info
        assert "components" in system_info["system_info"]
        assert "dependencies" in system_info["system_info"]
        assert "timestamp" in system_info["system_info"]

    @patch("subprocess.Popen")
    def test_run_detection_ui_success(self, mock_popen):
        """Test successful detection from UI."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create temporary directory with mock results
        results_file = Path(self.temp_dir) / "results.json"
        mock_results = {
            "drone_images": [{"total_detections": 5}, {"total_detections": 3}]
        }
        with open(results_file, "w") as f:
            json.dump(mock_results, f)

        result = self.integration.run_detection_ui(
            images=["image1.jpg", "image2.jpg"], output=self.temp_dir
        )

        assert result["success"] is True
        assert result["total_images"] == 2
        assert result["total_detections"] == 8

    @patch("subprocess.Popen")
    def test_run_census_ui_success(self, mock_popen):
        """Test successful census campaign from UI."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create temporary directory with mock campaign results
        campaign_file = Path(self.temp_dir) / "campaign_report.json"
        mock_campaign = {
            "campaign_id": "test_campaign",
            "status": "completed",
            "statistics": {"total_images": 10},
        }
        with open(campaign_file, "w") as f:
            json.dump(mock_campaign, f)

        result = self.integration.run_census_ui(
            campaign_id="test_campaign", images=["image1.jpg"], output=self.temp_dir
        )

        assert result["success"] is True
        assert result["campaign_id"] == "test_campaign"
        assert "results" in result

    @patch("subprocess.Popen")
    def test_analyze_results_ui_success(self, mock_popen):
        """Test successful analysis from UI."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create temporary directory with mock analysis results
        analysis_file = Path(self.temp_dir) / "analysis_report.json"
        mock_analysis = {
            "total_images": 5,
            "total_detections": 15,
            "species_breakdown": {"elephant": 10, "giraffe": 5},
        }
        with open(analysis_file, "w") as f:
            json.dump(mock_analysis, f)

        result = self.integration.analyze_results_ui(
            results_path="results.json", output_dir=self.temp_dir
        )

        assert result["success"] is True
        assert "analysis_results" in result

    @patch("subprocess.Popen")
    def test_visualize_results_ui_success(self, mock_popen):
        """Test successful visualization from UI."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create temporary directory with mock visualization results
        viz_file = Path(self.temp_dir) / "visualization_report.json"
        mock_viz = {
            "total_images": 5,
            "total_detections": 15,
            "species_counts": {"elephant": 10, "giraffe": 5},
        }
        with open(viz_file, "w") as f:
            json.dump(mock_viz, f)

        result = self.integration.visualize_results_ui(
            results_path="results.json", output_dir=self.temp_dir
        )

        assert result["success"] is True
        assert "visualization_data" in result

    def test_parse_system_info(self):
        """Test parsing of system information from CLI output."""
        stdout = """
Component | Status | Details
PyTorch | ✓ | Version 2.0.0
CUDA | ✓ | Available - RTX 3080
PIL | ✓ | Installed
numpy | ✓ | Installed
        """.strip()

        system_info = self.integration._parse_system_info(stdout)

        assert "components" in system_info
        assert "dependencies" in system_info
        assert "PyTorch" in system_info["components"]
        assert "CUDA" in system_info["components"]
        assert system_info["components"]["PyTorch"]["status"] == "✓"

    def test_load_campaign_results(self):
        """Test loading campaign results from file."""
        campaign_file = Path(self.temp_dir) / "campaign_report.json"
        mock_campaign = {"campaign_id": "test_campaign", "status": "completed"}
        with open(campaign_file, "w") as f:
            json.dump(mock_campaign, f)

        result = self.integration._load_campaign_results(self.temp_dir, "test_campaign")
        assert result["campaign_id"] == "test_campaign"
        assert result["status"] == "completed"

    def test_clear_results_ui(self):
        """Test clearing results from UI."""
        with patch.object(self.integration, "_run_cli_command") as mock_run:
            mock_run.return_value = {"success": True}

            result = self.integration.clear_results_ui("results")

            assert result["success"] is True
            assert "cleared successfully" in result["message"]
            mock_run.assert_called_once_with(
                "clear-results", ["results"], progress_bar=None, status_text=None
            )

    def test_error_handling(self):
        """Test error handling in subprocess execution."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = Exception("Subprocess error")

            result = self.integration._run_cli_command("detect", ["image.jpg"])

            assert result["success"] is False
            assert "Subprocess error" in result["error"]
            assert result["return_code"] == -1

    @patch("subprocess.Popen")
    def test_run_detection_ui_with_temp_output(self, mock_popen):
        """Test detection with temporary output directory."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create a temporary directory with mock results
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / "results.json"
            mock_results = {"drone_images": [{"total_detections": 5}]}
            with open(results_file, "w") as f:
                json.dump(mock_results, f)

            # Mock the tempfile.mkdtemp to return our temp_dir
            with patch("tempfile.mkdtemp", return_value=temp_dir):
                result = self.integration.run_detection_ui(images=["image1.jpg"])

                assert result["success"] is True
                assert "output_dir" in result

    @patch("subprocess.Popen")
    def test_run_census_ui_with_optional_params(self, mock_popen):
        """Test census with optional parameters."""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create temporary directory with mock campaign results
        campaign_file = Path(self.temp_dir) / "campaign_report.json"
        mock_campaign = {"campaign_id": "test_campaign", "status": "completed"}
        with open(campaign_file, "w") as f:
            json.dump(mock_campaign, f)

        result = self.integration.run_census_ui(
            campaign_id="test_campaign",
            images=["image1.jpg"],
            output=self.temp_dir,
            pilot_name="Test Pilot",
            target_species=["elephant", "giraffe"],
            create_map=False,
        )

        assert result["success"] is True
        assert result["campaign_id"] == "test_campaign"


def test_integration_import():
    """Test that the integration can be imported and instantiated."""
    from wildetect.cli_ui_integration import cli_ui_integration

    assert cli_ui_integration is not None
    assert isinstance(cli_ui_integration, CLIUIIntegration)


if __name__ == "__main__":
    pytest.main([__file__])
