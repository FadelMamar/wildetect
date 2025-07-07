"""
Tests for CLI-UI Integration.

This module tests the integration between CLI and UI functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from wildetect.cli_ui_integration import CLIUIIntegration


class TestCLIUIIntegration:
    """Test CLI-UI integration functionality."""

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
        assert hasattr(self.integration, "run_detection_ui")
        assert hasattr(self.integration, "run_census_ui")
        assert hasattr(self.integration, "analyze_results_ui")
        assert hasattr(self.integration, "visualize_results_ui")
        assert hasattr(self.integration, "get_system_info_ui")

    def test_system_info_ui(self):
        """Test system information retrieval."""
        system_info = self.integration.get_system_info_ui()

        assert isinstance(system_info, dict)
        assert "components" in system_info
        assert "dependencies" in system_info
        assert "timestamp" in system_info

        # Check that PyTorch is checked
        assert "PyTorch" in system_info["components"]
        assert "CUDA" in system_info["components"]

        # Check that dependencies are checked
        assert "numpy" in system_info["dependencies"]
        assert "PIL" in system_info["dependencies"]

    def test_analyze_results_ui_with_valid_file(self):
        """Test results analysis with valid JSON file."""
        # Create a sample results file
        sample_results = [
            {
                "image_path": "test1.jpg",
                "total_detections": 5,
                "class_counts": {"elephant": 3, "lion": 2},
            },
            {
                "image_path": "test2.jpg",
                "total_detections": 2,
                "class_counts": {"elephant": 1, "giraffe": 1},
            },
        ]

        results_file = Path(self.temp_dir) / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(sample_results, f)

        # Test analysis
        result = self.integration.analyze_results_ui(
            str(results_file), output_dir=self.temp_dir, create_map=False
        )

        assert result["success"] is True
        assert "analysis_results" in result
        assert result["analysis_results"]["total_images"] == 2
        assert result["analysis_results"]["total_detections"] == 7
        assert "elephant" in result["analysis_results"]["species_breakdown"]
        assert result["analysis_results"]["species_breakdown"]["elephant"] == 4

    def test_analyze_results_ui_with_invalid_file(self):
        """Test results analysis with invalid file."""
        invalid_file = Path(self.temp_dir) / "nonexistent.json"

        result = self.integration.analyze_results_ui(
            str(invalid_file), output_dir=self.temp_dir
        )

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"]

    def test_visualize_results_ui_with_valid_file(self):
        """Test results visualization with valid JSON file."""
        # Create a sample results file
        sample_results = [
            {
                "image_path": "test1.jpg",
                "total_detections": 3,
                "class_counts": {"elephant": 2, "lion": 1},
            }
        ]

        results_file = Path(self.temp_dir) / "test_viz_results.json"
        with open(results_file, "w") as f:
            json.dump(sample_results, f)

        # Test visualization
        result = self.integration.visualize_results_ui(
            str(results_file),
            output_dir=self.temp_dir,
            show_confidence=True,
            create_map=False,
        )

        assert result["success"] is True
        assert "visualization_data" in result
        assert result["visualization_data"]["total_images"] == 1
        assert result["visualization_data"]["total_detections"] == 3
        assert "elephant" in result["visualization_data"]["species_counts"]

    def test_convert_drone_images_to_ui_format(self):
        """Test conversion of drone images to UI format."""
        # Mock drone image with statistics
        mock_drone_image = Mock()
        mock_drone_image.get_statistics.return_value = {
            "image_path": "test.jpg",
            "total_detections": 5,
            "class_counts": {"elephant": 3, "lion": 2},
        }

        drone_images = [mock_drone_image]
        results = self.integration._convert_drone_images_to_ui_format(drone_images)

        assert len(results) == 1
        assert results[0]["image_path"] == "test.jpg"
        assert results[0]["total_detections"] == 5
        assert results[0]["class_counts"] == {"elephant": 3, "lion": 2}
        assert results[0]["species_counts"] == {"elephant": 3, "lion": 2}
        assert results[0]["total_count"] == 5

    def test_analyze_detection_results(self):
        """Test analysis of detection results."""
        sample_results = [
            {"total_detections": 5, "class_counts": {"elephant": 3, "lion": 2}},
            {"total_detections": 2, "class_counts": {"elephant": 1, "giraffe": 1}},
        ]

        analysis = self.integration._analyze_detection_results(sample_results)

        assert analysis["total_images"] == 2
        assert analysis["total_detections"] == 7
        assert analysis["species_breakdown"]["elephant"] == 4
        assert analysis["species_breakdown"]["lion"] == 2
        assert analysis["species_breakdown"]["giraffe"] == 1

    def test_extract_visualization_data(self):
        """Test extraction of visualization data."""
        sample_results = [
            {"total_detections": 3, "class_counts": {"elephant": 2, "lion": 1}}
        ]

        viz_data = self.integration._extract_visualization_data(sample_results)

        assert viz_data["total_images"] == 1
        assert viz_data["total_detections"] == 3
        assert viz_data["species_counts"]["elephant"] == 2
        assert viz_data["species_counts"]["lion"] == 1
        assert "timestamp" in viz_data

    @patch("wildetect.cli_ui_integration.GeographicVisualizer")
    def test_create_geographic_visualization(self, mock_visualizer):
        """Test geographic visualization creation."""
        # Mock drone images
        mock_drone_image = Mock()
        mock_drone_images = [mock_drone_image]

        # Mock visualizer
        mock_viz_instance = Mock()
        mock_visualizer.return_value = mock_viz_instance
        mock_map = Mock()
        mock_viz_instance.create_map.return_value = mock_map

        # Test visualization creation
        self.integration._create_geographic_visualization(
            mock_drone_images, output_dir=self.temp_dir
        )

        # Verify visualizer was called
        mock_visualizer.assert_called_once()
        mock_viz_instance.create_map.assert_called_once_with(mock_drone_images)
        mock_map.save.assert_called_once()

    def test_export_analysis_report(self):
        """Test analysis report export."""
        analysis_results = {
            "total_images": 2,
            "total_detections": 7,
            "species_breakdown": {"elephant": 4, "lion": 3},
        }

        self.integration._export_analysis_report(analysis_results, self.temp_dir)

        report_file = Path(self.temp_dir) / "analysis_report.json"
        assert report_file.exists()

        with open(report_file, "r") as f:
            exported_data = json.load(f)

        assert exported_data["total_images"] == 2
        assert exported_data["total_detections"] == 7
        assert exported_data["species_breakdown"]["elephant"] == 4

    def test_export_visualization_report(self):
        """Test visualization report export."""
        viz_data = {
            "total_images": 1,
            "total_detections": 3,
            "species_counts": {"elephant": 2, "lion": 1},
        }

        self.integration._export_visualization_report(
            "test_results.json", viz_data, self.temp_dir
        )

        report_file = Path(self.temp_dir) / "visualization_report.json"
        assert report_file.exists()

        with open(report_file, "r") as f:
            exported_data = json.load(f)

        assert exported_data["total_images"] == 1
        assert exported_data["total_detections"] == 3
        assert exported_data["species_counts"]["elephant"] == 2


def test_integration_import():
    """Test that the integration can be imported."""
    from wildetect.cli_ui_integration import cli_ui_integration

    assert cli_ui_integration is not None
    assert isinstance(cli_ui_integration, CLIUIIntegration)


if __name__ == "__main__":
    pytest.main([__file__])
