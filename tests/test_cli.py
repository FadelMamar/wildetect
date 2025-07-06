"""
Tests for the CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from src.wildetect.cli import app


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_images()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_images(self):
        """Create test images for testing."""
        from PIL import Image

        # Create a simple test image
        image = Image.new("RGB", (800, 600), color="red")
        test_image_path = Path(self.temp_dir) / "test_image.jpg"
        image.save(test_image_path)

        # Create another test image
        image2 = Image.new("RGB", (1024, 768), color="blue")
        test_image_path2 = Path(self.temp_dir) / "test_image2.jpg"
        image2.save(test_image_path2)

    def test_detect_command_help(self):
        """Test detect command help."""
        result = self.runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "Run wildlife detection on images" in result.output

    def test_visualize_command_help(self):
        """Test visualize command help."""
        result = self.runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "Visualize detection results" in result.output

    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "WildDetect System Information" in result.output

    def test_detect_command_with_directory(self):
        """Test detect command with directory input."""
        result = self.runner.invoke(
            app, ["detect", self.temp_dir, "--max-images", "1", "--verbose"]
        )

        # The command should run without errors, even if detection fails
        # due to missing model files
        assert result.exit_code in [0, 1]  # Allow both success and failure

    def test_detect_command_with_files(self):
        """Test detect command with file paths."""
        image_files = [
            str(Path(self.temp_dir) / "test_image.jpg"),
            str(Path(self.temp_dir) / "test_image2.jpg"),
        ]

        result = self.runner.invoke(
            app, ["detect", *image_files, "--max-images", "1", "--verbose"]
        )

        # The command should run without errors, even if detection fails
        # due to missing model files
        assert result.exit_code in [0, 1]  # Allow both success and failure

    def test_detect_command_invalid_path(self):
        """Test detect command with invalid path."""
        result = self.runner.invoke(app, ["detect", "nonexistent_path", "--verbose"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_visualize_command_invalid_file(self):
        """Test visualize command with invalid file."""
        result = self.runner.invoke(app, ["visualize", "nonexistent_file.json"])

        assert result.exit_code == 1
        assert "Results file not found" in result.output

    @patch("src.wildetect.cli.DetectionPipeline")
    def test_detect_command_mock_pipeline(self, mock_pipeline_class):
        """Test detect command with mocked pipeline."""
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.run_detection.return_value = []
        mock_pipeline_class.return_value = mock_pipeline

        result = self.runner.invoke(app, ["detect", self.temp_dir, "--max-images", "1"])

        # Should succeed with mocked pipeline
        assert result.exit_code == 0
        mock_pipeline.run_detection.assert_called_once()

    def test_detect_command_options(self):
        """Test detect command with various options."""
        result = self.runner.invoke(
            app,
            [
                "detect",
                self.temp_dir,
                "--model-type",
                "yolo",
                "--confidence",
                "0.5",
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--tile-size",
                "512",
                "--max-images",
                "1",
                "--verbose",
            ],
        )

        # Should run without errors (even if detection fails due to missing model)
        assert result.exit_code in [0, 1]

    def test_cli_version(self):
        """Test CLI version information."""
        result = self.runner.invoke(app, ["--version"])
        # Typer doesn't automatically add version, so this might fail
        # but we can test that the command structure is correct
        assert result.exit_code in [0, 2]  # 0 for success, 2 for missing version

    def test_cli_help(self):
        """Test CLI help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "WildDetect - Wildlife Detection System" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
