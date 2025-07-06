"""
Tests for the processor classes defined in processor.py.
"""

import logging
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from wildetect.core.data.detection import Detection
from wildetect.core.processor.processor import (
    Classifier,
    FeatureExtractor,
    Processor,
    RoIPostProcessor,
    check_images_sequences,
    get_processor,
)


class TestProcessorBase:
    """Test base processor functionality."""

    def test_get_processor_valid_names(self):
        """Test get_processor with valid processor names."""
        assert get_processor("feature_extractor") == FeatureExtractor
        assert get_processor("classifier") == Classifier
        assert get_processor("roi_post") == RoIPostProcessor

    def test_get_processor_invalid_name(self):
        """Test get_processor with invalid processor name."""
        with pytest.raises(NotImplementedError):
            get_processor("invalid_processor")

    def test_check_images_sequences_valid(self):
        """Test check_images_sequences with valid input."""
        images = [Image.new("RGB", (100, 100)) for _ in range(3)]
        # Should not raise any exception
        check_images_sequences(images)

    def test_check_images_sequences_not_sequence(self):
        """Test check_images_sequences with non-sequence input."""
        with pytest.raises(AssertionError):
            check_images_sequences("not a sequence")

    def test_check_images_sequences_invalid_image(self):
        """Test check_images_sequences with invalid image types."""
        images = [
            Image.new("RGB", (100, 100)),
            "not an image",
            Image.new("RGB", (100, 100)),
        ]
        with pytest.raises(AssertionError):
            check_images_sequences(images)

    def test_processor_context_manager(self):
        """Test processor context manager functionality."""

        class TestProcessor(Processor):
            def run(self, *args, **kwargs):
                return "test"

            def cleanup(self):
                self.cleaned_up = True

        processor = TestProcessor()
        with processor as p:
            assert p is processor
        assert hasattr(processor, "cleaned_up")


class TestFeatureExtractor:
    """Test FeatureExtractor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_images()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_images(self):
        """Create test images for testing."""
        # Create test images
        for i in range(3):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = Path(self.temp_dir) / f"test_image_{i}.jpg"
            img.save(img_path)

    @patch("wildetect.core.processor.processor.AutoImageProcessor")
    @patch("wildetect.core.processor.processor.AutoModel")
    def test_feature_extractor_initialization(
        self, mock_auto_model, mock_auto_processor
    ):
        """Test FeatureExtractor initialization."""
        # Mock the HuggingFace components
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model

        extractor = FeatureExtractor("test/path")

        assert extractor.processor == mock_processor
        assert extractor.extractor == mock_model
        assert extractor.device == "cpu"

    @patch("wildetect.core.processor.processor.AutoImageProcessor")
    @patch("wildetect.core.processor.processor.AutoModel")
    def test_feature_extractor_initialization_failure(
        self, mock_auto_model, mock_auto_processor
    ):
        """Test FeatureExtractor initialization failure."""
        mock_auto_processor.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(Exception):
            FeatureExtractor("invalid/model/path")

    @patch("wildetect.core.processor.processor.AutoImageProcessor")
    @patch("wildetect.core.processor.processor.AutoModel")
    def test_feature_extractor_run_empty_images(
        self, mock_auto_model, mock_auto_processor
    ):
        """Test FeatureExtractor run with empty images."""
        # Mock the components
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model

        extractor = FeatureExtractor("test/path")

        with pytest.raises(ValueError, match="Images sequence cannot be empty"):
            extractor.run([])

    @patch("wildetect.core.processor.processor.AutoImageProcessor")
    @patch("wildetect.core.processor.processor.AutoModel")
    def test_feature_extractor_run_success(self, mock_auto_model, mock_auto_processor):
        """Test FeatureExtractor run with valid images."""
        # Mock the components
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Mock the outputs
        mock_outputs = Mock()
        mock_outputs.pooler_output = torch.randn(2, 768)  # 2 images, 768 features

        # Mock processor to return a dict-like object with .to() method
        class MockBatch(dict):
            def to(self, device):
                return self

        mock_inputs = MockBatch({"input_ids": torch.randn(2, 3, 224, 224)})
        mock_processor.return_value = mock_inputs
        mock_model.return_value = mock_outputs

        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model

        extractor = FeatureExtractor("test/path")

        # Create test images
        images = [Image.new("RGB", (224, 224)) for _ in range(2)]

        features = extractor.run(images, batch_size=2)

        assert isinstance(features, np.ndarray)
        assert features.shape == (2, 768)

    @patch("wildetect.core.processor.processor.AutoImageProcessor")
    @patch("wildetect.core.processor.processor.AutoModel")
    def test_feature_extractor_cleanup(self, mock_auto_model, mock_auto_processor):
        """Test FeatureExtractor cleanup."""
        # Mock the components
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model

        extractor = FeatureExtractor("test/model/path")

        # Test cleanup
        extractor.cleanup()

        # Check that GPU cache is cleared (mocked)
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            extractor.cleanup()
            if torch.cuda.is_available():
                mock_empty_cache.assert_called_once()

    def test_feature_extractor_real(self):
        """Test FeatureExtractor with real model (requires internet connection)."""
        try:
            extractor = FeatureExtractor("facebook/dinov2-with-registers-small")

            # Create test images
            images = [
                Image.new("RGB", (224, 224), color="red"),
                Image.new("RGB", (224, 224), color="blue"),
                Image.new("RGB", (224, 224), color="green"),
            ]

            # Extract features
            features = extractor.run(images, batch_size=2)

            # Verify output
            assert isinstance(features, np.ndarray)
            assert features.shape[0] == 3  # One feature vector per image
            assert features.shape[1] > 0  # Feature dimension should be positive

            # Test with single image
            single_features = extractor.run([images[0]], batch_size=1)
            assert single_features.shape[0] == 1
            assert (
                single_features.shape[1] == features.shape[1]
            )  # Same feature dimension

            # Cleanup
            extractor.cleanup()

        except Exception as e:
            # Fail the test if model download fails (no internet, etc.)
            pytest.fail(f"Real FeatureExtractor test failed: {e}")


class TestClassifier:
    """Test Classifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_model()
        self.create_test_images()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_model(self):
        """Create a test model file."""
        # Create a simple test model
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3),  # 3 classes
        )

        # Save as TorchScript
        model.eval()
        traced_model = torch.jit.script(model)
        self.model_path = Path(self.temp_dir) / "test_model.pt"
        traced_model.save(str(self.model_path))

    def create_test_images(self):
        """Create test images for testing."""
        for i in range(3):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = Path(self.temp_dir) / f"test_image_{i}.jpg"
            img.save(img_path)

    def test_classifier_initialization(self):
        """Test Classifier initialization."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}

        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        assert classifier.label_map == label_map
        assert classifier.device == "cpu"
        assert classifier.feature_extractor is None

    def test_classifier_initialization_with_feature_extractor(self):
        """Test Classifier initialization with feature extractor."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}

        with patch("wildetect.core.processor.processor.FeatureExtractor") as mock_fe:
            mock_fe_instance = Mock()
            mock_fe.return_value = mock_fe_instance

            classifier = Classifier(
                model_path=str(self.model_path),
                label_map=label_map,
                feature_extractor_path="test/feature/extractor",
                device="cpu",
            )

            assert classifier.feature_extractor is not None

    def test_classifier_initialization_invalid_label_map(self):
        """Test Classifier initialization with invalid label map."""
        with pytest.raises(ValueError, match="label_map must be a dictionary"):
            Classifier(
                model_path=str(self.model_path), label_map="not a dict", device="cpu"
            )

    def test_classifier_pil_to_numpy(self):
        """Test _pil_to_numpy method."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}
        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        numpy_img = classifier._pil_to_numpy(img)

        assert isinstance(numpy_img, np.ndarray)
        assert numpy_img.shape == (100, 100, 3)

    def test_classifier_run_empty_images(self):
        """Test Classifier run with empty images."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}
        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        with pytest.raises(ValueError, match="Images sequence cannot be empty"):
            classifier.run([])

    def test_classifier_run_invalid_images(self):
        """Test Classifier run with invalid images."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}
        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        with pytest.raises(AssertionError):
            classifier.run(["not an image"])

    def test_classifier_run_success(self):
        """Test Classifier run with valid images."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}
        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        # Create test images
        images = [Image.new("RGB", (224, 224)) for _ in range(2)]

        predictions = classifier.run(images)

        assert isinstance(predictions, list)
        assert len(predictions) == 2
        assert all(pred in label_map.values() for pred in predictions)

    def test_classifier_cleanup(self):
        """Test Classifier cleanup."""
        label_map = {0: "class1", 1: "class2", 2: "class3"}
        classifier = Classifier(
            model_path=str(self.model_path), label_map=label_map, device="cpu"
        )

        # Test cleanup
        classifier.cleanup()

        # Check that GPU cache is cleared (mocked)
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            classifier.cleanup()
            if torch.cuda.is_available():
                mock_empty_cache.assert_called_once()


class TestRoIPostProcessor:
    """Test RoIPostProcessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_model()
        self.create_test_image()
        self.create_test_detections()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_model(self):
        """Create a test model file."""
        # Create a simple test model
        model = torch.nn.Sequential(
            torch.nn.Linear(3 * 96 * 96, 512),  # roi_size = 96
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),  # 2 classes: groundtruth, other
        )

        # Save as TorchScript
        model.eval()
        traced_model = torch.jit.script(model)
        self.model_path = Path(self.temp_dir) / "test_roi_model.pt"
        traced_model.save(str(self.model_path))

    def create_test_image(self):
        """Create a test image."""
        img_array = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        img.save(self.test_image_path)
        self.test_image = img

    def create_test_detections(self):
        """Create test detections."""
        self.test_detections = [
            Detection(
                bbox=[100, 100, 200, 200],
                confidence=0.8,
                class_id=0,
                class_name="test_class",
            ),
            Detection(
                bbox=[300, 300, 400, 400],
                confidence=0.9,
                class_id=1,
                class_name="test_class2",
            ),
            Detection(
                bbox=[500, 500, 600, 600],
                confidence=0.7,
                class_id=0,
                class_name="test_class",
            ),
        ]

    def test_roi_post_processor_initialization(self):
        """Test RoIPostProcessor initialization."""
        label_map = {0: "groundtruth", 1: "other"}

        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        assert processor.box_size == 96
        assert processor.keep == ["groundtruth"]
        assert processor.classifier is not None

    def test_roi_post_processor_initialization_with_custom_keep_classes(self):
        """Test RoIPostProcessor initialization with custom keep classes."""
        label_map = {0: "groundtruth", 1: "other"}

        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
            keep_classes=["groundtruth", "other"],
        )

        assert processor.keep == ["groundtruth", "other"]

    def test_roi_post_processor_initialization_with_classifier(self):
        """Test RoIPostProcessor initialization with existing classifier."""
        label_map = {0: "groundtruth", 1: "other"}
        classifier = Mock()

        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
            classifier=classifier,
        )

        assert processor.classifier == classifier

    def test_roi_post_processor_run_invalid_image(self):
        """Test RoIPostProcessor run with invalid image."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        with pytest.raises(AssertionError, match="image must be a PIL Image"):
            processor.run(self.test_detections, "not an image")

    def test_roi_post_processor_run_empty_detections(self):
        """Test RoIPostProcessor run with empty detections."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        result = processor.run([], self.test_image)
        assert result == []

    def test_roi_post_processor_run_detections_missing_coordinates(self):
        """Test RoIPostProcessor run with detections missing coordinates."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        # Use a mock Detection object that lacks x_center/y_center
        invalid_detection = Mock(spec=Detection)
        del invalid_detection.x_center
        del invalid_detection.y_center

        result = processor.run([invalid_detection], self.test_image)
        assert result == []

    @patch("wildetect.core.processor.processor.tqdm")
    def test_roi_post_processor_run_success(self, mock_tqdm):
        """Test RoIPostProcessor run with valid detections."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        # Mock the classifier to return predictable results
        processor.classifier.run = Mock(
            return_value=["groundtruth", "other", "groundtruth"]
        )

        result = processor.run(self.test_detections, self.test_image)

        # Should keep detections classified as "groundtruth"
        assert len(result) == 2  # 2 detections classified as "groundtruth"
        assert all(det.class_name == "test_class" for det in result)

    @patch("wildetect.core.processor.processor.tqdm")
    def test_roi_post_processor_run_with_verbose(self, mock_tqdm):
        """Test RoIPostProcessor run with verbose mode."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        # Mock the classifier
        processor.classifier.run = Mock(
            return_value=["groundtruth", "other", "groundtruth"]
        )

        # Mock tqdm
        mock_tqdm_instance = Mock()
        mock_tqdm_instance.__iter__ = Mock(return_value=iter([Mock(), Mock(), Mock()]))
        mock_tqdm.return_value = mock_tqdm_instance

        result = processor.run(self.test_detections, self.test_image, verbose=True)

        # Should call tqdm
        mock_tqdm.assert_called_once()
        assert len(result) == 2

    def test_roi_post_processor_run_crop_too_small(self):
        """Test RoIPostProcessor run with crops that are too small."""
        label_map = {0: "groundtruth", 1: "other"}
        processor = RoIPostProcessor(
            model_path=str(self.model_path),
            label_map=label_map,
            roi_size=96,
            device="cpu",
        )

        # Create detection that would result in very small crop
        small_detection = Detection(
            bbox=[0, 0, 5, 5],  # Very small bbox
            confidence=0.8,
            class_id=0,
            class_name="test_class",
        )

        # Should raise a RuntimeError due to shape mismatch
        with pytest.raises(RuntimeError):
            processor.run([small_detection], self.test_image)


def run_all_processor_tests():
    """Run all processor tests."""
    logger = logging.getLogger(__name__)
    logger.info("Starting processor tests...")

    # Run tests using pytest
    import pytest

    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_processor_tests()
