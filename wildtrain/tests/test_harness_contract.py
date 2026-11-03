"""Characterization tests for current configs and the phase-1 harness."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError
from typer.testing import CliRunner

from wildtrain.cli.cli import app
from wildtrain.harness.config import load_experiment
from wildtrain.harness.registry import TaskRegistry
from wildtrain.harness.result import FitResult
from wildtrain.models.register import ModelMetadata, ModelTask, ModelType
from wildtrain.shared.config_loader import ConfigLoader
from wildtrain.shared.config_types import ConfigType
from wildtrain.shared.validation import ConfigValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "classification" / "classification_train.yaml"
)
DETECTION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "detection" / "yolo_configs" / "yolo.yaml"
)

RUNNER = CliRunner()


def _resolved_mapping(path: Path) -> dict:
    loaded = OmegaConf.load(path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_shipped_classification_yaml_is_roi_collapse() -> None:
    """Shipped classifier experiment collapses ROI labels to wildlife."""
    config = load_experiment(CLASSIFICATION_YAML)

    assert config.task == "classification"
    assert config.data.format == "roi"
    assert config.data.label_policy.mode == "collapse"
    assert config.data.label_policy.positive_name == "wildlife"
    assert config.data.label_policy.background_name == "background"
    assert config.model.backend == "timm"
    assert config.checkpoint.monitor == "val_f1score"
    mapped = config.trainer_config()
    assert mapped.dataset.dataset_type == "roi"
    assert mapped.dataset.single_class.single_class_name == "wildlife"
    assert mapped.model.input_size == mapped.dataset.input_size


def test_shipped_detection_yaml_is_single_class_localizer() -> None:
    """The committed detector experiment trains a class-agnostic localizer."""
    config = load_experiment(DETECTION_YAML)

    assert config.task == "detection"
    assert config.data.format == "yolo"
    assert config.train.single_cls is True
    assert config.model.backend == "ultralytics"
    assert config.model.weights == "yolo12s.pt"
    assert config.data.discard == [
        "rocks",
        "vegetation",
        "other",
        "termite mound",
    ]
    mapped = config.trainer_config()
    assert mapped.use_custom_yolo is False
    assert mapped.train.single_cls is True
    assert mapped.train.batch == 32


def test_old_training_yaml_is_rejected(tmp_path: Path) -> None:
    payload = {
        "task": "classification",
        "dataset": {
            "root_data_directory": str(tmp_path),
            "dataset_type": "roi",
            "single_class": {"single_class_name": "wildlife"},
        },
        "mlflow": {"experiment_name": "old"},
    }
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer accepted"):
        load_experiment(path)


def test_experiment_rejects_unknown_task(tmp_path: Path) -> None:
    detection = _resolved_mapping(DETECTION_YAML)
    detection["task"] = "segmentation"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(detection), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_experiment(path)


def test_registry_dispatches_known_tasks_only() -> None:
    class Stub:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, config, debug: bool = False) -> FitResult:
            return FitResult(best_path=self.name, best_score=None)

    registry = TaskRegistry()
    registry.register(Stub("classification"))
    registry.register(Stub("detection"))

    assert registry.get("classification").name == "classification"
    assert registry.names() == ["classification", "detection"]
    with pytest.raises(KeyError, match="Unknown task"):
        registry.get("segmentation")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Stub("classification"))


def test_config_loader_dispatch_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freeze which loader each ConfigType calls.

    DETECTOR_REGISTRATION and MODEL_REGISTRATION both use the detector schema.
    """
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    called: list[str] = []

    def _record(name: str):
        def _loader(*args, **kwargs):
            called.append(name)
            return args[0] if args else None

        return _loader

    monkeypatch.setattr(
        ConfigLoader, "load_classification_config", _record("classification")
    )
    monkeypatch.setattr(ConfigLoader, "load_detection_config", _record("detection"))
    monkeypatch.setattr(
        ConfigLoader, "load_classification_eval_config", _record("classification_eval")
    )
    monkeypatch.setattr(
        ConfigLoader, "load_detection_eval_config", _record("detection_eval")
    )
    monkeypatch.setattr(
        ConfigLoader,
        "load_classification_visualization_config",
        _record("classification_visualization"),
    )
    monkeypatch.setattr(
        ConfigLoader,
        "load_detection_visualization_config",
        _record("detection_visualization"),
    )
    monkeypatch.setattr(ConfigLoader, "load_pipeline_config", _record("pipeline"))
    monkeypatch.setattr(
        ConfigLoader,
        "load_localizer_registration_config",
        _record("localizer_registration"),
    )
    monkeypatch.setattr(
        ConfigLoader,
        "load_classifier_registration_config",
        _record("classifier_registration"),
    )
    monkeypatch.setattr(
        ConfigLoader,
        "load_detector_registration_config",
        _record("detector_registration"),
    )

    expected = {
        ConfigType.CLASSIFICATION: "classification",
        ConfigType.DETECTION: "detection",
        ConfigType.CLASSIFICATION_EVAL: "classification_eval",
        ConfigType.DETECTION_EVAL: "detection_eval",
        ConfigType.CLASSIFICATION_VISUALIZATION: "classification_visualization",
        ConfigType.DETECTION_VISUALIZATION: "detection_visualization",
        ConfigType.CLASSIFICATION_PIPELINE: "pipeline",
        ConfigType.DETECTION_PIPELINE: "pipeline",
        ConfigType.DETECTOR_REGISTRATION: "detector_registration",
        ConfigType.CLASSIFIER_REGISTRATION: "classifier_registration",
        ConfigType.MODEL_REGISTRATION: "detector_registration",
    }
    for config_type, loader_name in expected.items():
        called.clear()
        assert ConfigLoader.validate_config_file(path, config_type) is True
        assert called == [loader_name]


def test_registered_model_metadata_keys() -> None:
    """Keys ModelMetadata.to_dict() writes for the census MLflow load."""
    metadata = ModelMetadata(
        batch=8,
        imgsz=800,
        task=ModelTask.DETECT,
        num_classes=2,
        model_type=ModelType.DETECTOR,
        cls_export_format="pt",
    )
    payload = metadata.to_dict()

    assert set(payload) == {
        "batch",
        "imgsz",
        "task",
        "num_classes",
        "model_type",
        "cls_export_format",
    }
    assert payload["task"] == "detect"
    assert payload["model_type"] == "detector"
    # Detector.input_shape reads imgsz. Census maps that key to tile size.
    assert "tilesize" not in payload


def test_wrapper_tasks_delegate_to_existing_trainers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wildtrain.harness.tasks.classification import ClassificationTask
    from wildtrain.harness.tasks.detection import DetectionTask

    class FakeClassifier:
        def __init__(self, config) -> None:
            self.config = config
            self.best_model_path = "clf.ckpt"
            self.best_model_score = 0.75

        def run(self, debug: bool = False) -> None:
            self.debug = debug

    class FakeDetector:
        def __init__(self, config) -> None:
            self.config = config
            self.best_model_path = "det.pt"
            self.best_model_score = None
            self.best_fitness = 0.4

        def run(self, debug: bool = False) -> None:
            self.debug = debug

    monkeypatch.setattr(
        "wildtrain.harness.tasks.classification.ClassifierTrainer", FakeClassifier
    )
    monkeypatch.setattr(
        "wildtrain.harness.tasks.detection.UltralyticsDetectionTrainer", FakeDetector
    )

    classification = ClassificationTask().run(object(), debug=True)
    detection = DetectionTask().run(object(), debug=False)

    assert classification == FitResult(
        best_path="clf.ckpt", best_score=0.75, score_name="val_f1score"
    )
    assert detection == FitResult(
        best_path="det.pt", best_score=0.4, score_name="fitness"
    )


def test_run_command_help_and_removed_train_commands() -> None:
    run_help = RUNNER.invoke(app, ["run", "--help"])
    train_help = RUNNER.invoke(app, ["train", "--help"])
    removed = RUNNER.invoke(app, ["train"])

    assert run_help.exit_code == 0
    assert "--config" in run_help.output
    assert train_help.exit_code == 0
    assert "wildtrain run" in train_help.output
    assert removed.exit_code == 1
    assert "removed" in removed.output.lower()


def test_run_command_dispatches_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = DETECTION_YAML

    seen: dict = {}

    def fake_run(config, debug: bool = False):
        seen["task"] = config.task
        seen["debug"] = debug
        seen["single_cls"] = config.train.single_cls
        return FitResult(best_path="best.pt", best_score=0.2, score_name="fitness")

    monkeypatch.setattr("wildtrain.cli.commands.run.run_experiment", fake_run)
    result = RUNNER.invoke(app, ["run", "-c", str(path), "--debug"])

    assert result.exit_code == 0, result.output
    assert seen == {"task": "detection", "debug": True, "single_cls": True}


def test_removed_train_subcommands_do_not_train() -> None:
    classification = RUNNER.invoke(
        app, ["train", "classifier", "--config", str(CLASSIFICATION_YAML)]
    )
    detection = RUNNER.invoke(
        app, ["train", "detector", "--config", str(DETECTION_YAML)]
    )

    assert classification.exit_code != 0
    assert detection.exit_code != 0


def test_detector_registration_type_uses_the_detector_schema(tmp_path: Path) -> None:
    localizer_only = tmp_path / "localizer.yaml"
    localizer_only.write_text(
        yaml.safe_dump(
            {
                "yolo": {"imgsz": 64, "weights": "best.pt"},
                "processing": {"batch_size": 2, "name": "localizer"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError):
        ConfigLoader.validate_config_file(
            localizer_only, ConfigType.DETECTOR_REGISTRATION
        )

    detector = tmp_path / "detector.yaml"
    detector.write_text(
        yaml.safe_dump(
            {
                "localizer": {
                    "yolo": {"imgsz": 64, "weights": "best.pt"},
                    "processing": {"batch_size": 2},
                },
                "classifier": {"weights": None, "processing": {"batch_size": 2}},
                "processing": {"name": "detector", "batch_size": 2},
            }
        ),
        encoding="utf-8",
    )
    assert (
        ConfigLoader.validate_config_file(detector, ConfigType.DETECTOR_REGISTRATION)
        is True
    )
    assert (
        ConfigLoader.validate_config_file(detector, ConfigType.MODEL_REGISTRATION)
        is True
    )


def test_hydra_and_openmim_are_not_dependencies() -> None:
    project = (REPO_ROOT / "wildtrain" / "pyproject.toml").read_text(encoding="utf-8")
    dependency_block = project.split("[project]", 1)[1].split("[project.scripts]", 1)[0]
    assert "hydra-core" not in dependency_block
    assert "openmim" not in dependency_block
    assert "omegaconf" in dependency_block
