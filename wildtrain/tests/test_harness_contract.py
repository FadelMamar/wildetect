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
from wildtrain.shared.schemas.classification import ClassificationConfig
from wildtrain.shared.schemas.detection import DetectionConfig

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


def test_shipped_classification_yaml_is_roi_single_class(tmp_path: Path) -> None:
    """Shipped classifier config is the ROI wildlife setup once its root exists."""
    payload = _resolved_mapping(CLASSIFICATION_YAML)
    payload["dataset"]["root_data_directory"] = str(tmp_path)

    config = ClassificationConfig.model_validate(payload)

    assert config.dataset.dataset_type == "roi"
    assert config.dataset.single_class is not None
    assert config.dataset.single_class.enable is True
    assert config.dataset.single_class.single_class_name == "wildlife"
    assert config.dataset.single_class.background_class_name == "background"
    assert config.model.input_size == config.dataset.input_size
    assert config.checkpoint.monitor == "val_f1score"
    assert set(payload) >= {"dataset", "model", "train", "checkpoint", "mlflow"}


def test_shipped_detection_yaml_is_single_class_localizer() -> None:
    """The committed detector config trains a class-agnostic localizer."""
    payload = _resolved_mapping(DETECTION_YAML)
    config = DetectionConfig.model_validate(payload)

    assert config.train.single_cls is True
    assert config.use_custom_yolo is False
    assert config.model.weights == "yolo12s.pt"
    assert config.dataset.discard_classes == [
        "rocks",
        "vegetation",
        "other",
        "termite mound",
    ]
    assert set(payload) >= {
        "dataset",
        "curriculum",
        "pretraining",
        "model",
        "name",
        "project",
        "mlflow",
        "train",
    }


def test_experiment_envelope_reuses_existing_bodies(tmp_path: Path) -> None:
    classification = _resolved_mapping(CLASSIFICATION_YAML)
    classification["dataset"]["root_data_directory"] = str(tmp_path)
    classification["task"] = "classification"
    classification_path = tmp_path / "classification.yaml"
    classification_path.write_text(yaml.safe_dump(classification), encoding="utf-8")

    detection = _resolved_mapping(DETECTION_YAML)
    detection["task"] = "detection"
    detection_path = tmp_path / "detection.yaml"
    detection_path.write_text(yaml.safe_dump(detection), encoding="utf-8")

    classification_experiment = load_experiment(classification_path)
    detection_experiment = load_experiment(detection_path)

    assert classification_experiment.task == "classification"
    assert classification_experiment.dataset.dataset_type == "roi"
    assert detection_experiment.task == "detection"
    assert detection_experiment.train.single_cls is True


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
    """Freeze which loader each ConfigType calls, including the registration quirk."""
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
        # Current behavior: detector registration validates the localizer schema.
        ConfigType.DETECTOR_REGISTRATION: "localizer_registration",
        ConfigType.CLASSIFIER_REGISTRATION: "classifier_registration",
    }
    for config_type, loader_name in expected.items():
        called.clear()
        assert ConfigLoader.validate_config_file(path, config_type) is True
        assert called == [loader_name]

    with pytest.raises(Exception, match="Unknown config type"):
        ConfigLoader.validate_config_file(path, ConfigType.MODEL_REGISTRATION)


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


def test_run_command_help_and_old_train_commands() -> None:
    run_help = RUNNER.invoke(app, ["run", "--help"])
    train_help = RUNNER.invoke(app, ["train", "--help"])

    assert run_help.exit_code == 0
    assert "--config" in run_help.output
    assert train_help.exit_code == 0
    assert "classifier" in train_help.output
    assert "detector" in train_help.output


def test_run_command_dispatches_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _resolved_mapping(DETECTION_YAML)
    payload["task"] = "detection"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

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


def test_train_commands_dispatch_through_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classification = _resolved_mapping(CLASSIFICATION_YAML)
    classification["dataset"]["root_data_directory"] = str(tmp_path)
    classification_path = tmp_path / "classification.yaml"
    classification_path.write_text(yaml.safe_dump(classification), encoding="utf-8")

    called: list[str] = []

    def fake_run(config, debug: bool = False, registry=None) -> FitResult:
        if config.task == "classification":
            called.append(f"classification:{config.dataset.dataset_type}")
        else:
            called.append(f"detection:{config.train.single_cls}")
        return FitResult(best_path=None, best_score=None)

    monkeypatch.setattr("wildtrain.cli.commands.train.run_experiment", fake_run)

    classification_result = RUNNER.invoke(
        app, ["train", "classifier", "--config", str(classification_path)]
    )
    detection_result = RUNNER.invoke(
        app, ["train", "detector", "--config", str(DETECTION_YAML)]
    )

    assert classification_result.exit_code == 0, classification_result.output
    assert detection_result.exit_code == 0, detection_result.output
    assert called == ["classification:roi", "detection:True"]


def test_hydra_and_openmim_are_not_dependencies() -> None:
    project = (REPO_ROOT / "wildtrain" / "pyproject.toml").read_text(encoding="utf-8")
    dependency_block = project.split("[project]", 1)[1].split("[project.scripts]", 1)[0]
    assert "hydra-core" not in dependency_block
    assert "openmim" not in dependency_block
    assert "omegaconf" in dependency_block
