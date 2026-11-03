"""Schema, legacy YAML, sweep, and API/CLI dispatch for the unified experiment."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError
from typer.testing import CliRunner

from wildtrain.cli.cli import app
from wildtrain.harness.config import SweepBlock, load_experiment
from wildtrain.harness.result import FitResult
from wildtrain.harness.sweep import ExperimentSweeper, load_sweep
from wildtrain.trainers.sweepers import ClassifierSweeper, DetectionSweeper

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "classification" / "classification_train.yaml"
)
DETECTION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "detection" / "yolo_configs" / "yolo.yaml"
)
RUNNER = CliRunner()


def _resolved(path: Path) -> dict:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _classification_file(tmp_path: Path) -> Path:
    text = CLASSIFICATION_YAML.read_text(encoding="utf-8")
    text = text.replace(r"D:\PhD\workspace\data", tmp_path.as_posix())
    path = tmp_path / "classification.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_legacy_yaml_converts_for_both_tasks(tmp_path: Path) -> None:
    classification = load_experiment(_classification_file(tmp_path))
    detection = load_experiment(DETECTION_YAML)

    assert classification.task == "classification"
    assert classification.dataset.dataset_type == "roi"
    assert classification.body.model.input_size == classification.dataset.input_size
    trainer = classification.trainer_config()
    assert trainer.checkpoint.dirpath == classification.run.output_dir
    assert trainer.mlflow.experiment_name == (
        classification.tracking.mlflow.experiment_name
    )

    assert detection.task == "detection"
    assert detection.train.single_cls is True
    assert detection.trainer_config().project == detection.run.project
    assert detection.trainer_config().name == detection.run.name


def test_canonical_schema_validates_both_tasks(tmp_path: Path) -> None:
    classification = _resolved(_classification_file(tmp_path))
    canonical_classification = {
        "task": "classification",
        "tracking": {"mlflow": classification.pop("mlflow")},
        "run": {
            "name": "canonical-cls",
            "project": "wildtrain",
            "output_dir": classification["checkpoint"]["dirpath"],
        },
        "classification": classification,
    }
    loaded = load_experiment(
        _write(tmp_path / "canonical-cls.yaml", canonical_classification)
    )
    assert loaded.task == "classification"
    assert loaded.trainer_config().mlflow.run_name == "canonical-cls"
    assert loaded.dataset.dataset_type == "roi"

    detection = _resolved(DETECTION_YAML)
    canonical_detection = {
        "task": "detection",
        "tracking": {"mlflow": detection.pop("mlflow")},
        "run": {
            "name": detection.pop("name"),
            "project": detection.pop("project"),
            "output_dir": "runs/detection",
        },
        "detection": detection,
    }
    loaded_detection = load_experiment(
        _write(tmp_path / "canonical-det.yaml", canonical_detection)
    )
    assert loaded_detection.task == "detection"
    assert loaded_detection.train.single_cls is True
    assert loaded_detection.trainer_config().name == canonical_detection["run"]["name"]


def test_unknown_task_and_two_bodies_are_rejected(tmp_path: Path) -> None:
    detection = _resolved(DETECTION_YAML)
    detection["task"] = "segmentation"
    with pytest.raises(ValidationError):
        load_experiment(_write(tmp_path / "bad.yaml", detection))

    classification = _resolved(_classification_file(tmp_path))
    both = {
        "task": "classification",
        "tracking": {"mlflow": classification["mlflow"]},
        "classification": classification,
        "detection": detection,
    }
    with pytest.raises(ValueError, match="only one"):
        load_experiment(_write(tmp_path / "both.yaml", both))


def test_sweep_applies_dotted_choices_for_each_task(tmp_path: Path) -> None:
    classification = load_experiment(_classification_file(tmp_path))
    original_lr = classification.train.lr
    classification.sweep = SweepBlock(
        n_trials=1,
        seed=1,
        name="cls",
        objective="f1_score",
        parameters={"model.dropout": [0.4], "train.lr": [0.0007]},
    )
    seen: dict = {}

    def classification_objective(config):
        seen["dropout"] = config.classification.model.dropout
        seen["lr"] = config.train.lr
        return 0.8

    study = ExperimentSweeper(classification, objective=classification_objective).run()
    assert seen == {"dropout": 0.4, "lr": 0.0007}
    assert study.best_value == 0.8
    assert classification.train.lr == original_lr

    detection = load_experiment(DETECTION_YAML)
    detection.sweep = SweepBlock(
        n_trials=1,
        seed=2,
        direction="maximize",
        objective="map_50",
        parameters={"train.lr0": [0.02], "train.imgsz": [320]},
    )

    def detection_objective(config):
        seen["lr0"] = config.detection.train.lr0
        seen["imgsz"] = config.train.imgsz
        return 0.3

    detection_study = ExperimentSweeper(
        detection, objective=detection_objective
    ).run()
    assert seen["lr0"] == 0.02
    assert seen["imgsz"] == 320
    assert detection_study.best_value == 0.3


def test_legacy_sweep_yaml_uses_the_same_runner(tmp_path: Path) -> None:
    base = _classification_file(tmp_path)
    sweep_path = _write(
        tmp_path / "classification_sweep.yaml",
        {
            "base_config": str(base),
            "parameters": {
                "model": {"backbone": ["timm/resnet18.a1_in1k"], "dropout": [0.1]},
                "train": {
                    "lr": [0.0004],
                    "lrf": [0.1],
                    "label_smoothing": [0.0],
                    "weight_decay": [0.0],
                    "batch_size": [2],
                    "epochs": [1],
                },
            },
            "sweep_name": "classification-unit",
            "n_trials": 3,
            "seed": 7,
        },
    )
    experiment = load_sweep(sweep_path)
    assert experiment.task == "classification"
    assert experiment.sweep is not None
    assert experiment.sweep.parameters["model.backbone"] == ["timm/resnet18.a1_in1k"]
    assert experiment.sweep.direction == "maximize"
    experiment.sweep.n_trials = 1
    experiment.sweep.storage = None

    def objective(config):
        assert config.body.model.backbone == "timm/resnet18.a1_in1k"
        assert config.train.epochs == 1
        return 0.6

    assert ExperimentSweeper(experiment, objective=objective).run().best_value == 0.6

    detection_base = tmp_path / "detection.yaml"
    detection_base.write_text(
        DETECTION_YAML.read_text(encoding="utf-8"), encoding="utf-8"
    )
    detection_sweep = _write(
        tmp_path / "detection_sweep.yaml",
        {
            "base_config": str(detection_base),
            "parameters": {
                "model": None,
                "train": {
                    "lr0": [0.01],
                    "lrf": [0.1],
                    "batch": [2],
                    "epochs": [1],
                    "imgsz": [64],
                    "optimizer": ["AdamW"],
                    "weight_decay": [0.0],
                },
            },
            "sweep_name": "detection-unit",
            "n_trials": 4,
            "seed": 3,
            "objective": "fitness",
            "direction": "maximize",
        },
    )
    sweeper = DetectionSweeper(str(detection_sweep))
    sweeper._runner.config.sweep.n_trials = 1
    sweeper._runner.config.sweep.storage = None
    sweeper._runner.objective = lambda config: (
        0.5 if config.train.imgsz == 64 and config.train.lr0 == 0.01 else 0.0
    )
    study = sweeper.run()
    assert study.best_value == 0.5

    classifier = ClassifierSweeper(str(sweep_path))
    classifier._runner.config.sweep.n_trials = 1
    classifier._runner.config.sweep.storage = None
    classifier._runner.objective = lambda config: 0.25
    assert classifier.run().best_value == 0.25


def test_cli_sweep_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _classification_file(tmp_path)
    sweep_path = _write(
        tmp_path / "sweep.yaml",
        {
            "base_config": str(base),
            "parameters": {
                "model": {"backbone": ["timm/resnet18.a1_in1k"], "dropout": [0.0]},
                "train": {
                    "lr": [0.001],
                    "lrf": [0.1],
                    "label_smoothing": [0.0],
                    "weight_decay": [0.0],
                    "batch_size": [2],
                    "epochs": [1],
                },
            },
            "sweep_name": "cli-sweep",
            "n_trials": 1,
            "seed": 1,
        },
    )
    seen: dict = {}

    class FakeSweeper:
        def __init__(self, config, objective=None, debug: bool = False):
            seen["task"] = config.task
            seen["debug"] = debug

        def run(self):
            trial = type("Trial", (), {"value": 0.9, "params": {"train.lr": 0.001}})()
            return type("Study", (), {"best_trial": trial})()

    monkeypatch.setattr("wildtrain.cli.commands.sweep.ExperimentSweeper", FakeSweeper)
    result = RUNNER.invoke(app, ["sweep", "-c", str(sweep_path), "--debug"])
    assert result.exit_code == 0, result.output
    assert seen == {"task": "classification", "debug": True}


def test_api_routes_dispatch_through_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_run(config, debug: bool = False, registry=None) -> FitResult:
        seen.append(config.task)
        return FitResult(best_path="best.pt", best_score=0.2, score_name="score")

    def inline(task_func, *args, metadata=None, **kwargs):
        task_func(*args, **kwargs)
        return "job-1"

    monkeypatch.setattr(
        "wildtrain.api.services.training_service.run_experiment", fake_run
    )
    monkeypatch.setattr("wildtrain.api.routers.training.create_background_job", inline)

    from wildtrain.api.main import fastapi_app

    client = TestClient(fastapi_app)
    detection = _resolved(DETECTION_YAML)
    created = client.post("/training/experiments", json={"config": detection})
    assert created.status_code == 200, created.text
    assert created.json()["job_id"] == "job-1"

    aliased = client.post("/experiments", json={"config": detection})
    assert aliased.status_code == 200, aliased.text

    adapted = client.post("/training/detector", json={"config": detection})
    assert adapted.status_code == 200, adapted.text
    assert seen == ["detection", "detection", "detection"]
