"""New experiment schema, rejected legacy files, sweep, and API dispatch."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from wildtrain.cli.cli import app
from wildtrain.harness.config import (
    SweepBlock,
    experiment_from_mapping,
    load_experiment,
)
from wildtrain.harness.result import FitResult
from wildtrain.harness.sweep import ExperimentSweeper, load_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "classification" / "classification_train.yaml"
)
DETECTION_YAML = (
    REPO_ROOT / "wildtrain" / "configs" / "detection" / "yolo_configs" / "yolo.yaml"
)
CLASSIFICATION_SWEEP = (
    REPO_ROOT / "wildtrain" / "configs" / "classification" / "classification_sweep.yaml"
)
DETECTION_SWEEP = (
    REPO_ROOT / "wildtrain" / "configs" / "detection" / "detection_sweep.yaml"
)
RUNNER = CliRunner()


def _minimal_classification(root: Path) -> dict:
    return {
        "task": "classification",
        "name": "unit",
        "seed": 1,
        "tracking": {
            "mlflow": {
                "experiment_name": "unit",
                "run_name": "unit",
                "log_model": False,
            }
        },
        "data": {
            "format": "roi",
            "root": str(root),
            "input_size": 32,
            "batch_size": 2,
            "stats": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
        "model": {
            "backend": "timm",
            "backbone": "vit_tiny_patch16_224",
            "dropout": 0.1,
        },
        "train": {"epochs": 1, "lr": 0.001},
        "checkpoint": {"dirpath": str(root / "checkpoints")},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_omitted_label_policy_is_identity(tmp_path: Path) -> None:
    experiment = experiment_from_mapping(_minimal_classification(tmp_path))
    assert experiment.data.label_policy.mode == "identity"
    mapped = experiment.trainer_config()
    assert mapped.dataset.label_policy.mode == "identity"
    assert mapped.dataset.label_policy.enable is False


def test_trainer_mapping_renames_public_fields(tmp_path: Path) -> None:
    payload = _minimal_classification(tmp_path)
    payload["data"]["label_policy"] = {
        "mode": "collapse",
        "positive_name": "wildlife",
        "background_name": "background",
        "discard": ["rocks"],
    }
    payload["name"] = "mapped-run"
    payload["tracking"]["mlflow"].pop("run_name")
    experiment = experiment_from_mapping(payload)
    mapped = experiment.trainer_config()
    assert mapped.dataset.single_class.single_class_name == "wildlife"
    assert mapped.dataset.single_class.discard_classes == ["rocks"]
    assert mapped.mlflow.run_name == "mapped-run"
    assert mapped.checkpoint.dirpath == str(tmp_path / "checkpoints")

    detection = load_experiment(DETECTION_YAML)
    assert detection.trainer_config().train.batch == detection.train.batch_size
    assert detection.trainer_config().use_custom_yolo is False


def test_legacy_files_explain_the_replacement(tmp_path: Path) -> None:
    flat = _write(
        tmp_path / "flat.yaml",
        {
            "dataset": {
                "root_data_directory": str(tmp_path),
                "dataset_type": "roi",
                "single_class": {"single_class_name": "wildlife"},
            }
        },
    )
    sweep = _write(
        tmp_path / "sweep.yaml",
        {"base_config": str(flat), "parameters": {"train": {"lr": [0.001]}}},
    )
    nested = _write(
        tmp_path / "nested.yaml",
        {"task": "classification", "classification": {"dataset": {}}},
    )

    with pytest.raises(ValueError, match="data.label_policy.positive_name"):
        load_experiment(flat)
    with pytest.raises(ValueError, match="metrics/mAP50\\(B\\)"):
        load_experiment(sweep)
    with pytest.raises(ValueError, match="data.format"):
        load_experiment(nested)


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    payload = _minimal_classification(tmp_path)
    payload["task"] = "segmentation"
    with pytest.raises(ValidationError):
        experiment_from_mapping(payload)


def test_sweep_applies_new_field_paths(tmp_path: Path) -> None:
    classification = experiment_from_mapping(_minimal_classification(tmp_path))
    original = classification.train.lr
    classification.sweep = SweepBlock(
        n_trials=1,
        seed=1,
        objective="f1_score",
        parameters={"model.dropout": [0.4], "train.lr": [0.0007]},
    )
    seen = {}

    def classification_objective(config):
        seen["dropout"] = config.model.dropout
        seen["lr"] = config.train.lr
        return 0.8

    study = ExperimentSweeper(classification, objective=classification_objective).run()
    assert seen == {"dropout": 0.4, "lr": 0.0007}
    assert study.best_value == 0.8
    assert classification.train.lr == original

    detection = load_experiment(DETECTION_YAML)
    detection.sweep = SweepBlock(
        n_trials=1,
        seed=2,
        objective="metrics/mAP50(B)",
        parameters={"train.lr0": [0.02], "train.batch_size": [4]},
    )

    def detection_objective(config):
        seen["lr0"] = config.train.lr0
        seen["batch"] = config.trainer_config().train.batch
        return 0.3

    assert (
        ExperimentSweeper(detection, objective=detection_objective).run().best_value
        == 0.3
    )
    assert seen["lr0"] == 0.02
    assert seen["batch"] == 4


def test_shipped_sweep_files_load() -> None:
    classification = load_sweep(CLASSIFICATION_SWEEP)
    detection = load_sweep(DETECTION_SWEEP)
    assert classification.task == "classification"
    assert classification.sweep.objective == "f1_score"
    assert "model.backbone" in classification.sweep.parameters
    assert detection.sweep.objective == "metrics/mAP50(B)"
    assert "map_50" not in detection.sweep.objective


def test_cli_sweep_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    class FakeSweeper:
        def __init__(self, config, objective=None, debug: bool = False):
            seen["task"] = config.task
            seen["objective"] = config.sweep.objective
            seen["debug"] = debug

        def run(self):
            trial = type("Trial", (), {"value": 0.9, "params": {}})()
            return type("Study", (), {"best_trial": trial})()

    monkeypatch.setattr("wildtrain.cli.commands.sweep.ExperimentSweeper", FakeSweeper)
    result = RUNNER.invoke(app, ["sweep", "-c", str(CLASSIFICATION_SWEEP), "--debug"])
    assert result.exit_code == 0, result.output
    assert seen == {"task": "classification", "objective": "f1_score", "debug": True}


def test_api_only_exposes_post_experiments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def fake_run(config, debug: bool = False, registry=None) -> FitResult:
        seen.append(config.task)
        return FitResult(best_path="best.pt", best_score=0.2, score_name="fitness")

    def inline(task_func, *args, metadata=None, **kwargs):
        task_func(*args, **kwargs)
        return "job-1"

    monkeypatch.setattr(
        "wildtrain.api.services.training_service.run_experiment", fake_run
    )
    monkeypatch.setattr("wildtrain.api.routers.training.create_background_job", inline)

    from wildtrain.api.main import fastapi_app

    client = TestClient(fastapi_app)
    detection = load_experiment(DETECTION_YAML).model_dump(
        mode="json", exclude_none=True
    )
    created = client.post("/experiments", json={"config": detection})
    assert created.status_code == 200, created.text
    assert created.json()["job_id"] == "job-1"
    removed = (
        "/training/experiments",
        "/training/classifier",
        "/training/detector",
    )
    for route in removed:
        response = client.post(route, json={"config": detection})
        assert response.status_code == 404
    assert seen == ["detection"]
