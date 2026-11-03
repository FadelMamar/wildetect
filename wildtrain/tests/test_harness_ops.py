"""Eval, visualization, registration, and config validate on the harness."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from wildtrain.cli.cli import app
from wildtrain.harness.config import load_experiment
from wildtrain.harness.ops import validate_harness_file
from wildtrain.harness.registry import TaskRegistry
from wildtrain.harness.result import FitResult
from wildtrain.harness.runner import run_experiment
from wildtrain.harness.tasks.classification import ClassificationTask
from wildtrain.harness.tasks.detection import DetectionTask

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "wildtrain" / "configs"
RUNNER = CliRunner()


def test_shipped_files_match_their_kinds() -> None:
    expected = {
        CONFIGS / "classification" / "classification_train.yaml": "experiment",
        CONFIGS / "detection" / "yolo_configs" / "yolo.yaml": "experiment",
        CONFIGS / "classification" / "classification_eval.yaml": "eval",
        CONFIGS / "detection" / "yolo_configs" / "yolo_eval.yaml": "eval",
        CONFIGS
        / "classification"
        / "classification_visualization.yaml": "visualization",
        CONFIGS / "detection" / "visualization.yaml": "visualization",
        CONFIGS
        / "registration"
        / "classifier_registration_example.yaml": "registration",
        CONFIGS
        / "registration"
        / "detector_registration_example.yaml": "registration",
    }
    for path, kind in expected.items():
        assert validate_harness_file(path) == kind


def test_evaluate_refuses_to_train_when_weights_are_missing() -> None:
    experiment = load_experiment(
        CONFIGS / "classification" / "classification_train.yaml"
    )
    experiment.model.weights = None
    experiment.checkpoint.dirpath = str(CONFIGS / "missing-checkpoints")
    called = {"run": False}

    def fail_run(self, config, debug: bool = False):
        called["run"] = True
        raise AssertionError("evaluate started training")

    monkeypatch_run = ClassificationTask.run
    ClassificationTask.run = fail_run
    try:
        try:
            ClassificationTask().evaluate(experiment)
        except ValueError as exc:
            assert "Train first" in str(exc)
            assert "model.weights" in str(exc)
        else:
            raise AssertionError("missing weights were accepted")
    finally:
        ClassificationTask.run = monkeypatch_run
    assert called["run"] is False


def test_old_pipeline_yaml_names_eval_enabled(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "disable_train": False,
                "train": {"config": "configs/classification/classification_train.yaml"},
                "disable_eval": False,
                "eval": {"config": "configs/classification/classification_eval.yaml"},
                "results_dir": "results/classification",
            }
        ),
        encoding="utf-8",
    )
    try:
        validate_harness_file(path)
    except ValueError as exc:
        assert "eval.enabled" in str(exc)
        assert "wildtrain run -c" in str(exc)
    else:
        raise AssertionError("old pipeline file was accepted")


def test_old_eval_yaml_explains_the_replacement(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "classifier": "best.ckpt",
                "dataset": {
                    "root_data_directory": str(tmp_path),
                    "single_class": {"single_class_name": "wildlife"},
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        validate_harness_file(path)
    except ValueError as exc:
        assert "data.label_policy.positive_name" in str(exc)
    else:
        raise AssertionError("old eval file was accepted")


def test_config_validate_cli_accepts_shipped_documents() -> None:
    documents = [
        (CONFIGS / "classification" / "classification_train.yaml", "experiment"),
        (CONFIGS / "classification" / "classification_eval.yaml", "eval"),
        (CONFIGS / "detection" / "visualization.yaml", "visualization"),
        (
            CONFIGS / "registration" / "detector_registration_example.yaml",
            "registration",
        ),
    ]
    for path, kind in documents:
        result = RUNNER.invoke(app, ["config", "validate", str(path), "--type", kind])
        assert result.exit_code == 0, result.output
        assert kind in result.output


def test_tasks_evaluate_visualize_and_register_through_the_registry(
    monkeypatch, tmp_path: Path
) -> None:
    seen = []

    def record(name):
        def _call(self, config, *args, **kwargs):
            seen.append((name, config.task))
            return {"f1": 1.0}

        return _call

    monkeypatch.setattr(ClassificationTask, "evaluate", record("eval"))
    monkeypatch.setattr(DetectionTask, "evaluate", record("eval"))
    monkeypatch.setattr(ClassificationTask, "visualize", record("viz"))
    monkeypatch.setattr(DetectionTask, "visualize", record("viz"))
    monkeypatch.setattr(ClassificationTask, "register", record("register"))
    monkeypatch.setattr(DetectionTask, "register", record("register"))

    registry = TaskRegistry()
    registry.register(ClassificationTask())
    registry.register(DetectionTask())
    for task_name, method in (
        ("classification", "evaluate"),
        ("detection", "visualize"),
        ("classification", "register"),
    ):
        document = type("Doc", (), {"task": task_name})()
        getattr(registry.get(task_name), method)(document)
    assert seen == [
        ("eval", "classification"),
        ("viz", "detection"),
        ("register", "classification"),
    ]


def test_run_evaluates_when_the_eval_block_is_enabled(monkeypatch) -> None:
    experiment = load_experiment(
        CONFIGS / "classification" / "classification_train.yaml"
    )
    experiment.eval.enabled = True
    called = {}

    class FakeTask:
        name = "classification"

        def run(self, config, debug: bool = False) -> FitResult:
            return FitResult(best_path="best.ckpt", best_score=0.5, score_name="f1")

        def evaluate(self, config, weights=None, debug: bool = False):
            called["weights"] = weights
            return {"f1": [1.0]}

    monkeypatch.setattr(
        "wildtrain.harness.runner.default_registry",
        lambda: type("R", (), {"get": lambda self, name: FakeTask()})(),
    )
    result = run_experiment(experiment)
    assert called["weights"] == "best.ckpt"
    assert result.metrics["eval"]["f1"] == [1.0]
