"""ROI label policy, shared classification metrics, and a one-epoch CPU train."""

import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from wildtrain.data.classification_datamodule import ClassificationDataModule
from wildtrain.data.label_policy import resolve_label_policy
from wildtrain.evaluators.classification import ClassificationEvaluator
from wildtrain.evaluators.classification_metrics import build_classification_metrics
from wildtrain.harness.config import load_experiment
from wildtrain.harness.runner import run_experiment
from wildtrain.shared.schemas.classification import ClassificationEvalConfig
from wildtrain.trainers import classification_trainer as trainer_module

CLASSES = {1: "background", 2: "elephant", 7: "zebra"}


def _write_split(
    root: Path, dataset: str, split: str, labels: list[tuple[str, int]]
) -> None:
    image_dir = root / "framework_formats" / "roi" / dataset / "images" / split
    label_dir = root / "framework_formats" / "roi" / dataset / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, class_id in labels:
        # Distinct colors so the three classes are not identical pixels.
        color = (class_id * 20 % 255, 40, 80)
        Image.new("RGB", (8, 8), color).save(image_dir / name)
        records.append({"file_name": name, "class_id": class_id})
    mapping = {str(class_id): name for class_id, name in CLASSES.items()}
    (label_dir / "class_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    (label_dir / "roi_labels.json").write_text(json.dumps(records), encoding="utf-8")


def _synthetic_roi(root: Path) -> None:
    train = [
        ("bg0.png", 1),
        ("bg1.png", 1),
        ("el0.png", 2),
        ("el1.png", 2),
        ("zb0.png", 7),
        ("zb1.png", 7),
    ]
    val = [("bg.png", 1), ("el.png", 2), ("zb.png", 7)]
    _write_split(root, "species", "train", train)
    _write_split(root, "species", "val", val)


def _payload(root: Path, checkpoint_dir: Path, mlflow_uri: str, policy: dict) -> dict:
    return {
        "task": "classification",
        "dataset": {
            "root_data_directory": str(root),
            "dataset_type": "roi",
            "input_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "stats": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "rebalance": False,
            "label_policy": policy,
        },
        "model": {
            "backbone": "vit_tiny_patch16_224",
            "pretrained": False,
            "backbone_source": "timm",
            "dropout": 0.0,
            "freeze_backbone": True,
            "input_size": 32,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "hidden_dim": 16,
            "num_layers": 2,
            "weights": None,
        },
        "train": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 1e-3,
            "label_smoothing": 0.0,
            "weight_decay": 0.0,
            "lrf": 0.1,
            "precision": "32",
            "accelerator": "cpu",
            "num_workers": 0,
            "val_check_interval": 1,
        },
        "checkpoint": {
            "monitor": "val_f1score",
            "save_top_k": 1,
            "mode": "max",
            "save_last": False,
            "dirpath": str(checkpoint_dir),
            "patience": 10,
            "save_weights_only": True,
            "filename": "best",
            "min_delta": 0.0,
        },
        "mlflow": {
            "experiment_name": "roi-species-smoke",
            "run_name": "epoch1",
            "log_model": False,
            "tracking_uri": mlflow_uri,
        },
    }


def _datamodule(root: Path, policy: dict | None) -> ClassificationDataModule:
    dataset = {
        "root_data_directory": str(root),
        "dataset_type": "roi",
        "batch_size": 2,
    }
    if policy is not None:
        dataset["label_policy"] = policy
    config = type("Cfg", (), {})()
    config.dataset = dataset
    config.train = {"num_workers": 0}
    return ClassificationDataModule.from_dict_config(config)


def test_omitted_policy_collapses_to_wildlife(tmp_path: Path) -> None:
    _synthetic_roi(tmp_path)
    assert resolve_label_policy().mode == "collapse"
    module = _datamodule(tmp_path, None)
    module.setup(stage="fit")
    assert module.class_mapping == {0: "background", 1: "wildlife"}


def test_identity_keeps_species_and_remaps_ids(tmp_path: Path) -> None:
    _synthetic_roi(tmp_path)
    module = _datamodule(tmp_path, {"mode": "identity", "enable": True})
    module.setup(stage="fit")
    assert module.class_mapping == {0: "background", 1: "elephant", 2: "zebra"}
    labels = {int(module.train_dataset[i][1]) for i in range(len(module.train_dataset))}
    assert labels == {0, 1, 2}


def test_enable_false_is_identity(tmp_path: Path) -> None:
    _synthetic_roi(tmp_path)
    module = _datamodule(
        tmp_path,
        {
            "enable": False,
            "background_class_name": "background",
            "single_class_name": "wildlife",
        },
    )
    module.setup(stage="fit")
    assert set(module.class_mapping.values()) == {"background", "elephant", "zebra"}


def test_train_and_eval_metrics_match() -> None:
    probs = torch.softmax(torch.randn(8, 3), dim=1)
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    trained = build_classification_metrics(3)
    evaluated = build_classification_metrics(3)
    for metric in trained.values():
        metric.update(probs, target)
    for metric in evaluated.values():
        metric.update(probs, target)
    for name in trained:
        left = trained[name].compute()
        right = evaluated[name].compute()
        assert torch.allclose(left, right, equal_nan=True)


def test_multiclass_roi_trains_and_evaluates(tmp_path: Path, monkeypatch) -> None:
    _synthetic_roi(tmp_path)
    tracking = tmp_path / "mlflow.db"
    mlflow_uri = f"sqlite:///{tracking.as_posix()}"
    env_file = tmp_path / ".env"
    env_file.write_text(f"MLFLOW_TRACKING_URI={mlflow_uri}\n", encoding="utf-8")
    monkeypatch.setattr(trainer_module, "ENV_FILE", env_file)

    experiment_path = tmp_path / "experiment.yaml"
    payload = _payload(
        tmp_path,
        tmp_path / "checkpoints",
        mlflow_uri,
        {"mode": "identity"},
    )
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    experiment = load_experiment(experiment_path)

    result = run_experiment(experiment, debug=False)

    assert result.best_path
    assert Path(result.best_path).exists()

    eval_path = tmp_path / "eval.yaml"
    eval_payload = {
        "classifier": result.best_path,
        "split": "val",
        "device": "cpu",
        "batch_size": 2,
        "dataset": {
            "root_data_directory": str(tmp_path),
            "label_policy": {"mode": "identity"},
        },
    }
    eval_path.write_text(yaml.safe_dump(eval_payload), encoding="utf-8")
    evaluator = ClassificationEvaluator(ClassificationEvalConfig.from_yaml(eval_path))
    report = evaluator.evaluate()

    assert set(build_classification_metrics(3)) <= set(report)
    assert "f1_class_elephant" in report
    assert "f1_class_zebra" in report
