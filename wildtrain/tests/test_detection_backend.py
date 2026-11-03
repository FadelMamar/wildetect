"""Detection backend boundary, single-class contract, and a tiny YOLO run."""

from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from PIL import Image

from wildtrain.harness.detection.backends import (
    CustomYoloBackend,
    UltralyticsBackend,
    select_detection_backend,
)
from wildtrain.harness.detection.metrics import build_detection_metrics
from wildtrain.models.localizer import (
    UltralyticsLocalizer,
    apply_single_class_id_shift,
)
from wildtrain.shared.schemas.detection import DetectionConfig
from wildtrain.trainers.detection_trainer import UltralyticsDetectionTrainer


def _box() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )


def _data_yaml(root: Path, names: dict[int, str]) -> Path:
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in (0, 1):
            Image.new("RGB", (64, 64), (20 + index, 40, 60)).save(
                image_dir / f"sample{index}.png"
            )
            (label_dir / f"sample{index}.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": [str(root / "images" / "train")],
                "val": [str(root / "images" / "val")],
                "names": names,
                "nc": len(names),
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(root: Path, **train_overrides) -> DetectionConfig:
    train = {
        "batch": 2,
        "epochs": 1,
        "lr0": 0.01,
        "lrf": 0.1,
        "imgsz": 64,
        "device": "cpu",
        "workers": 0,
        "freeze": 0,
        "single_cls": True,
    }
    train.update(train_overrides)
    payload = {
        "dataset": {
            "data_cfg": str(_data_yaml(root, {0: "wildlife", 1: "elephant"})),
            "discard_classes": ["elephant"],
        },
        "curriculum": {"enable": False},
        "pretraining": {"data_cfg": None},
        "model": {
            "pretrained": False,
            "weights": None,
            "architecture_file": "yolov8n.yaml",
        },
        "name": "smoke",
        "project": str(root / "runs"),
        "mlflow": {
            "experiment_name": "detection-smoke",
            "run_name": "epoch1",
            "alias": "demo",
            "name": "localizer",
            "tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        },
        "use_custom_yolo": False,
        "custom_yolo_kwargs": {},
        "train": train,
    }
    return DetectionConfig.model_validate(payload)


def test_default_backend_is_ultralytics_without_custom_fork() -> None:
    import sys

    sys.modules.pop("wildtrain.trainers.yolo_utils", None)
    config = type("Cfg", (), {"use_custom_yolo": False})()
    backend = select_detection_backend(config)
    assert isinstance(backend, UltralyticsBackend)
    assert "wildtrain.trainers.yolo_utils" not in sys.modules


def test_custom_backend_is_opt_in(monkeypatch) -> None:
    built = {}

    class FakeYOLO:
        def __init__(self, **kwargs):
            built["kwargs"] = kwargs

        def load(self, weights):
            built["loaded"] = weights

    import sys
    import types

    module = types.ModuleType("wildtrain.trainers.yolo_utils")
    module.CustomYOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "wildtrain.trainers.yolo_utils", module)

    config = type("Cfg", (), {})()
    config.use_custom_yolo = True
    config.model = type("Model", (), {"architecture_file": None, "weights": "w.pt"})()
    config.custom_yolo_kwargs = {"box_size": 32, "count_loss_weight": 0.5}
    backend = select_detection_backend(config)
    assert isinstance(backend, CustomYoloBackend)
    backend.build(config)
    assert built["kwargs"]["box_size"] == 32
    assert built["kwargs"]["model"] == "w.pt"
    assert built["loaded"] == "w.pt"


def test_single_class_id_shift_for_one_name_only() -> None:
    shifted = apply_single_class_id_shift(_box(), {0: "wildlife"})
    assert shifted.class_id.tolist() == [1]
    assert shifted.metadata["class_mapping"] == {0: "wildlife"}

    unchanged = apply_single_class_id_shift(_box(), {0: "elephant", 1: "zebra"})
    assert unchanged.class_id.tolist() == [0]

    localizer = object.__new__(UltralyticsLocalizer)
    localizer.model = type("Model", (), {"names": {0: "wildlife"}})()
    via_localizer = localizer._update_class_mapping(_box())
    assert via_localizer.class_id.tolist() == [1]


def test_single_cls_false_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    recorded = {}

    class FakeModel:
        def __init__(self):
            self.trainer = type(
                "Inner",
                (),
                {"best": "best.pt", "best_fitness": 0.2, "metrics": {}},
            )()

        def train(self, **kwargs):
            recorded.update(kwargs)

    config = _config(tmp_path, single_cls=False)
    assert config.train.single_cls is False
    trainer = UltralyticsDetectionTrainer(config)
    monkeypatch.setattr(trainer, "get_model", lambda: FakeModel())
    trainer.run(debug=True)
    assert recorded["single_cls"] is False


def test_single_cls_defaults_to_true(tmp_path: Path, monkeypatch) -> None:
    recorded = {}

    class FakeModel:
        def __init__(self):
            self.trainer = type(
                "Inner",
                (),
                {"best": "best.pt", "best_fitness": 0.2, "metrics": {}},
            )()

        def train(self, **kwargs):
            recorded.update(kwargs)

    payload = _config(tmp_path).model_dump()
    payload["train"].pop("single_cls")
    config = DetectionConfig.model_validate(payload)
    assert config.train.single_cls is True
    trainer = UltralyticsDetectionTrainer(config)
    monkeypatch.setattr(trainer, "get_model", lambda: FakeModel())
    trainer.run(debug=True)
    assert recorded["single_cls"] is True


def test_detection_metrics_score_a_perfect_box() -> None:
    pred = _box()
    target = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        class_id=np.array([0]),
    )
    metric = build_detection_metrics(class_agnostic=True, average="macro")["mAP"]
    metric.update(pred, target)
    score = metric.compute()
    assert float(score.map50) == 1.0


def test_tiny_ultralytics_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    trainer = UltralyticsDetectionTrainer(config)
    trainer.run(debug=False)
    assert trainer.best_model_path
    assert Path(trainer.best_model_path).exists()
