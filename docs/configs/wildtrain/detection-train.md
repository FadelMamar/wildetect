# Detection experiment

Train with:

```bash
wildtrain run -c configs/detection/yolo_configs/yolo.yaml
```

Set `DETECTION_DATA_CFG` to the YOLO data yaml. It is opened when training starts. The shipped file is the class-agnostic localizer: `model.backend: ultralytics`, `train.single_cls: true`, `model.weights: yolo12s.pt`.

| Field | Meaning |
|-------|---------|
| `task` | `detection` |
| `name` | Ultralytics run name |
| `project` | Ultralytics project directory |
| `data.format` | `yolo` |
| `data.data_cfg` | Path to the YOLO data yaml |
| `data.discard` | Class names removed before training |
| `model.backend` | `ultralytics` or `custom_yolo` |
| `model.weights` | Starting weights |
| `model.architecture` | Optional architecture yaml, such as `yolov8n.yaml` |
| `train.batch_size` | Ultralytics batch |
| `train.imgsz` | Image size. Registration still stores this as `imgsz` |
| `train.single_cls` | `true` is the product default. `false` keeps dataset class ids |
| `train.lr0` | Initial learning rate |

`model.backend: custom_yolo` is the only path that imports the custom YOLO fork. Put fork settings under `model.custom`.

## Sweep

```bash
wildtrain sweep -c configs/detection/detection_sweep.yaml
```

`sweep.parameters` uses dotted paths such as `train.lr0` and `train.batch_size`. `sweep.objective` must be an Ultralytics metrics key exactly. This install reports:

- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `fitness`

`map_50` is not rewritten to `metrics/mAP50(B)`.

## Evaluate and register

`wildtrain evaluate -c configs/detection/yolo_configs/yolo_eval.yaml` scores that eval document. Registration still writes `batch`, `imgsz`, `task`, `num_classes`, `model_type`, and `cls_export_format` for census.
