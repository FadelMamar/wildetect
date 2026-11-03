# Classification experiment

Train with:

```bash
wildtrain run -c configs/classification/classification_train.yaml
```

The file is an experiment document, not the old `dataset` / `single_class` config. Set `DATA_ROOT` to the ROI dataset root. A full example is in the WildTrain README and in `wildtrain/configs/classification/classification_train.yaml`.

| Field | Meaning |
|-------|---------|
| `task` | `classification` |
| `name` | Run name. Used as the MLflow run name when `tracking.mlflow.run_name` is omitted |
| `data.format` | `roi` for the product crops. `crop` is optional |
| `data.root` | Dataset root. Checked when training starts |
| `data.label_policy.mode` | `collapse` or `identity`. Omitted means `identity` |
| `data.label_policy.positive_name` | Collapsed foreground name (`wildlife` in the shipped file) |
| `data.label_policy.background_name` | Background class name |
| `data.label_policy.discard` | Class names to drop |
| `model.backend` | `timm` |
| `model.backbone` | timm model name |
| `train.epochs`, `train.lr` | Lightning loop |
| `checkpoint.dirpath` | Checkpoint directory |
| `checkpoint.monitor` | `val_f1score` in the shipped file |

`wildtrain evaluate classifier` still uses `classification_eval.yaml`. The `eval` block on the experiment is recorded only.

Sweep the same document with `wildtrain sweep -c configs/classification/classification_sweep.yaml`. `sweep.objective: f1_score` uses the validation F1 from training.
