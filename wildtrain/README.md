# WildTrain

WildTrain trains the two models the census app uses:

- a class-agnostic Ultralytics YOLO localizer (`task: detection`, `train.single_cls: true`)
- an ROI image classifier on crops from those boxes (`task: classification`, PyTorch Lightning + timm)

Configs are YAML files validated with Pydantic. OmegaConf resolves interpolations such as `${dataset.input_size}`. Experiment tracking uses MLflow. There is no Hydra config group and no MMDetection backend.

## Layout

```
configs/          # example YAML for classification, detection, eval, registration
src/wildtrain/
  harness/        # one experiment schema, registry, sweep, and pipeline
  trainers/       # ClassifierTrainer and UltralyticsDetectionTrainer
  models/         # GenericClassifier, two-stage Detector, MLflow registration
  cli/            # Typer commands
```

## Train

From `wildtrain/` after `uv sync`:

```bash
wildtrain run -c path/to/experiment.yaml
wildtrain sweep -c path/to/sweep.yaml
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml
wildtrain train classifier -c configs/classification/classification_train.yaml
```

`wildtrain run` and `wildtrain sweep` dispatch through the harness registry. `train classifier` and `train detector` are adapters: they load the existing flat YAML and call the same runner. `POST /training/experiments` and `POST /experiments` do the same. `POST /training/classifier` and `POST /training/detector` stay as adapters.

A new experiment file has shared `tracking`, `run`, and optional `sweep` blocks, plus one task body. The body is still a classification or detection training config:

```yaml
task: classification          # classification | detection
tracking:
  mlflow:
    experiment_name: wildtrain_classification
    run_name: base
    tracking_uri: http://localhost:5000
    log_model: false
run:
  name: base
  project: wildtrain_classification
  output_dir: checkpoints/base
  seed: 0
classification:
  dataset: { ... }            # same fields as classification_train.yaml
  model: { ... }
  train: { ... }
  checkpoint:
    dirpath: checkpoints/base
sweep:                        # optional
  n_trials: 8
  direction: maximize         # maximize | minimize
  objective: f1_score         # detection default in old files is map_50
  seed: 1
  parameters:
    train.lr: [0.0003, 0.001]
    model.dropout: [0.0, 0.2]
```

Detection uses a `detection:` body instead of `classification:`. `run.name` and `run.project` are the Ultralytics run name and project. `run.output_dir` is the classification checkpoint directory. When both the shared blocks and the nested body set MLflow or the run name, the shared `run` values are what the trainer sees.

Today's flat YAML still loads. Files with no `task` key are inferred (`checkpoint` or ROI fields mean classification; `curriculum` or `single_cls` mean detection). The old sweep files (`base_config` plus `parameters.model` / `parameters.train`) load too. Their grids become dotted paths such as `train.lr` and `train.lr0`.

`run.seed` is stored on the experiment. The sweep sampler uses `sweep.seed`. The trainers do not read `run.seed` yet.

## Evaluate and register

```bash
wildtrain evaluate detector -c configs/detection/yolo_configs/yolo_eval.yaml
wildtrain evaluate classifier -c configs/classification/classification_eval.yaml
wildtrain register detector configs/registration/detector_registration_example.yaml
```
