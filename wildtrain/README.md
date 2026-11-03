# WildTrain

WildTrain trains the two models the census app uses:

- a class-agnostic Ultralytics YOLO localizer (`task: detection`, `train.single_cls: true`)
- an ROI image classifier on crops from those boxes (`task: classification`, PyTorch Lightning + timm)

Configs are YAML files validated with Pydantic. OmegaConf resolves interpolations such as `${dataset.input_size}`. Experiment tracking uses MLflow. There is no Hydra config group and no MMDetection backend.

## Layout

```
configs/          # example YAML for classification, detection, eval, registration
src/wildtrain/
  harness/        # task registry and `wildtrain run`
  trainers/       # ClassifierTrainer and UltralyticsDetectionTrainer
  models/         # GenericClassifier, two-stage Detector, MLflow registration
  cli/            # Typer commands
```

## Train

From `wildtrain/` after `uv sync`:

```bash
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml
wildtrain train classifier -c configs/classification/classification_train.yaml
```

The same documents work with `wildtrain run` once they contain a top-level `task` key (`detection` or `classification`):

```bash
wildtrain run -c path/to/experiment.yaml
```

`wildtrain train classifier` and `wildtrain train detector` stay the commands for the existing files, which do not have a `task` key.

## Evaluate and register

```bash
wildtrain evaluate detector -c configs/detection/yolo_configs/yolo_eval.yaml
wildtrain evaluate classifier -c configs/classification/classification_eval.yaml
wildtrain register detector configs/registration/detector_registration_example.yaml
```
