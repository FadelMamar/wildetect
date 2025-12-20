# WildTrain: Modular Computer Vision Framework

## Overview
WildTrain is a modular, scalable computer vision framework supporting object detection (YOLOv8, MMDetection) and image classification (PyTorch Lightning). It follows Clean Architecture, SOLID principles, and uses Hydra for configuration and MLflow for experiment tracking.

## Project Structure
```
configs/         # Hydra/YAML configs for experiments
models/          # Model definitions (detection/classification)
tasks/           # Task logic (training, evaluation)
trainers/        # Training loops and orchestration
utils/           # Utilities, logging, registry, etc.
data/            # Data modules, transforms, loaders
scripts/         # Example scripts for training/eval
main.py          # CLI entry point
requirements.txt # Python dependencies
pyproject.toml   # Project metadata
```

## Features
- Object Detection: YOLOv8, MMDetection
- Image Classification: PyTorch Lightning
- Hydra-configurable experiments
- MLflow tracking
- Optuna/Ray Tune hyperparameter tuning
- Modular, extensible, type-annotated codebase

## Usage
### Install MMDet
```bash
uv pip install torch==2.0.0 torchvision
uv pip install -U openmim
uv run mim install mmengine
```
- If CPU only:
``uv pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html``
- If CUDA
``uv pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html``
- Mopre info at: https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv-lite

- Install mmdet
``uv run mim install mmdet``
``uv pip install numpy==1.26.4``


### Train a model
```bash
python main.py task=detection framework=yolov8
python main.py task=classification
```

### Hyperparameter tuning
```bash
python main.py task=classification tune=true
```

### Evaluate
```bash
python main.py task=detection mode=evaluate
```

## Configuration
Edit configs in `configs/` or override via CLI using Hydra syntax.

## Contributing
PRs and issues welcome!