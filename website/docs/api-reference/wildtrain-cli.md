# WildTrain CLI Reference

Complete command-line interface reference for the `wildtrain` CLI.

WildTrain uses [Typer](https://typer.tiangolo.com/) with nested subcommand groups for training, evaluation, model registration, pipelines, and visualization.

```bash
wildtrain [COMMAND_GROUP] [COMMAND] [OPTIONS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Enable verbose logging |
| `-c`, `--config-dir` | Configuration directory |
| `--help` | Show help message and exit |

---

## `train` — Training Commands

Train detection and classification models.

### `train classifier`

Train a classification model using PyTorch Lightning.

```bash
wildtrain train classifier [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to training configuration YAML file |

**Example:**

```bash
wildtrain train classifier -c configs/classification/classification_train.yaml
```

See [Classification Training Config](../configs/wildtrain/classification-train.md) for config details.

---

### `train detector`

Train an object detection model (YOLO via Ultralytics).

```bash
wildtrain train detector [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to training configuration YAML file |

**Example:**

```bash
wildtrain train detector -c configs/detection/yolo_configs/yolo_train.yaml
```

See [Detection Training Config](../configs/wildtrain/detection-train.md) for config details.

---

## `evaluate` — Evaluation Commands

Evaluate trained models on test/validation datasets.

### `evaluate classifier`

Evaluate a classification model using a YAML config file.

```bash
wildtrain evaluate classifier [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to classification evaluation YAML config |
| `--debug` | `bool` | `false` | Enable debug mode |

**Example:**

```bash
wildtrain evaluate classifier -c configs/classification/classification_eval.yaml
```

---

### `evaluate detector`

Evaluate a YOLO detection model using a YAML config file.

```bash
wildtrain evaluate detector [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to YOLO evaluation YAML config |
| `--debug` | `bool` | `false` | Enable debug mode |

**Example:**

```bash
wildtrain evaluate detector -c configs/detection/detection_sweep.yaml
```

---

### `evaluate yolo-model`

Run direct Ultralytics YOLO model validation (bypass WildTrain wrapper).

```bash
wildtrain evaluate yolo-model [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to YOLO evaluation YAML config |

The config YAML should contain a `model` key (path to weights) plus any Ultralytics `val()` parameters.

---

## `register` — Model Registration Commands

Register trained models to the MLflow Model Registry.

### `register classifier`

Register a classification model to MLflow.

```bash
wildtrain register classifier [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to registration configuration file |
| `--weights-path` | `PATH` | `None` | Path to model checkpoint file |
| `-n`, `--name` | `str` | `classifier` | Model name for registration |
| `-b`, `--batch-size` | `int` | `8` | Batch size for inference |
| `--mlflow-uri` | `str` | `http://localhost:5000` | MLflow tracking server URI |

You can use either `--config` or provide options directly. When using `--config`, don't provide other options.

**Examples:**

```bash
# Using config file
wildtrain register classifier -c configs/registration/classifier_registration_example.yaml

# Using direct options
wildtrain register classifier --weights-path model.ckpt --name my_classifier --mlflow-uri http://localhost:5000
```

See [Registration Config](../configs/wildtrain/registration.md) for config details.

---

### `register detector`

Register a detection model to MLflow.

```bash
wildtrain register detector CONFIG_PATH
```

| Argument | Type | Description |
|----------|------|-------------|
| `CONFIG_PATH` | `PATH` | Path to detector registration configuration file |

**Example:**

```bash
wildtrain register detector configs/registration/detector_registration_example.yaml
```

---

## `pipeline` — Pipeline Commands

Run full train + eval pipelines in a single command.

### `pipeline detection`

Run the full detection pipeline (train + evaluate).

```bash
wildtrain pipeline detection [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to unified detection pipeline YAML config |

**Example:**

```bash
wildtrain pipeline detection -c configs/detection/detection_sweep.yaml
```

---

### `pipeline classification`

Run the full classification pipeline (train + evaluate).

```bash
wildtrain pipeline classification [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to unified classification pipeline YAML config |

**Example:**

```bash
wildtrain pipeline classification -c configs/classification/classification_pipeline_config.yaml
```

---

## `config` — Configuration Management

Validate and generate configuration templates.

### `config validate`

Validate a configuration file against Pydantic models.

```bash
wildtrain config validate CONFIG_PATH [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `CONFIG_PATH` | `PATH` | *(required)* | Path to configuration file |
| `-t`, `--type` | `str` | `classification` | Config type (see below) |

**Supported config types:** `classification`, `detection`, `classification_eval`, `detection_eval`, `classification_visualization`, `detection_visualization`, `pipeline`, `detector_registration`, `classifier_registration`, `model_registration`

**Example:**

```bash
wildtrain config validate configs/classification/classification_train.yaml --type classification
```

---

### `config template`

Generate a default YAML configuration template.

```bash
wildtrain config template CONFIG_TYPE [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `CONFIG_TYPE` | `str` | *(required)* | Configuration type to generate template for |
| `-s`, `--save` | `PATH` | `None` | Save template to file (prints to stdout if omitted) |

**Example:**

```bash
# Print template to console
wildtrain config template classification

# Save to file
wildtrain config template detection -s my_detection_config.yaml
```

---

## `dataset` — Dataset Commands

Dataset analysis and statistics.

### `dataset stats`

Compute dataset statistics (mean, standard deviation) for normalization.

```bash
wildtrain dataset stats DATA_DIR [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `DATA_DIR` | `PATH` | *(required)* | Path to dataset directory |
| `--split` | `str` | `train` | Split to compute statistics for |
| `-o`, `--output` | `PATH` | `None` | Output file for statistics JSON |

**Example:**

```bash
wildtrain dataset stats D:/data/roi_dataset --split train -o stats.json
```

---

## `visualize` — Visualization Commands

Upload model predictions to FiftyOne for interactive visualization.

### `visualize classifier-predictions`

Upload classifier predictions to a FiftyOne dataset.

```bash
wildtrain visualize classifier-predictions [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to classification visualization YAML config |

**Example:**

```bash
wildtrain visualize classifier-predictions -c configs/classification/classification_visualization.yaml
```

---

### `visualize detector-predictions`

Upload detector predictions to a FiftyOne dataset.

```bash
wildtrain visualize detector-predictions [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `""` | Path to detection visualization YAML config |

**Example:**

```bash
wildtrain visualize detector-predictions -c configs/detection/visualization.yaml
```

---

## `run-server` — Inference Server

Start a LitServe-based inference server for model serving.

```bash
wildtrain run-server [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | `int` | `4141` | Port to run the server on |
| `-w` | `int` | `1` | Number of workers per device |
| `-c`, `--config` | `PATH` | `None` | Path to inference config file |

When a config file is provided, it sets MLflow environment variables and overrides port/workers.

**Example:**

```bash
# Using config file
wildtrain run-server -c configs/inference.yaml

# Using direct options
wildtrain run-server --port 4141 -w 2
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `wildtrain train classifier -c CONFIG` | Train a classifier |
| `wildtrain train detector -c CONFIG` | Train a YOLO detector |
| `wildtrain evaluate classifier -c CONFIG` | Evaluate a classifier |
| `wildtrain evaluate detector -c CONFIG` | Evaluate a detector |
| `wildtrain register classifier -c CONFIG` | Register classifier to MLflow |
| `wildtrain register detector CONFIG` | Register detector to MLflow |
| `wildtrain pipeline detection -c CONFIG` | Full detection pipeline |
| `wildtrain pipeline classification -c CONFIG` | Full classification pipeline |
| `wildtrain config validate CONFIG --type TYPE` | Validate a config file |
| `wildtrain config template TYPE` | Generate config template |
| `wildtrain dataset stats DATA_DIR` | Compute dataset stats |
| `wildtrain visualize classifier-predictions -c CONFIG` | Visualize classifier predictions |
| `wildtrain visualize detector-predictions -c CONFIG` | Visualize detector predictions |
| `wildtrain run-server -c CONFIG` | Start inference server |

---

For configuration file details, see [Configuration Reference](../configs/wildtrain/index.md).  
For shell scripts, see [WildTrain Scripts](../scripts/wildtrain/index.md).
