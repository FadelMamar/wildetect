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

## `run` — Train an experiment

Train the task named in the experiment file. `wildtrain train classifier` and `wildtrain train detector` have been removed. `wildtrain train` prints that message and exits.

```bash
wildtrain run -c configs/classification/classification_train.yaml
wildtrain run -c configs/detection/yolo_configs/yolo.yaml
wildtrain run -c path/to/experiment.yaml --debug
```

| Option | Type | Description |
|--------|------|-------------|
| `-c`, `--config` | `PATH` | Experiment YAML (`task`, `tracking`, `data`, `model`, `train`) |
| `--debug` | flag | Short debug training pass |

Classification uses `data.format: roi` and `data.label_policy` (`mode`, `positive_name`, `background_name`, `discard`). Detection uses `data.format: yolo`, `model.backend` (`ultralytics` or `custom_yolo`), and `train.batch_size`. An old flat training file is rejected with the field names to change. Full examples are in the WildTrain README.

---

## `sweep` — One Optuna study

```bash
wildtrain sweep -c configs/classification/classification_sweep.yaml
wildtrain sweep -c configs/detection/detection_sweep.yaml
```

| Option | Type | Description |
|--------|------|-------------|
| `-c`, `--config` | `PATH` | Experiment file that includes a `sweep` block |
| `--debug` | flag | Short debug training trials |

`sweep.parameters` maps a dotted path to a list of choices (`train.lr`, `model.backbone`, `train.lr0`, `train.batch_size`). A detection `objective` must match an Ultralytics metrics key exactly. Examples: `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `fitness`. `map_50` is not translated. Classification `f1_score` uses the validation F1 on the training result.

---

## `evaluate` — Evaluation

Score the task named in an experiment file or a standalone eval document.

```bash
wildtrain evaluate -c configs/classification/classification_eval.yaml
wildtrain evaluate -c configs/detection/yolo_configs/yolo_eval.yaml
```

| Option | Type | Description |
|--------|------|-------------|
| `-c`, `--config` | `PATH` | Experiment YAML, or a standalone eval document |
| `--debug` | flag | Score a few batches only |

A standalone file has `task`, `weights`, `data`, and `eval`. An experiment is accepted too. `wildtrain run` evaluates after training when that experiment sets `eval.enabled: true`. `wildtrain evaluate classifier`, `wildtrain evaluate detector`, and `wildtrain evaluate yolo-model` are removed.

---

## `register` — Model registration

Register the task named in a registration document to the MLflow Model Registry.

```bash
wildtrain register -c configs/registration/classifier_registration_example.yaml
wildtrain register -c configs/registration/detector_registration_example.yaml
```

| Option | Type | Description |
|--------|------|-------------|
| `-c`, `--config` | `PATH` | Registration document |

The registered metadata is `batch`, `imgsz`, `task`, `num_classes`, `model_type`, and `cls_export_format`. A detector document sets `localizer.weights`. Add `classifier_weights` for the two-stage localizer plus ROI classifier. `wildtrain register classifier` and `wildtrain register detector` are removed.

See [Registration Config](../configs/wildtrain/registration.md).

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
| `-t`, `--type` | `str` | auto | `experiment`, `eval`, `visualization`, `registration`, or `pipeline` |

With no `--type`, the file is detected from its keys. Old trainer types such as `classification` are not accepted.

**Example:**

```bash
wildtrain config validate configs/classification/classification_train.yaml
wildtrain config validate configs/classification/classification_eval.yaml --type eval
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

## `visualize` — Visualization

Upload predictions for the task named in the file. Classification and detection both go through FiftyOne. The detection file can also carry Label Studio fields.

```bash
wildtrain visualize -c configs/classification/classification_visualization.yaml
wildtrain visualize -c configs/detection/visualization.yaml
```

| Option | Type | Description |
|--------|------|-------------|
| `-c`, `--config` | `PATH` | Visualization document (`task`, `dataset_name`, `weights`) |

`wildtrain visualize classifier-predictions` and `wildtrain visualize detector-predictions` are removed.

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
| `wildtrain run -c CONFIG` | Train the task named in an experiment file |
| `wildtrain sweep -c CONFIG` | Optuna sweep for either task |
| `wildtrain evaluate -c CONFIG` | Evaluate an experiment or eval document |
| `wildtrain register -c CONFIG` | Register a classifier or detector |
| `wildtrain pipeline detection -c CONFIG` | Full detection pipeline |
| `wildtrain pipeline classification -c CONFIG` | Full classification pipeline |
| `wildtrain config validate CONFIG` | Validate an experiment, eval, visualization, or registration file |
| `wildtrain config template TYPE` | Generate config template |
| `wildtrain dataset stats DATA_DIR` | Compute dataset stats |
| `wildtrain visualize -c CONFIG` | Upload predictions for the task in the file |
| `wildtrain run-server -c CONFIG` | Start inference server |

---

For configuration file details, see [Configuration Reference](../configs/wildtrain/index.md).  
For shell scripts, see [WildTrain Scripts](../scripts/wildtrain/index.md).
