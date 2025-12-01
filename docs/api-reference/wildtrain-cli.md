# WildTrain CLI Reference

Complete command-line interface reference for WildTrain.

## Main Commands

### train

Train a model.

```bash
wildtrain train <task> [OPTIONS]
```

**Arguments:**
- `task`: Task type (classifier/detector)

**Options:**
- `-c, --config PATH`: Config file (required)
- `--resume PATH`: Resume from checkpoint
- `--dry-run`: Dry run without training

**Examples:**
```bash
# Train classifier
wildtrain train classifier -c configs/classification/train.yaml

# Train detector
wildtrain train detector -c configs/detection/yolo.yaml
```

### eval

Evaluate a trained model.

```bash
wildtrain eval <task> [OPTIONS]
```

**Arguments:**
- `task`: Task type (classifier/detector)

**Options:**
- `-c, --config PATH`: Config file (required)
- `--checkpoint PATH`: Model checkpoint
- `--split TEXT`: Dataset split (test/val)

**Examples:**
```bash
wildtrain eval classifier -c configs/classification/eval.yaml
wildtrain eval detector -c configs/detection/yolo_eval.yaml
```

### register

Register model to MLflow registry.

```bash
wildtrain register <model_type> <config>
```

**Arguments:**
- `model_type`: Model type (classifier/detector)
- `config`: Registration config file

**Example:**
```bash
wildtrain register detector configs/registration/detector_registration.yaml
```

### tune

Run hyperparameter tuning.

```bash
wildtrain tune <task> [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Config file
- `--n-trials INTEGER`: Number of trials

**Example:**
```bash
wildtrain tune classifier -c configs/classification/sweep.yaml --n-trials 50
```

### serve

Start inference server.

```bash
wildtrain serve [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Inference config
- `--port INTEGER`: Server port
- `--workers INTEGER`: Number of workers

---

For detailed script documentation, see [WildTrain Scripts](../scripts/wildtrain/index.md).

