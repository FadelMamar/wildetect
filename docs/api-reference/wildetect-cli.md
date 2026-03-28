# WildDetect CLI Reference

Complete command-line interface reference for the `wildetect` CLI.

WildDetect uses [Typer](https://typer.tiangolo.com/) with nested subcommand groups. All commands are config-driven — you provide a YAML configuration file rather than many individual flags.

```bash
wildetect [COMMAND_GROUP] [COMMAND] [OPTIONS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show the package version and exit |
| `--help` | Show help message and exit |

---

## `detection` — Detection and Analysis Commands

Core commands for running wildlife detection, census campaigns, and analysis.

### `detection detect`

Run wildlife detection on images using AI models.

```bash
wildetect detection detect [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to YAML configuration file |

The detection command is **config-driven**. All parameters (model, image paths, processing options, etc.) are specified in the YAML config file.

**Example:**

```bash
wildetect detection detect -c config/detection.yaml
```

See [Detection Config Reference](../configs/wildetect/detection.md) for all configuration fields.

---

### `detection census`

Run a wildlife census campaign with detection, statistics, geographic analysis, and reporting.

```bash
wildetect detection census [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to YAML census configuration file |

**Example:**

```bash
wildetect detection census -c config/census.yaml
```

The census command runs a complete campaign pipeline:

1. Load images (from directory, paths list, or Label Studio)
2. Run detection pipeline
3. Compute species counts and statistics
4. Export to FiftyOne / Label Studio
5. Generate geographic visualizations

See [Census Config Reference](../configs/wildetect/census.md) for all configuration fields.

---

### `detection analyze`

Analyze detection results with geographic and statistical analysis.

```bash
wildetect detection analyze RESULTS_PATH [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `RESULTS_PATH` | `PATH` | *(required)* | Path to detection results JSON file |
| `-o`, `--output` | `PATH` | `analysis` | Output directory for analysis results |
| `-v`, `--verbose` | `bool` | `false` | Enable verbose logging |

**Example:**

```bash
wildetect detection analyze results/results.json -o analysis_output/
```

---

## `visualization` — Visualization Commands

Commands for geographic visualization and GPS coordinate extraction.

### `visualization visualize`

Create geographic visualizations from detection results.

```bash
wildetect visualization visualize [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to YAML visualization configuration file |
| `-v`, `--verbose` | `bool` | `false` | Enable verbose logging |

!!! warning
    This command is currently not fully implemented (raises `NotImplementedError`).

See [Visualization Config Reference](../configs/wildetect/visualization.md) for configuration details.

---

### `visualization extract-gps-coordinates`

Extract GPS coordinates from images and export detection/annotation reports to CSV.

```bash
wildetect visualization extract-gps-coordinates [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to YAML configuration file |
| `-v`, `--verbose` | `bool` | `false` | Enable verbose logging |

Uses the `visualize` config type. The detection type can be set to `annotations` or `predictions` to control what is exported.

See [GPS Extraction Config Reference](../configs/wildetect/extract-gps.md).

---

## `services` — Service Management Commands

Commands for launching web interfaces, APIs, and external tools.

### `services ui`

Launch the WildDetect Streamlit web interface.

```bash
wildetect services ui [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-p`, `--port` | `int` | `8501` | Port to run the UI on |
| `-h`, `--host` | `str` | `localhost` | Host to bind to |
| `--no-browser` | `bool` | `true` | Don't open browser automatically |

**Example:**

```bash
wildetect services ui --port 8502
```

---

### `services api`

Launch the WildDetect FastAPI REST server.

```bash
wildetect services api [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h`, `--host` | `str` | `0.0.0.0` | Host to bind to |
| `-p`, `--port` | `int` | `8000` | Port to run the API on |
| `--reload` | `bool` | `false` | Enable auto-reload for development |

**Example:**

```bash
wildetect services api --port 8080 --reload
```

API documentation is available at `http://<host>:<port>/docs` (Swagger) and `http://<host>:<port>/redoc`.

---

### `services fiftyone`

Manage FiftyOne datasets for wildlife detection — launch the app, get dataset info, or export data.

```bash
wildetect services fiftyone [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-d`, `--dataset` | `str` | `wildlife_detection` | Dataset name |
| `-a`, `--action` | `str` | `launch` | Action: `launch`, `info`, or `export` |
| `-f`, `--format` | `str` | `coco` | Export format (`coco`, `yolo`, `pascal`) |
| `-o`, `--output` | `PATH` | `None` | Export output path |

**Examples:**

```bash
# Launch FiftyOne app
wildetect services fiftyone -a launch

# Get dataset info
wildetect services fiftyone -a info -d my_dataset

# Export dataset
wildetect services fiftyone -a export -d my_dataset -f coco -o exports/
```

---

## `utils` — Utility Commands

System utilities, configuration helpers, and maintenance.

### `utils info`

Display system information, dependency status, and environment variables.

```bash
wildetect utils info
```

Shows: Python version, PyTorch version, CUDA availability, GPU info, installed dependencies, project paths, and key environment variables.

---

### `utils clear-results`

Delete all detection results in a specified directory (with confirmation prompt).

```bash
wildetect utils clear-results [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--results-dir` | `str` | `results` | Directory containing results to clear |

---

### `utils install-cuda`

Install PyTorch with CUDA support for optimal performance.

```bash
wildetect utils install-cuda [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--cuda-version` | `str` | `None` | Specific CUDA version (`118`, `121`) |
| `--cpu-only` | `bool` | `false` | Force CPU-only installation |
| `-v`, `--verbose` | `bool` | `false` | Verbose logging |

---

### `utils create-config`

Generate a default YAML configuration file for a command type.

```bash
wildetect utils create-config COMMAND_TYPE [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `COMMAND_TYPE` | `str` | *(required)* | Command type: `detect`, `census`, or `visualize` |
| `-o`, `--output` | `PATH` | `None` | Output path for the config file |
| `--pydantic` | `bool` | `true` | Use Pydantic models for validation |

**Example:**

```bash
# Generate a default detection config
wildetect utils create-config detect -o my_detection_config.yaml
```

---

## `benchmarking` — Benchmarking Commands

Performance testing and hyperparameter optimization for detection pipelines.

### `benchmarking detection`

Run detection pipeline benchmarking with Optuna hyperparameter optimization.

```bash
wildetect benchmarking detection [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | *(required)* | Path to benchmark configuration file |

**Example:**

```bash
wildetect benchmarking detection -c config/benchmark.yaml
```

The benchmark command will:

1. Load the benchmark configuration from the YAML file
2. Find test images in the configured directory
3. Run Optuna optimization to find the best hyperparameters
4. Save results and optionally generate performance plots

See [Benchmark Config Reference](../configs/wildetect/benchmark.md) for all configuration fields.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `wildetect detection detect -c CONFIG` | Run wildlife detection |
| `wildetect detection census -c CONFIG` | Run census campaign |
| `wildetect detection analyze RESULTS` | Analyze detection results |
| `wildetect visualization extract-gps-coordinates -c CONFIG` | Extract GPS coordinates |
| `wildetect services ui` | Launch web UI |
| `wildetect services api` | Launch REST API |
| `wildetect services fiftyone -a launch` | Launch FiftyOne viewer |
| `wildetect utils info` | Show system info |
| `wildetect utils create-config TYPE` | Generate default config |
| `wildetect benchmarking detection -c CONFIG` | Run benchmarks |

---

For configuration file details, see [Configuration Reference](../configs/wildetect/index.md).  
For shell scripts, see [WildDetect Scripts](../scripts/wildetect/index.md).
