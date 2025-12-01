# Benchmark Configuration

> **Location**: `config/benchmark.yaml`

**Purpose**: Configuration file for performance benchmarking and hyperparameter optimization of the detection pipeline. Uses Optuna for automated hyperparameter search to find optimal settings for batch size, tile size, worker counts, and other processing parameters.

## Configuration Structure

### Complete Parameter Reference

```yaml
# Benchmark Configuration for WildDetect
# This file configures the benchmarking of the detection pipeline

# Core benchmark execution settings
execution:
  n_trials: 30                    # Number of optimization trials
  timeout: 3600                   # Maximum time for optimization in seconds
  direction: "minimize"           # Optimization direction: "minimize" or "maximize"
  sampler: "TPE"                  # Optuna sampler: "TPE", "Random", or "Grid"
  seed: 42                        # Random seed for reproducibility

# Test data configuration
test_images:
  path: "test_images"             # Path to test images directory
  recursive: true                 # Search recursively for images
  max_images: 100                # Maximum number of images to use
  supported_formats:              # Supported image formats
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".tiff"
    - ".tif"
    - ".bmp"

# Hyperparameter search space
hyperparameters:
  batch_size:                     # Batch sizes to test
    - 8
    - 16
    - 32
    - 64
    - 128
    - 256
    - 512
  num_workers:                    # Number of workers to test
    - 0
    - 2
    - 4
    - 8
    - 16
  tile_size:                      # Tile sizes to test
    - 400
    - 800
    - 1200
    - 1600
  overlap_ratio:                  # Overlap ratios to test
    - 0.1
    - 0.2
    - 0.3

# Output configuration
output:
  directory: "results/benchmarks" # Output directory for results
  save_plots: true               # Save performance plots
  save_results: true             # Save detailed results
  format: "json"                 # Output format: "json", "csv", or "both"
  include_optimization_history: true  # Include optimization history
  auto_open: false               # Auto-open results after completion

# Model configuration (inherits from existing models)
model:
  mlflow_model_name: null        # Will use environment variable if null
  mlflow_model_alias: null       # Will use environment variable if null
  device: "auto"                 # Device to run inference on

# Processing configuration
processing:
  tile_size: 800                 # Default tile size for processing
  overlap_ratio: 0.2             # Default overlap ratio
  pipeline_type: "single"        # Pipeline type: "single", "multi", or "async"
  queue_size: 64                 # Queue size for multi-threaded pipeline
  batch_size: 32                 # Default batch size for inference
  num_workers: 0                 # Default number of workers
  max_concurrent: 4              # Maximum concurrent inference tasks

# Flight specifications
flight_specs:
  sensor_height: 24.0            # Sensor height in mm
  focal_length: 35.0             # Focal length in mm
  flight_height: 180.0           # Flight height in meters

# Inference service configuration
inference_service:
  url: null                      # Inference service URL (if using external service)
  timeout: 60                    # Timeout for inference in seconds

# Logging configuration
logging:
  verbose: false                 # Verbose logging
  log_file: null                 # Log file path

# Profiling configuration
profiling:
  enable: false                  # Enable profiling
  memory_profile: false          # Enable memory profiling
  line_profile: false            # Enable line-by-line profiling
  gpu_profile: false             # Enable GPU profiling
```

### Parameter Descriptions

#### `execution`
Core benchmark execution settings using Optuna optimization framework.

- **`n_trials`** (int): Number of optimization trials to run (more trials = better results, but slower)
- **`timeout`** (int): Maximum time in seconds for the entire optimization process
- **`direction`** (string): Optimization direction. Options: `"minimize"` (for latency), `"maximize"` (for throughput)
- **`sampler`** (string): Optuna sampler algorithm. Options: `"TPE"` (Tree-structured Parzen Estimator), `"Random"`, `"Grid"`
- **`seed`** (int): Random seed for reproducibility

#### `test_images`
Test data configuration for benchmarking.

- **`path`** (string): Path to directory containing test images
- **`recursive`** (bool): Whether to search subdirectories recursively
- **`max_images`** (int): Maximum number of images to use for benchmarking
- **`supported_formats`** (list): List of supported image file extensions

#### `hyperparameters`
Hyperparameter search space - defines which values to test for each parameter.

- **`batch_size`** (list): List of batch sizes to test (typically powers of 2: 8, 16, 32, 64, ...)
- **`num_workers`** (list): List of worker counts to test
- **`tile_size`** (list): List of tile sizes to test (in pixels)
- **`overlap_ratio`** (list): List of overlap ratios to test (0.0-1.0)

#### `output`
Output configuration for benchmark results.

- **`directory`** (string): Output directory for benchmark results
- **`save_plots`** (bool): Whether to save performance visualization plots
- **`save_results`** (bool): Whether to save detailed results
- **`format`** (string): Output format. Options: `"json"`, `"csv"`, `"both"`
- **`include_optimization_history`** (bool): Whether to include full optimization history
- **`auto_open`** (bool): Whether to automatically open results after completion

#### `model`
Model configuration (can use environment variables).

- **`mlflow_model_name`** (string, optional): Model name. If `null`, uses environment variable
- **`mlflow_model_alias`** (string, optional): Model alias. If `null`, uses environment variable
- **`device`** (string): Device for inference. Options: `"auto"`, `"cpu"`, `"cuda"`

#### `processing`
Default processing configuration (used as baseline).

- **`tile_size`** (int): Default tile size
- **`overlap_ratio`** (float): Default overlap ratio
- **`pipeline_type`** (string): Pipeline type. Options: `"single"`, `"multi"`, `"async"`
- **`queue_size`** (int): Queue size for multi-threaded pipeline
- **`batch_size`** (int): Default batch size
- **`num_workers`** (int): Default number of workers
- **`max_concurrent`** (int): Maximum concurrent inference tasks

#### `flight_specs`
Flight specifications (for geographic calculations if needed).

- **`sensor_height`** (float): Camera sensor height in millimeters
- **`focal_length`** (float): Lens focal length in millimeters
- **`flight_height`** (float): Flight altitude in meters

#### `inference_service`, `logging`, `profiling`
Same as detection configuration. See [Detection Config](detection.md) for details.

---

## Example Configurations

### Quick Benchmark (Few Trials)

```yaml
execution:
  n_trials: 10
  timeout: 1800
  direction: "minimize"
  sampler: "TPE"
  seed: 42

test_images:
  path: "test_images"
  recursive: true
  max_images: 20

hyperparameters:
  batch_size: [8, 16, 32, 64]
  num_workers: [0, 2, 4]
  tile_size: [400, 800, 1200]

output:
  directory: "results/quick_benchmark"
  save_plots: true
  format: "json"
```

### Comprehensive Benchmark

```yaml
execution:
  n_trials: 50
  timeout: 7200  # 2 hours
  direction: "minimize"
  sampler: "TPE"
  seed: 42

test_images:
  path: "D:/benchmark_images/"
  recursive: true
  max_images: 200

hyperparameters:
  batch_size: [4, 8, 16, 32, 64, 128]
  num_workers: [0, 2, 4, 8, 16]
  tile_size: [400, 600, 800, 1000, 1200]
  overlap_ratio: [0.1, 0.2, 0.3, 0.4]

output:
  directory: "results/comprehensive_benchmark"
  save_plots: true
  save_results: true
  format: "both"
  include_optimization_history: true
```

### GPU-Specific Benchmark

```yaml
execution:
  n_trials: 30
  direction: "maximize"  # Maximize throughput
  sampler: "TPE"

test_images:
  path: "test_images"
  max_images: 100

hyperparameters:
  batch_size: [16, 32, 64, 128, 256]  # Larger batches for GPU
  num_workers: [2, 4, 8]
  tile_size: [800, 1200, 1600]

model:
  device: "cuda"

processing:
  pipeline_type: "multi"
  pin_memory: true
```

### CPU-Only Benchmark

```yaml
execution:
  n_trials: 20
  direction: "minimize"
  sampler: "Random"

test_images:
  path: "test_images"
  max_images: 50

hyperparameters:
  batch_size: [1, 2, 4, 8]  # Smaller batches for CPU
  num_workers: [0, 1, 2, 4]
  tile_size: [400, 600, 800]

model:
  device: "cpu"

processing:
  pipeline_type: "single"
```

---

## Best Practices

1. **Number of Trials**: 
   - Start with 10-20 trials for quick results
   - Use 30-50 trials for comprehensive optimization
   - More trials = better results but slower

2. **Test Images**:
   - Use representative images from your actual use case
   - Include variety in image sizes and content
   - Don't use too many images (50-100 is usually sufficient)

3. **Hyperparameter Ranges**:
   - Start with wide ranges, then narrow based on results
   - Consider hardware constraints (GPU memory, CPU cores)
   - Test powers of 2 for batch sizes (8, 16, 32, 64)

4. **Optimization Direction**:
   - Use `"minimize"` to optimize for latency (faster processing)
   - Use `"maximize"` to optimize for throughput (more images/second)

5. **Sampler Selection**:
   - Use `"TPE"` for most cases (efficient exploration)
   - Use `"Random"` for quick baseline
   - Use `"Grid"` for exhaustive search (only with few parameters)

6. **Output Analysis**:
   - Enable `save_plots` to visualize performance
   - Use `format: "both"` to get JSON and CSV outputs
   - Review optimization history to understand parameter relationships

7. **Reproducibility**:
   - Set `seed` for reproducible results
   - Save results for comparison across runs
   - Document optimal parameters found

---

## Troubleshooting

### Benchmark Takes Too Long

**Issue**: Optimization runs for hours without completing

**Solutions**:
1. Reduce `n_trials` (try 10-20 instead of 50+)
2. Set `timeout` to limit maximum time
3. Reduce `max_images` in test data
4. Narrow hyperparameter search space
5. Use `sampler: "Random"` for faster sampling

### Out of Memory During Benchmark

**Issue**: Memory errors when testing large batch sizes

**Solutions**:
1. Remove large batch sizes from `hyperparameters.batch_size`
2. Reduce `max_images` in test data
3. Test smaller tile sizes first
4. Close other applications
5. Use CPU device if GPU memory is limited

### No Improvement Found

**Issue**: Benchmark doesn't find better parameters

**Solutions**:
1. Increase `n_trials` for more exploration
2. Widen hyperparameter ranges
3. Check if baseline configuration is already optimal
4. Verify test images are representative
5. Try different sampler (`"TPE"` vs `"Random"`)

### Results Not Saved

**Issue**: Benchmark completes but no output files

**Solutions**:
1. Verify `output.directory` exists or can be created
2. Check `save_results: true` is enabled
3. Check file permissions for output directory
4. Review logs for error messages
5. Ensure sufficient disk space

### Inconsistent Results

**Issue**: Results vary between benchmark runs

**Solutions**:
1. Set `seed` for reproducibility
2. Use same test images across runs
3. Ensure consistent hardware state (no other processes)
4. Use fixed model version (not "latest" alias)
5. Run multiple trials and average results

---

## Related Documentation

- [Configuration Overview](index.md)
- [Detection Config](detection.md)
- [Profiling Script](../../scripts/wildetect/profile_census.md)
- [Performance Optimization](../../tutorials/end-to-end-detection.md)

