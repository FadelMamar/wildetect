# Benchmark Integration Plan

## Overview
Integrate the existing `BenchmarkPipeline` class from `src/wildetect/utils/benchmark.py` into the CLI benchmarking commands in `src/wildetect/cli/commands/benchmarking.py`.

## Current State Analysis
- **BenchmarkPipeline class**: Already implemented with Optuna optimization for hyperparameter tuning
- **CLI command**: Basic structure exists but not implemented
- **Config file**: `config/benchmark.yaml` exists but is empty
- **Dependencies**: Uses Optuna, PyTorch, and existing config classes

## TODO List

### Phase 1: Configuration Setup
- [x] **Create benchmark configuration schema** in `config/benchmark.yaml`
  - Define benchmark-specific parameters (n_trials, timeout, direction)
  - Include image paths or directory for testing
  - Add hyperparameter search ranges
  - Configure output directory for results

- [x] **Extend config models** in `src/wildetect/core/config_models.py`
  - Create `BenchmarkConfigModel` class
  - Add validation for benchmark parameters
  - Ensure compatibility with existing `PredictionConfig` and `LoaderConfig`

### Phase 2: CLI Command Implementation
- [x] **Implement the `detection` command** in `src/wildetect/cli/commands/benchmarking.py`
  - Load configuration from YAML file
  - Parse command-line arguments and overrides
  - Initialize `BenchmarkPipeline` with proper configs
  - Execute benchmarking with progress reporting
  - Save results to specified output directory

- [x] **Add additional CLI options**
  - `--output-dir`: Directory to save benchmark results
  - `--trials`: Number of optimization trials
  - `--timeout`: Maximum time for optimization
  - `--direction`: Optimization direction (minimize/maximize)
  - `--save-results`: Whether to save detailed results

### Phase 3: Integration
- [x] **Update main CLI entry point**
  - Ensure benchmarking commands are properly registered
  - Add help text and examples



### CLI Command Structure
```bash
# Basic usage
wildetect benchmarking detection --config config/benchmark.yaml


# Help
wildetect benchmarking detection --help
```

## Implementation Order
1. **Configuration schema** - Foundation for all other work
2. **Basic CLI command** - Get minimal functionality working
3. **Enhanced features** - Add result saving and visualization
4. **Testing and documentation** - Ensure reliability and usability

## Success Criteria
- [ ] CLI command successfully loads benchmark configuration
- [ ] BenchmarkPipeline executes with proper error handling
- [ ] Results are saved in specified format and location
- [ ] Command provides helpful error messages for configuration issues
- [ ] Integration doesn't break existing functionality
- [ ] Tests pass and provide good coverage
