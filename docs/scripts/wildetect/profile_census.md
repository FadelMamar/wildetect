# Census Profiling Script

> **Location**: `scripts/profile_census.bat`

**Purpose**: Run census campaign with detailed performance profiling enabled. This script executes the census command with profiling flags to analyze performance bottlenecks, memory usage, and execution time.

## Usage

```batch
scripts\profile_census.bat
```

The script automatically:
1. Changes to the project root directory
2. Runs census with profiling enabled

## Command Executed

```batch
uv run wildetect detection census -c config/census.yaml --profile --gpu-profile --line-profile
```

## Profiling Flags

The script enables three types of profiling:

- **`--profile`**: General profiling (timing, function calls)
- **`--gpu-profile`**: GPU memory and utilization profiling
- **`--line-profile`**: Line-by-line execution time profiling

## Configuration

**Config File**: `config/census.yaml`

See [Census Configuration](../../configs/wildetect/census.md) for complete parameter reference.

## Prerequisites

1. **Census Config**: `config/census.yaml` must be properly configured
2. **Environment**: `.env` file should exist
3. **Model Available**: Model should be accessible
4. **Profiling Tools**: Profiling dependencies should be installed

## Example Workflow

### 1. Configure Census

Edit `config/census.yaml` (see [Census Config](../../configs/wildetect/census.md))

### 2. Run Profiling

```batch
scripts\profile_census.bat
```

### 3. Analyze Results

Profiling results will be saved to:
- Profile reports in output directory
- Performance metrics and statistics
- Memory usage analysis
- GPU utilization data
- Line-by-line timing information

## Output

The profiling script generates:

- **Profile Reports**: Detailed timing and performance reports
- **Memory Analysis**: Memory usage statistics and peak usage
- **GPU Profiling**: GPU memory and utilization metrics
- **Line Profiles**: Line-by-line execution time breakdown
- **Bottleneck Identification**: Identification of slow functions and operations
- **Performance Metrics**: Throughput, latency, and efficiency metrics

## Use Cases

### Performance Optimization

Use profiling to identify bottlenecks:
- Slow functions or operations
- Memory-intensive operations
- GPU utilization issues
- Inefficient data loading

### Resource Planning

Analyze resource requirements:
- Memory usage patterns
- GPU memory requirements
- CPU utilization
- Disk I/O patterns

### Comparison

Compare different configurations:
- Different batch sizes
- Different pipeline types
- Different hardware setups

## Interpreting Results

### General Profile

- Function call counts
- Total time per function
- Cumulative time
- Call graph

### GPU Profile

- GPU memory allocation
- Memory peak usage
- GPU utilization percentage
- Memory leaks

### Line Profile

- Time spent per line of code
- Hot spots in code
- Inefficient operations

## Troubleshooting

### Profiling Overhead

**Issue**: Profiling significantly slows down execution

**Solutions**:
1. This is expected - profiling adds overhead
2. Use smaller test dataset for profiling
3. Disable line profiling for faster runs
4. Profile only specific sections if needed

### Memory Profiling Fails

**Issue**: Memory profiling errors

**Solutions**:
1. Verify memory profiling tools are installed
2. Check Python version compatibility
3. Ensure sufficient system memory
4. Try disabling other profiling types

### GPU Profiling Not Available

**Issue**: GPU profiling doesn't work

**Solutions**:
1. Verify GPU is available and accessible
2. Check CUDA is properly installed
3. Ensure PyTorch has GPU support
4. Verify GPU profiling tools are installed

### Results Not Saved

**Issue**: Profiling completes but no reports

**Solutions**:
1. Check output directory exists
2. Verify file permissions
3. Review logs for error messages
4. Ensure sufficient disk space

### Too Much Data

**Issue**: Profiling generates too much data

**Solutions**:
1. Use smaller test dataset
2. Disable line profiling (most verbose)
3. Profile only specific time periods
4. Filter results to relevant sections

## Best Practices

1. **Test Dataset**: Use representative but smaller dataset for profiling
2. **Selective Profiling**: Enable only needed profiling types
3. **Baseline First**: Run without profiling first to establish baseline
4. **Compare Runs**: Compare profiling results across different configurations
5. **Document Findings**: Document performance bottlenecks and optimizations

## Related Documentation

- [Census Configuration](../../configs/wildetect/census.md)
- [Census Script](run_census.md)
- [Benchmark Config](../../configs/wildetect/benchmark.md)
- [Performance Optimization](../../tutorials/end-to-end-detection.md)

