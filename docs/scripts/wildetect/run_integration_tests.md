# Integration Tests Script

> **Location**: `scripts/run_integration_tests.bat`

**Purpose**: Run integration tests for the detection pipeline to verify end-to-end functionality with real images and data.

## Usage

```batch
scripts\run_integration_tests.bat
```

The script automatically:
1. Changes to the project root directory
2. Runs multiple integration test suites

## Commands Executed

The script runs two test suites:

```batch
# Detection pipeline tests
uv run pytest tests/test_detection_pipeline.py::TestDetectionPipeline::test_detection_pipeline_with_real_images -v

# Data loading tests
uv run pytest tests/test_data_loading.py -v
```

## Test Coverage

### Detection Pipeline Tests

Tests the complete detection pipeline:
- Detection with real images
- Model loading and inference
- Result formatting
- Output generation

### Data Loading Tests

Tests data loading and preprocessing:
- Image loading
- Data preprocessing
- Batch creation
- Data transformations

## Prerequisites

1. **Test Data**: Test images should be available in `tests/data/` (if required)
2. **Model Available**: Test model should be accessible (MLflow or local)
3. **Dependencies**: All test dependencies installed via `uv sync`
4. **pytest**: pytest should be installed and available

## Example Workflow

### 1. Prepare Test Data

Ensure test data is available:
- Test images in appropriate location
- Test model accessible
- Test configurations ready

### 2. Run Tests

```batch
scripts\run_integration_tests.bat
```

### 3. Review Results

- Check test output for pass/fail status
- Review any error messages
- Verify all tests complete successfully

## Output

The test script provides:

- **Test Results**: Pass/fail status for each test
- **Verbose Output**: Detailed test execution information (`-v` flag)
- **Error Messages**: Detailed error information for failed tests
- **Execution Time**: Time taken for each test

## Running Individual Tests

You can also run tests individually:

```batch
# Run all detection pipeline tests
uv run pytest tests/test_detection_pipeline.py -v

# Run all data loading tests
uv run pytest tests/test_data_loading.py -v

# Run specific test
uv run pytest tests/test_detection_pipeline.py::TestDetectionPipeline::test_detection_pipeline_with_real_images -v
```

## Troubleshooting

### Tests Fail

**Issue**: Tests fail with errors

**Solutions**:
1. Check test data is available and accessible
2. Verify model is available (MLflow or local path)
3. Check test configuration is correct
4. Review error messages for specific issues
5. Ensure all dependencies are installed

### Test Data Not Found

**Issue**: Cannot find test data files

**Solutions**:
1. Verify test data exists in expected location
2. Check file paths in test files
3. Ensure test data is included in repository
4. Check file permissions

### Model Loading Fails in Tests

**Issue**: Tests fail when loading model

**Solutions**:
1. Verify test model is available
2. Check MLflow server is running (if using MLflow)
3. Verify model path/name in test configuration
4. Ensure model file format is correct

### Tests Take Too Long

**Issue**: Integration tests run very slowly

**Solutions**:
1. Use smaller test images
2. Reduce number of test images
3. Use CPU instead of GPU for faster startup
4. Skip slow tests if not needed: `-k "not slow"`

### Import Errors

**Issue**: Import errors when running tests

**Solutions**:
1. Verify Python environment is correct
2. Run `uv sync` to install dependencies
3. Check PYTHONPATH includes project directories
4. Verify test files are in correct location

## Best Practices

1. **Run Before Commits**: Run integration tests before committing changes
2. **Fix Failing Tests**: Don't commit with failing tests
3. **Add New Tests**: Add tests for new functionality
4. **Keep Tests Updated**: Update tests when functionality changes
5. **Use Real Data**: Use representative real data in tests

## Related Documentation

- [Testing Documentation](../../troubleshooting.md)
- [Detection Pipeline](../../tutorials/end-to-end-detection.md)
- [pytest Documentation](https://docs.pytest.org/)

