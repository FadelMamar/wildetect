# Wildetect Test Suite

This directory contains comprehensive tests for the wildetect project, covering all major functionality including data loading, tile processing, drone image management, and campaign analysis.

## Test Structure

### Core Test Files

- **`test_basic_functionality.py`** - Basic functionality tests for imports and core operations
- **`test_data_loading.py`** - Tests for data loading, image discovery, and tile creation
- **`test_census_data.py`** - Tests for CensusData campaign management functionality
- **`test_tile_and_drone_image.py`** - Tests for Tile and DroneImage classes
- **`test_utils.py`** - Tests for TileUtils and utility functions
- **`run_all_tests.py`** - Comprehensive test runner for executing all tests

## Running Tests

### Run All Tests
```bash
# From the project root
python test/run_all_tests.py

# With detailed output
python test/run_all_tests.py --detailed

# Save results to file
python test/run_all_tests.py --output results.json
```

### Run Specific Tests
```bash
# Run a specific test
python test/run_all_tests.py --test data_loading
python test/run_all_tests.py --test census_data
python test/run_all_tests.py --test tile_and_drone_image
python test/run_all_tests.py --test utils
python test/run_all_tests.py --test basic_functionality
```

### Run Individual Test Files
```bash
# Run individual test files directly
python test/test_basic_functionality.py
python test/test_data_loading.py
python test/test_census_data.py
python test/test_tile_and_drone_image.py
python test/test_utils.py
```

## Test Coverage

### Basic Functionality Tests (`test_basic_functionality.py`)
- **Import Tests**: Verify all core modules can be imported
- **Config Tests**: Test LoaderConfig and FlightSpecs instantiation
- **Basic Operations**: Test Tile, Detection, and DroneImage creation
- **Utility Functions**: Test get_images_paths and TileUtils validation
- **CensusData Basic**: Test basic CensusData functionality

### Data Loading Tests (`test_data_loading.py`)
- **Image Discovery**: Test get_images_paths function
- **Loader Config**: Test LoaderConfig creation and validation
- **ImageTileDataset**: Test dataset creation and iteration
- **DataLoader**: Test batch loading and iteration
- **Tile Loading**: Test load_images_as_tiles function
- **DroneImage Loading**: Test load_images_as_drone_images function
- **Loader Creation**: Test create_loader function

### CensusData Tests (`test_census_data.py`)
- **Initialization**: Test CensusData creation with various configs
- **Image Management**: Test adding images from paths and directories
- **DroneImage Creation**: Test creating DroneImage instances
- **Campaign Statistics**: Test statistics generation
- **Data Access**: Test getting drone images and detections
- **Export Functionality**: Test detection report export
- **Metadata Handling**: Test campaign metadata functionality

### Tile and DroneImage Tests (`test_tile_and_drone_image.py`)
- **Tile Creation**: Test Tile instantiation and properties
- **Detection Handling**: Test setting and managing detections
- **Offset Management**: Test coordinate offset functionality
- **Filtering**: Test detection filtering and NMS
- **DroneImage Creation**: Test DroneImage instantiation
- **Tile Management**: Test adding and managing tiles
- **Detection Aggregation**: Test merging detections across tiles
- **Statistics**: Test statistics generation
- **Visualization**: Test drawing detections (if OpenCV available)

### Utils Tests (`test_utils.py`)
- **Image Discovery**: Test get_images_paths function
- **Validation**: Test TileUtils parameter validation
- **Patch Extraction**: Test patch extraction functionality
- **Patch Count**: Test patch count calculations
- **Edge Cases**: Test validation edge cases
- **Error Handling**: Test error handling for invalid inputs

## Test Features

### Temporary File Management
All tests use `tempfile.TemporaryDirectory()` to create isolated test environments that are automatically cleaned up.

### Image Generation
Tests create synthetic test images using PIL and numpy for consistent testing without requiring external image files.

### Comprehensive Logging
All tests include detailed logging to help debug issues and understand test execution flow.

### Error Handling
Tests include proper error handling and validation to ensure robust functionality.

### Performance Testing
Some tests include timing measurements to ensure performance requirements are met.

## Test Dependencies

### Required Dependencies
- `PIL` (Pillow) - For image creation and manipulation
- `numpy` - For array operations
- `torch` - For tensor operations (optional)
- `torchvision` - For transforms (optional)

### Optional Dependencies
- `opencv-python` - For visualization tests
- `fiftyone` - For FiftyOne integration tests

## Test Output

### Console Output
Tests provide detailed console output including:
- Test progress and status
- Timing information
- Error details for failed tests
- Summary statistics

### JSON Results
When using `--output`, results are saved in JSON format with:
- Test summary statistics
- Individual test results
- Error messages and stack traces
- Timing information

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Exit codes indicate success/failure
- JSON output can be parsed by CI systems
- Tests are isolated and don't require external dependencies
- Timeout handling prevents hanging tests

## Adding New Tests

### Test File Structure
```python
"""
Test description for the module being tested.
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from src.wildetect.core.data import ...

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_function_name():
    """Test description."""
    logger.info("=" * 60)
    logger.info("TESTING FUNCTION NAME")
    logger.info("=" * 60)
    
    # Test implementation
    assert condition, "Assertion message"
    logger.info("✓ Test passed")

def run_all_tests():
    """Run all tests in this file."""
    logger.info("Starting tests...")
    
    try:
        test_function_name()
        # Add more tests...
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_all_tests()
```

### Best Practices
1. Use descriptive test function names
2. Include comprehensive logging
3. Use temporary directories for file operations
4. Test both success and failure cases
5. Include edge case testing
6. Add proper error handling
7. Use assertions with descriptive messages

## Troubleshooting

### Common Issues

**Import Errors**: Ensure the project root is in the Python path
```python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

**Missing Dependencies**: Install required packages
```bash
pip install pillow numpy torch torchvision opencv-python
```

**Timeout Issues**: Increase timeout in test runner if needed
```python
timeout=600  # 10 minutes instead of 5
```

**Memory Issues**: Reduce test data size for large tests
```python
# Use smaller images for testing
img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
```

### Debug Mode
Run tests with detailed output:
```bash
python test/run_all_tests.py --detailed
```

### Individual Test Debugging
Run individual tests with full output:
```bash
python test/test_basic_functionality.py
``` 