# WildTrain Integration Tests

This directory contains integration tests for the WildTrain framework, focusing on end-to-end functionality testing rather than unit-level testing.

## Overview

The integration tests are designed to validate the complete functionality of WildTrain's major components by testing real workflows and data processing pipelines. These tests ensure that the framework works correctly when all components are used together.

## Test Structure

```
tests/
├── integration/
│   ├── data/                    # Data processing and filtering tests
│   ├── training/                # Training pipeline tests
│   ├── evaluation/              # Evaluation and inference tests
│   ├── visualization/           # Visualization and analysis tests
│   └── curriculum/              # Curriculum learning tests
├── fixtures/                    # Test data, configs, and helpers
│   ├── conftest.py             # Pytest fixtures and configuration
│   ├── mock_data.py            # Synthetic data generators
│   └── test_configs/           # Minimal test configurations
└── utils/                       # Test utilities and helpers
    ├── test_helpers.py         # Common test utilities
    └── mock_models.py          # Mock model implementations
```

## Testing Principles

### Data Management
- **Mock Data**: Use synthetic datasets for testing to avoid external dependencies
- **Small Real Data**: Use minimal real datasets only when necessary for integration testing
- **Data Isolation**: Each test should have isolated data to prevent interference
- **Cleanup**: Proper cleanup of temporary files and datasets after each test

### Configuration Management
- **Test Configs**: Separate test configurations from production configs
- **Environment Variables**: Use environment variables for paths and settings
- **Mock External Dependencies**: Mock external services (MLflow, MinIO, etc.)

### Performance Considerations
- **Fast Tests**: Use small datasets and limited epochs for quick feedback
- **Debug Mode**: Leverage existing debug flags in trainers
- **Parallel Execution**: Design tests to run in parallel where possible

## TODO List

### Phase 1: Core Data Processing Tests (Priority 1)

#### Data Processing & Filtering
- [ ] **test_crop_dataset_clustering.py**
  - [ ] Test PatchDataset creation with mock detection data
  - [ ] Test ClusteringFilter application with different reduction factors (0.3, 0.5, 0.7)
  - [ ] Test CropClusteringAdapter functionality and adapter pattern
  - [ ] Test DataLoader compatibility with clustered datasets
  - [ ] Test utility methods (get_crops_by_class, get_crops_by_type, get_crop_info)
  - [ ] Test class distribution analysis before/after clustering
  - [ ] Test adapter information retrieval (get_filter_info)

- [ ] **test_crop_dataset_rebalancing.py**
  - [ ] Test ClassificationRebalanceFilter with different methods (mean, min)
  - [ ] Test class distribution before/after rebalancing
  - [ ] Test exclude_extremes functionality
  - [ ] Test DataLoader compatibility with balanced datasets
  - [ ] Test utility methods (get_crops_by_class, get_crops_by_type)
  - [ ] Test random seed reproducibility
  - [ ] Test different filter configurations

- [ ] **test_classification_filtering.py**
  - [ ] Test ClassificationRebalanceFilter with synthetic annotation data
  - [ ] Test different balancing strategies (mean, min)
  - [ ] Test random seed reproducibility
  - [ ] Test class distribution analysis
  - [ ] Test edge cases (empty annotations, single class)

- [ ] **test_dataset_stats.py**
  - [ ] Test compute_dataset_stats function with mock dataset
  - [ ] Test mean and standard deviation calculation
  - [ ] Test with ClassificationDataModule
  - [ ] Test batch processing functionality
  - [ ] Test error handling for invalid datasets

- [ ] **test_dataset_creation.py**
  - [ ] Test DataPipeline with mock source data
  - [ ] Test ROI configuration and generation
  - [ ] Test transformation pipeline (BoundingBoxClippingTransformer)
  - [ ] Test dataset splitting (train/val)
  - [ ] Test different source formats (COCO, YOLO)

### Phase 2: Training Pipeline Tests (Priority 2)

#### Training Components
- [ ] **test_classifier_training.py**
  - [ ] Test ClassifierTrainer with minimal configuration
  - [ ] Test debug mode functionality
  - [ ] Test checkpoint saving/loading
  - [ ] Test training metrics collection
  - [ ] Test different backbone configurations
  - [ ] Test error handling for invalid configs
  - [ ] Test training completion validation

- [ ] **test_yolo_training.py**
  - [ ] Test UltralyticsDetectionTrainer with small dataset
  - [ ] Test YOLO configuration loading
  - [ ] Test training completion and model saving
  - [ ] Test different YOLO model configurations
  - [ ] Test debug mode functionality
  - [ ] Test training metrics and logging

- [ ] **test_mmdet_training.py**
  - [ ] Test MMDetectionTrainer with minimal config
  - [ ] Test MMDetection integration
  - [ ] Test different MMDetection model configurations
  - [ ] Test training completion validation
  - [ ] Test debug mode functionality

- [ ] **test_hpo.py**
  - [ ] Test ClassifierSweeper with minimal sweep configuration
  - [ ] Test Optuna integration
  - [ ] Test hyperparameter search completion
  - [ ] Test different search spaces
  - [ ] Test optimization metrics tracking
  - [ ] Test debug mode functionality

### Phase 3: Evaluation & Inference Tests (Priority 3)

#### Evaluation Components
- [ ] **test_classifier_evaluation.py**
  - [ ] Test ClassificationEvaluator with mock model
  - [ ] Test evaluation metrics calculation (accuracy, precision, recall, F1)
  - [ ] Test results saving functionality (JSON format)
  - [ ] Test debug mode functionality
  - [ ] Test different evaluation configurations
  - [ ] Test error handling for missing models/data

- [ ] **test_yolo_evaluation.py**
  - [ ] Test UltralyticsEvaluator with mock YOLO model
  - [ ] Test detection metrics calculation (mAP, precision, recall)
  - [ ] Test evaluation completion validation
  - [ ] Test debug mode functionality
  - [ ] Test different evaluation configurations

- [ ] **test_model_inference.py**
  - [ ] Test Detector model creation and inference
  - [ ] Test localizer + classifier pipeline
  - [ ] Test prediction format validation
  - [ ] Test batch inference functionality
  - [ ] Test different model configurations

### Phase 4: Visualization & Analysis Tests (Priority 4)

#### Visualization Components
- [ ] **test_classifier_visualization.py**
  - [ ] Test FiftyOne integration for classification
  - [ ] Test prediction visualization
  - [ ] Mock FiftyOne session for testing
  - [ ] Test add_predictions_from_classifier function
  - [ ] Test different visualization configurations
  - [ ] Test error handling for missing datasets

- [ ] **test_detection_visualization.py**
  - [ ] Test Detector model creation and inference
  - [ ] Test FiftyOne integration for detection
  - [ ] Test localizer + classifier pipeline
  - [ ] Test add_predictions_from_detector function
  - [ ] Test different detection configurations
  - [ ] Test visualization with supervision library

- [ ] **test_shap_explanation.py**
  - [ ] Test ClassifierSHAPExplainer with mock model
  - [ ] Test SHAP value calculation
  - [ ] Test explanation visualization
  - [ ] Test background sample generation
  - [ ] Test different explanation configurations
  - [ ] Test error handling for invalid models

### Phase 5: Curriculum Learning Tests (Priority 5)

#### Curriculum Components
- [ ] **test_curriculum_detection.py**
  - [ ] Test CurriculumDetectionDataset loading
  - [ ] Test curriculum progression (difficulty scaling)
  - [ ] Test available_indices updates
  - [ ] Test PatchDataset integration with curriculum
  - [ ] Test different curriculum configurations
  - [ ] Test curriculum state management
  - [ ] Test curriculum callback functionality

### Phase 6: Test Infrastructure Setup

#### Fixtures and Configuration
- [ ] **conftest.py**
  - [ ] Create mock_dataset_path fixture
  - [ ] Create test_config fixture
  - [ ] Create mock_model_checkpoint fixture
  - [ ] Create mock_fiftyone_session fixture
  - [ ] Create temporary directory fixtures
  - [ ] Create mock data generators

- [ ] **mock_data.py**
  - [ ] Generate synthetic COCO annotations
  - [ ] Generate synthetic YOLO dataset
  - [ ] Create mock model checkpoints
  - [ ] Generate test images
  - [ ] Create mock classification datasets
  - [ ] Create mock detection datasets

- [ ] **test_configs/**
  - [ ] Create test_classification_config.yaml
  - [ ] Create test_yolo_config.yaml
  - [ ] Create test_mmdet_config.yaml
  - [ ] Create test_evaluation_config.yaml
  - [ ] Create test_visualization_config.yaml
  - [ ] Create test_curriculum_config.yaml

- [ ] **test_helpers.py**
  - [ ] Create data validation helpers
  - [ ] Create model loading helpers
  - [ ] Create metric calculation helpers
  - [ ] Create cleanup utilities
  - [ ] Create test data generators

- [ ] **mock_models.py**
  - [ ] Create mock GenericClassifier
  - [ ] Create mock UltralyticsLocalizer
  - [ ] Create mock Detector
  - [ ] Create mock training checkpoints
  - [ ] Create mock evaluation results

## Running Tests

### Prerequisites
```bash
# Install test dependencies
uv add --dev pytest pytest-cov pytest-mock pytest-xdist

# Set up test environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Test Execution
```bash
# Run all integration tests
uv run python -m pytest tests/integration/ -v

# Run specific test category
uv run python -m pytest tests/integration/data/ -v

# Run with coverage
uv run python -m pytest tests/integration/ --cov=wildtrain --cov-report=html

# Run slow tests separately
uv run python -m pytest tests/integration/ -m slow -v

# Run tests in parallel
uv run python -m pytest tests/integration/ -n auto
```

### Test Markers
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.data`: Data processing tests
- `@pytest.mark.training`: Training pipeline tests
- `@pytest.mark.evaluation`: Evaluation tests
- `@pytest.mark.visualization`: Visualization tests
- `@pytest.mark.curriculum`: Curriculum learning tests

## Test Data Requirements

### Mock Data Structure
- **Classification**: Synthetic image dataset with class labels
- **Detection**: Synthetic COCO/YOLO format annotations
- **Models**: Mock checkpoints with known architectures
- **Configurations**: Minimal valid configurations for each component

### Environment Variables
```bash
# Test data paths
export WILDTRAIN_TEST_DATA_DIR="/path/to/test/data"
export WILDTRAIN_TEST_MODELS_DIR="/path/to/test/models"
export WILDTRAIN_TEST_CONFIGS_DIR="/path/to/test/configs"

# Test mode
export WILDTRAIN_TEST_MODE="true"
export WILDTRAIN_DEBUG_MODE="true"
```

## Best Practices

### Test Design
- Each test should be independent and isolated
- Use descriptive test names that explain the scenario
- Test both success and failure scenarios
- Include edge cases and error conditions
- Use appropriate assertions and validations

### Performance
- Keep tests fast (under 30 seconds each)
- Use debug mode where available
- Mock expensive operations
- Use small datasets for testing
- Implement timeouts for long-running operations

### Maintenance
- Update tests when APIs change
- Keep mock data up to date
- Document test data requirements
- Maintain test configuration files
- Regular cleanup of test artifacts

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Add appropriate test markers
3. Include comprehensive docstrings
4. Add to the relevant TODO section above
5. Update this README if needed
6. Ensure tests pass in CI/CD pipeline

## Notes

- Unit tests are intentionally skipped as they are too low-level
- Focus is on integration testing of complete workflows
- Tests should validate real functionality, not just API contracts
- Mock external dependencies to ensure test reliability
- Use debug mode extensively to speed up test execution 