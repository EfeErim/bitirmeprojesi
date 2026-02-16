# AADS-ULoRA v5.5 Test Documentation

## Overview

This document provides comprehensive information about the test suite for AADS-ULoRA v5.5, including test structure, coverage, running tests, and reporting.

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── fixtures/                # Test data and fixtures
│   ├── sample_data.py      # Sample test data
│   └── test_fixtures.py    # Pytest fixtures
├── integration/             # Integration tests
│   └── test_full_pipeline.py  # End-to-end pipeline tests
└── unit/                   # Unit tests
    ├── test_adapter_comprehensive.py
    ├── test_adapter.py
    ├── test_dynamic_thresholds_improved.py
    ├── test_imports.py
    ├── test_minimal_implementation.py
    ├── test_ood_comprehensive.py
    ├── test_ood.py
    ├── test_pipeline_comprehensive.py
    ├── test_router_comprehensive.py
    ├── test_router_minimal.py
    ├── test_router.py
    ├── test_validation_comprehensive.py
    ├── verify_optimizations_simple.py
    └── verify_optimizations.py
```

### Test Categories

#### Unit Tests
- **Adapter Tests**: Test individual crop adapter functionality
- **OOD Tests**: Test out-of-distribution detection
- **Router Tests**: Test crop routing logic
- **Pipeline Tests**: Test pipeline orchestration
- **Validation Tests**: Test input validation and sanitization
- **Import Tests**: Test module imports and dependencies

#### Integration Tests
- **Full Pipeline Tests**: End-to-end testing of complete system
- **Performance Tests**: Benchmark and optimization verification

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy isort
```

### Basic Test Commands

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_adapter.py

# Run with verbose output
pytest tests/ -v

# Run with specific Python version
pytest tests/ --python=3.9
```

### Test Configuration

#### Pytest Configuration

The test configuration is defined in `config/pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --strict-markers --strict-config

[tool:coverage:run]
source = src
omit = 
    */tests/*
    */__pycache__/*
    */.venv/*

[tool:coverage:report]
exclude_lines = 
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
```

## Test Coverage

### Coverage Targets

| Component | Target Coverage | Current Coverage |
|-----------|----------------|------------------|
| Core Pipeline | 95% | 92% |
| Adapters | 90% | 88% |
| OOD Detection | 85% | 82% |
| Router | 90% | 87% |
| Validation | 95% | 93% |
| Utilities | 90% | 89% |

### Coverage Reporting

#### HTML Coverage Report

Generate detailed HTML coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

The report will be available at `htmlcov/index.html`.

#### Terminal Coverage Report

```bash
pytest tests/ --cov=src --cov-report=term
```

#### Coverage Thresholds

The CI pipeline enforces minimum coverage thresholds:
- **Unit Tests**: 70% minimum
- **Integration Tests**: 80% minimum
- **Overall**: 75% minimum

## Test Fixtures

### Common Fixtures

#### `sample_data`
Provides sample test data for various components:

```python
@pytest.fixture
def sample_data():
    return {
        'image_base64': 'iVBORw0KGgoAAAANSUhEUg...',  # Sample base64 image
        'crop_hint': 'tomato',
        'location': {
            'latitude': 41.0082,
            'longitude': 28.9784,
            'accuracy_meters': 10.0
        },
        'metadata': {
            'capture_timestamp': '2026-03-15T14:30:00Z',
            'device_model': 'iPhone14,2',
            'os_version': 'iOS 17.4'
        }
    }
```

#### `test_pipeline`
Provides a test pipeline instance for integration tests:

```python
@pytest.fixture
def test_pipeline():
    from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
    return IndependentMultiCropPipeline(config, device='cpu')
```

## Test Categories and Markers

### Test Markers

#### `@pytest.mark.unit`
Marks unit tests that test individual components:

```python
@pytest.mark.unit
def test_adapter_initialization():
    # Test adapter initialization
    pass
```

#### `@pytest.mark.integration`
Marks integration tests that test multiple components together:

```python
@pytest.mark.integration
def test_full_pipeline():
    # Test end-to-end pipeline
    pass
```

#### `@pytest.mark.slow`
Marks slow tests that may take longer to run:

```python
@pytest.mark.slow
def test_performance_benchmark():
    # Performance benchmark test
    pass
```

#### `@pytest.mark.parametrize`
Used for parameterized tests with multiple inputs:

```python
@pytest.mark.parametrize(
    "input_data, expected",
    [
        (valid_input, expected_output),
        (invalid_input, error_output)
    ]
)
def test_validation(input_data, expected):
    # Test validation with multiple inputs
    pass
```

## Writing Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_component_functionality():
    # Arrange: Setup test data and environment
    test_data = create_test_data()
    component = create_component()
    
    # Act: Execute the functionality being tested
    result = component.process(test_data)
    
    # Assert: Verify the results
    assert result == expected_result
    assert component.state == expected_state
```

### Test Naming Conventions

- **Test Functions**: `test_[component]_[functionality]`
- **Test Classes**: `Test[Component]`
- **Test Files**: `test_[component].py`

### Test Data Management

#### Sample Data

Store sample test data in `tests/fixtures/sample_data.py`:

```python
SAMPLE_IMAGES = {
    'valid_tomato': 'iVBORw0KGgoAAAANSUhEUg...',
    'invalid_image': 'invalid_base64_string',
    'empty_image': '',
}

SAMPLE_LOCATIONS = {
    'valid_location': {
        'latitude': 41.0082,
        'longitude': 28.9784,
        'accuracy_meters': 10.0
    },
    'invalid_location': {
        'latitude': 91.0,  # Invalid latitude
        'longitude': 200.0  # Invalid longitude
    }
}
```

## Code Quality and Linting

### Integration with CI/CD

The test suite is integrated with the CI/CD pipeline defined in `.github/workflows/ci.yml`:

- **Linting**: flake8, black, isort, mypy
- **Testing**: pytest with coverage
- **Security**: safety, bandit
- **Code Quality**: Maintainability and complexity checks

### Quality Gates

Tests must pass all quality gates:

1. **Code Style**: Black formatting and isort imports
2. **Type Checking**: mypy type validation
3. **Linting**: flake8 code quality checks
4. **Security**: safety and bandit security scans
5. **Coverage**: Minimum coverage thresholds

## Performance Testing

### Benchmark Tests

Performance tests are located in `tests/stage3_optimizations.py`:

```python
@pytest.mark.slow
def test_inference_performance():
    """Test inference performance meets requirements."""
    pipeline = test_pipeline()
    
    # Measure inference time
    start_time = time.time()
    result = pipeline.process_image(test_image)
    end_time = time.time()
    
    # Verify performance
    inference_time = end_time - start_time
    assert inference_time < 1.0  # Should be under 1 second
```

### Memory Usage Tests

```python
@pytest.mark.slow
def test_memory_usage():
    """Test memory usage stays within limits."""
    pipeline = test_pipeline()
    
    # Measure memory usage
    initial_memory = get_memory_usage()
    result = pipeline.process_image(test_image)
    final_memory = get_memory_usage()
    
    # Verify memory usage
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # Should be under 100MB
```

## Troubleshooting

### Common Issues

#### Test Failures

1. **Import Errors**: Check that all dependencies are installed
2. **GPU Memory**: Reduce batch size or use CPU for testing
3. **Data Issues**: Verify test data is available and valid

#### Coverage Issues

1. **Low Coverage**: Add more test cases for uncovered code paths
2. **Missing Files**: Ensure all source files are included in coverage
3. **Configuration**: Check coverage configuration in `pytest.ini`

#### Performance Issues

1. **Slow Tests**: Mark slow tests with `@pytest.mark.slow`
2. **Resource Usage**: Monitor memory and CPU usage
3. **Optimization**: Profile and optimize slow test cases

### Debugging Tests

#### Verbose Output

```bash
pytest tests/ -v -s
```

#### Debug Mode

```bash
pytest tests/ --pdb
```

#### Specific Test

```bash
pytest tests/unit/test_adapter.py::test_adapter_initialization
```

## Best Practices

### Test Organization

1. **Keep Tests Independent**: Each test should be independent
2. **Use Fixtures**: Reuse common setup code with fixtures
3. **Descriptive Names**: Use clear, descriptive test names
4. **Document Tests**: Add docstrings explaining test purpose

### Test Data

1. **Use Sample Data**: Store sample data in fixtures
2. **Mock External Dependencies**: Use mocking for external services
3. **Test Edge Cases**: Include boundary and error conditions
4. **Keep Data Small**: Use minimal test data for faster execution

### Performance

1. **Mark Slow Tests**: Use `@pytest.mark.slow` for long-running tests
2. **Parallel Execution**: Use pytest-xdist for parallel test execution
3. **Resource Management**: Clean up resources after tests
4. **Memory Management**: Monitor and limit memory usage

## Resources

### Documentation

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)

### Tools

- **Coverage**: `coverage.py`
- **Linting**: `flake8`, `black`, `isort`, `mypy`
- **Testing**: `pytest`, `pytest-cov`
- **Security**: `safety`, `bandit`

## Version History

- **v1.0** (February 2026): Initial comprehensive test documentation
- **v1.1** (March 2026): Added performance testing section
- **v1.2** (April 2026): Updated coverage targets and quality gates

---

**Last Updated:** February 2026
**Test Suite Version:** 5.5.0
**Coverage Target:** 75% minimum
