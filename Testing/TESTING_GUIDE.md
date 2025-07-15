# MVPA Pipeline Testing Guide

## Overview

The MVPA pipeline now includes a comprehensive test suite built with pytest that provides thorough testing of all components including behavioral analysis, MVPA fitting, mask creation, and the new analysis class hierarchy.

## Test Structure

### Test Files

- **`test_pipeline_pytest.py`** - Main pytest test suite with comprehensive unit and integration tests
- **`test_pipeline.py`** - Original integration test (kept for backward compatibility)
- **`run_tests.py`** - Test launcher script with convenient options
- **`pytest.ini`** - Pytest configuration file

### Test Categories

The test suite is organized into several categories using pytest markers:

1. **Unit Tests** (`@pytest.mark.unit`) - Test individual functions
2. **Integration Tests** (`@pytest.mark.integration`) - Test complete workflows
3. **Behavioral Tests** (`@pytest.mark.behavioral`) - Test behavioral analysis components
4. **MVPA Tests** (`@pytest.mark.mvpa`) - Test MVPA analysis components
5. **Geometry Tests** (`@pytest.mark.geometry`) - Test geometry analysis components
6. **Factory Tests** (`@pytest.mark.factory`) - Test analysis factory pattern
7. **Error Tests** (`@pytest.mark.error`) - Test error handling
8. **Performance Tests** (`@pytest.mark.performance`) - Test performance characteristics

## Installation

### Install Testing Dependencies

```bash
pip install -r requirements.txt
```

This will install pytest and all related testing dependencies:
- `pytest` - Main testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Advanced mocking capabilities
- `pytest-timeout` - Test timeout handling
- `pytest-xdist` - Parallel test execution
- `coverage` - Coverage measurement

## Running Tests

### Basic Usage

```bash
# Run all tests
python run_tests.py

# Run all tests with pytest directly
pytest test_pipeline_pytest.py
```

### Test Selection

```bash
# Run specific test categories
python run_tests.py --unit                 # Unit tests only
python run_tests.py --integration          # Integration tests only
python run_tests.py --behavioral           # Behavioral analysis tests
python run_tests.py --mvpa                 # MVPA analysis tests
python run_tests.py --geometry             # Geometry analysis tests
python run_tests.py --factory              # Analysis factory tests
python run_tests.py --error                # Error handling tests
python run_tests.py --performance          # Performance tests

# Run fast tests only (exclude slow tests)
python run_tests.py --fast

# Run specific test class
python run_tests.py --class TestBehavioralAnalysis

# Run specific test method
python run_tests.py --method test_hyperbolic_discount_function

# Run specific test class and method
python run_tests.py --class TestBehavioralAnalysis --method test_hyperbolic_discount_function
```

### Output Options

```bash
# Verbose output
python run_tests.py --verbose

# Quiet output
python run_tests.py --quiet

# Generate HTML coverage report
python run_tests.py --html

# Run with coverage reporting
python run_tests.py --coverage

# Disable coverage reporting
python run_tests.py --no-cov
```

### Performance Options

```bash
# Run tests in parallel (auto-detect cores)
python run_tests.py --parallel

# Run tests in parallel with specific number of processes
python run_tests.py --parallel 4

# Run tests with specific traceback format
python run_tests.py --tb short  # Options: short, long, auto, no
```

### File Options

```bash
# Run tests from specific file
python run_tests.py --file test_pipeline_pytest.py

# Run original test_pipeline.py instead
python run_tests.py --old-tests
```

## Test Components

### 1. Analysis Base Classes Tests

Tests the new analysis class hierarchy:

```python
class TestAnalysisBase:
    def test_analysis_factory_creation(self):
        """Test AnalysisFactory can create all analysis types"""
        
    def test_base_analysis_initialization(self):
        """Test BaseAnalysis initialization"""
        
    def test_base_analysis_result_handling(self):
        """Test result saving and loading"""
        
    def test_base_analysis_cache_management(self):
        """Test data cache management"""
```

### 2. Behavioral Analysis Tests

Tests behavioral modeling and validation:

```python
class TestBehavioralAnalysis:
    def test_hyperbolic_discount_function(self):
        """Test hyperbolic discount function"""
        
    def test_subjective_value_calculation(self):
        """Test subjective value calculation"""
        
    def test_discount_rate_fitting(self):
        """Test discount rate fitting"""
        
    def test_behavioral_data_validation(self):
        """Test behavioral data validation"""
```

### 3. MVPA Analysis Tests

Tests pattern extraction and decoding:

```python
class TestMVPAAnalysis:
    def test_extract_trial_data(self):
        """Test trial data extraction"""
        
    def test_decode_choices(self):
        """Test choice decoding"""
        
    def test_decode_continuous_variable(self):
        """Test continuous variable decoding"""
        
    def test_create_summary_dataframe(self):
        """Test MVPA summary dataframe creation"""
```

### 4. Geometry Analysis Tests

Tests neural geometry and dimensionality reduction:

```python
class TestGeometryAnalysis:
    def test_compute_neural_rdm(self):
        """Test RDM computation"""
        
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction"""
        
    def test_behavioral_geometry_correlation(self):
        """Test behavioral-geometry correlations"""
        
    def test_compare_embeddings_by_condition(self):
        """Test condition-based embedding comparisons"""
```

### 5. Mask Creation Tests

Tests mask loading and validation:

```python
class TestMaskCreation:
    def test_create_maskers_success(self):
        """Test successful masker creation"""
        
    def test_create_maskers_missing_mask(self):
        """Test masker creation with missing mask"""
        
    def test_create_maskers_load_failure(self):
        """Test masker creation with load failure"""
```

### 6. Error Handling Tests

Tests error conditions and edge cases:

```python
class TestErrorHandling:
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        
    def test_invalid_roi_name(self):
        """Test handling of invalid ROI names"""
        
    def test_processing_stats_update(self):
        """Test processing statistics updates"""
```

## Test Fixtures

The test suite uses pytest fixtures for common test data:

### Data Fixtures

```python
@pytest.fixture
def synthetic_behavioral_data():
    """Create synthetic behavioral data for testing"""
    
@pytest.fixture
def synthetic_fmri_data():
    """Create synthetic fMRI data for testing"""
    
@pytest.fixture
def synthetic_neural_patterns():
    """Create synthetic neural pattern data"""
    
@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    
@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests"""
```

### Using Fixtures

```python
def test_behavioral_analysis(synthetic_behavioral_data, mock_config):
    """Test using synthetic data and mock config"""
    from behavioral_analysis import BehavioralAnalysis
    
    analysis = BehavioralAnalysis(config=mock_config)
    result = analysis.validate_behavioral_data(synthetic_behavioral_data)
    assert result['valid'] == True
```

## Mocking and Patching

The test suite uses extensive mocking to isolate units and avoid external dependencies:

### Example: Mock Data Loading

```python
@patch('behavioral_analysis.BehavioralAnalysis.load_behavioral_data')
def test_process_subject_success(mock_load_data, mock_config, synthetic_behavioral_data):
    """Test successful subject processing"""
    # Mock data loading
    mock_load_data.return_value = synthetic_behavioral_data
    
    analysis = BehavioralAnalysis(config=mock_config)
    result = analysis.process_subject('test_subject_001')
    
    assert result['success'] == True
    assert 'k' in result
```

### Example: Mock External Functions

```python
@patch('mvpa_analysis.extract_neural_patterns')
def test_extract_trial_data(mock_extract, mock_config, synthetic_fmri_data):
    """Test trial data extraction"""
    # Mock pattern extraction
    mock_extract.return_value = {
        'success': True,
        'patterns': np.random.randn(50, 100)
    }
    
    analysis = MVPAAnalysis(config=mock_config)
    X = analysis.extract_trial_data(synthetic_fmri_data, behavioral_data, 'test_roi')
    
    assert X.shape == (50, 100)
```

## Coverage Reporting

### Generate Coverage Report

```bash
# Terminal coverage report
python run_tests.py --coverage

# HTML coverage report
python run_tests.py --html

# Both terminal and HTML reports
python run_tests.py --coverage --html
```

### Coverage Files

- **`htmlcov/index.html`** - HTML coverage report (open in browser)
- **`.coverage`** - Coverage database file
- **`htmlcov/`** - Directory containing HTML coverage files

### Coverage Goals

The test suite aims for:
- **80%+ overall coverage** (enforced by pytest configuration)
- **90%+ coverage for core analysis classes**
- **100% coverage for critical functions** (hyperbolic discounting, data validation)

## Continuous Integration

### Running Tests in CI

```bash
# Basic CI test run
pytest test_pipeline_pytest.py --cov=. --cov-report=xml --cov-fail-under=80

# CI test run with parallel execution
pytest test_pipeline_pytest.py --cov=. --cov-report=xml --cov-fail-under=80 -n auto
```

### Test Configuration for CI

Add to your CI configuration:

```yaml
# Example for GitHub Actions
- name: Run tests
  run: |
    python -m pytest test_pipeline_pytest.py \
      --cov=. \
      --cov-report=xml \
      --cov-fail-under=80 \
      --timeout=300 \
      -v
```

## Performance Testing

### Running Performance Tests

```bash
# Run performance tests only
python run_tests.py --performance

# Run performance tests with profiling
python run_tests.py --performance --verbose
```

### Performance Benchmarks

The performance tests check:
- **Behavioral analysis** completes in < 5 seconds for 1000 trials
- **Cache operations** are efficient for large datasets
- **Memory usage** is within expected bounds

## Testing Best Practices

### 1. Test Organization

- Use descriptive test names that explain what is being tested
- Group related tests in test classes
- Use appropriate pytest markers for categorization
- Keep tests focused and isolated

### 2. Test Data

- Use fixtures for common test data
- Create synthetic data that's representative but controlled
- Mock external dependencies to avoid flaky tests
- Use temporary directories for file operations

### 3. Assertions

- Use specific assertions that clearly indicate what failed
- Test both success and failure scenarios
- Check edge cases and boundary conditions
- Validate both values and types

### 4. Error Testing

- Test that appropriate errors are raised for invalid inputs
- Test error handling and graceful degradation
- Test logging and error reporting functionality
- Test resource cleanup on failures

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test with verbose output
python run_tests.py --class TestBehavioralAnalysis --method test_hyperbolic_discount_function --verbose

# Run with debugger integration
python -m pytest test_pipeline_pytest.py::TestBehavioralAnalysis::test_hyperbolic_discount_function -v -s --pdb
```

### Debug Output

```bash
# Show print statements and logging
python run_tests.py --verbose -s

# Show test durations to identify slow tests
python run_tests.py --durations=10
```

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Issues**: Check that patches target the correct module paths
3. **Fixture Errors**: Verify fixture dependencies and scopes
4. **Timeout Issues**: Some tests may need longer timeouts for slower systems

## Advanced Usage

### Custom Test Markers

Add custom markers to `pytest.ini`:

```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU
    network: marks tests as requiring network access
```

### Test Parametrization

```python
@pytest.mark.parametrize("k_value,expected_decay", [
    (0.01, "moderate"),
    (0.1, "high"),
    (0.001, "low")
])
def test_discount_rate_categories(k_value, expected_decay):
    """Test discount rate categorization"""
    category = categorize_discount_rate(k_value)
    assert category == expected_decay
```

### Test Data Factories

```python
def create_behavioral_data(n_trials=50, choice_rate=0.5, k_value=0.02):
    """Factory function for creating test behavioral data"""
    # Generate controlled synthetic data
    return behavioral_data
```

## Troubleshooting

### Common Error Messages

1. **"pytest: command not found"**
   - Solution: Install pytest with `pip install pytest`

2. **"ModuleNotFoundError: No module named 'X'"**
   - Solution: Install dependencies with `pip install -r requirements.txt`

3. **"Collection failed"**
   - Solution: Check syntax errors in test files

4. **"Fixture 'X' not found"**
   - Solution: Verify fixture names and scopes

### Getting Help

```bash
# Show pytest help
pytest --help

# Show available markers
pytest --markers

# Show test collection without running
pytest --collect-only

# Show pytest version and plugins
pytest --version
```

## Summary

The new pytest-based testing system provides:

- **Comprehensive Coverage**: Tests all major components and edge cases
- **Organized Structure**: Clear categorization with pytest markers
- **Flexible Execution**: Multiple ways to run tests with different options
- **Detailed Reporting**: Coverage reports and performance metrics
- **CI Integration**: Ready for continuous integration workflows
- **Developer Friendly**: Easy debugging and development workflow

The test suite ensures reliability and maintainability of the MVPA pipeline while providing confidence in code changes and refactoring efforts. 