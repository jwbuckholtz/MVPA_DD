# MVPA Utilities Module Documentation

A comprehensive module for Multi-Voxel Pattern Analysis (MVPA) procedures used in the delay discounting fMRI analysis pipeline. This module centralizes all machine learning operations, eliminates code duplication, and provides consistent, well-tested implementations.

## Overview

The `mvpa_utils.py` module provides:

- **Centralized MVPA Functions**: Eliminates duplicate code across pipeline components
- **Multiple Algorithms**: Support for various classifiers and regressors
- **Consistent Interfaces**: Standardized function signatures and return formats
- **Comprehensive Validation**: Robust error handling and data quality checks
- **Advanced Features**: Pattern extraction, dimensionality reduction, feature importance
- **Easy Configuration**: Centralized parameter management
- **Integration Ready**: Works seamlessly with existing pipeline components

## Key Functions

### Core MVPA Functions

#### `run_classification(X, y, algorithm='svm', **kwargs)`
Perform classification with cross-validation and permutation testing.

**Parameters:**
- `X`: Feature matrix (n_samples, n_features)
- `y`: Class labels (n_samples,)
- `algorithm`: Classifier type ('svm', 'logistic', 'rf')
- `cv_strategy`: Cross-validation method ('stratified', 'kfold', 'group', 'loo')
- `n_permutations`: Number of permutation tests (default: 1000)
- `feature_selection`: Whether to perform feature selection
- `roi_name`: ROI name for reporting

**Returns:**
- Dictionary with accuracy scores, permutation results, and metadata

#### `run_regression(X, y, algorithm='ridge', **kwargs)`
Perform regression with cross-validation and permutation testing.

**Parameters:**
- `X`: Feature matrix (n_samples, n_features)
- `y`: Continuous target values (n_samples,)
- `algorithm`: Regressor type ('ridge', 'lasso', 'elastic', 'svr', 'rf')
- `cv_strategy`: Cross-validation method
- `variable_name`: Target variable name for reporting

**Returns:**
- Dictionary with R² scores, MSE, permutation results, and metadata

### Pattern Extraction

#### `extract_neural_patterns(img, events_df, masker, **kwargs)`
Extract trial-wise neural patterns with multiple methods.

**Pattern Types:**
- `'single_timepoint'`: Single TR at hemodynamic peak
- `'average_window'`: Average over time window
- `'temporal_profile'`: Full temporal profile
- `'peak_detection'`: Automatic peak detection

**Parameters:**
- `img`: fMRI data (nibabel image or path)
- `events_df`: Trial events with onset times
- `masker`: Fitted NiftiMasker object
- `pattern_type`: Extraction method
- `tr`: Repetition time in seconds
- `hemi_lag`: Hemodynamic lag in TRs
- `window_size`: Time window size for averaging

### Dimensionality Reduction

#### `run_dimensionality_reduction(X, method='pca', n_components=10)`
Perform dimensionality reduction with multiple algorithms.

**Methods:**
- `'pca'`: Principal Component Analysis
- `'mds'`: Multidimensional Scaling
- `'tsne'`: t-Distributed Stochastic Neighbor Embedding
- `'isomap'`: Isometric Mapping

**Returns:**
- Dictionary with embedding, reducer object, and explained variance

### Advanced Features

#### `run_searchlight_analysis(img, y, **kwargs)`
Whole-brain searchlight analysis.

#### `compute_feature_importance(X, y, method='univariate')`
Feature importance and selection.

**Methods:**
- `'univariate'`: Statistical tests (F-test)
- `'model_based'`: Random Forest importance

### Convenience Functions

#### `run_choice_classification(X, choices, roi_name='unknown')`
Specialized function for binary choice classification.

#### `run_continuous_decoding(X, values, variable_name='unknown')`
Specialized function for continuous variable regression.

## Configuration

### MVPAConfig Class

Central configuration for all MVPA parameters:

```python
class MVPAConfig:
    # Cross-validation settings
    CV_FOLDS = 5
    CV_SHUFFLE = True
    CV_RANDOM_STATE = 42
    
    # Permutation testing
    N_PERMUTATIONS = 1000
    PERM_RANDOM_STATE = 42
    
    # Algorithm defaults
    DEFAULT_CLASSIFIER = 'svm'
    DEFAULT_REGRESSOR = 'ridge'
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'linear'
    
    # Ridge regression
    RIDGE_ALPHA = 1.0
    
    # Feature selection
    FEATURE_SELECTION = False
    N_FEATURES = 1000
    
    # Preprocessing
    STANDARDIZE = True
    
    # Parallel processing
    N_JOBS = 1
    
    # Quality control
    MIN_SAMPLES_PER_CLASS = 5
    MIN_VARIANCE_THRESHOLD = 1e-8
```

### Configuration Updates

```python
from mvpa_utils import update_mvpa_config

# Update specific parameters
update_mvpa_config(
    cv_folds=10,
    n_permutations=5000,
    svm_c=0.1
)
```

## Usage Examples

### Basic Classification

```python
from mvpa_utils import run_classification

# Load your data
X = neural_patterns  # (n_trials, n_voxels)
y = choice_labels    # (n_trials,) binary labels

# Run classification
result = run_classification(
    X, y,
    algorithm='svm',
    cv_strategy='stratified',
    n_permutations=1000,
    roi_name='striatum'
)

if result['success']:
    print(f"Accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
    print(f"P-value: {result['p_value']:.4f}")
```

### Continuous Variable Regression

```python
from mvpa_utils import run_regression

# Decode subjective value
result = run_regression(
    X, subjective_values,
    algorithm='ridge',
    cv_strategy='kfold',
    variable_name='subjective_value',
    roi_name='vmpfc'
)

if result['success']:
    print(f"R²: {result['mean_r2']:.3f} ± {result['std_r2']:.3f}")
```

### Pattern Extraction

```python
from mvpa_utils import extract_neural_patterns
from nilearn.input_data import NiftiMasker

# Setup masker
masker = NiftiMasker(mask_img='roi_mask.nii.gz', standardize=True)

# Extract patterns
result = extract_neural_patterns(
    img='subject_func.nii.gz',
    events_df=behavioral_events,
    masker=masker,
    pattern_type='average_window',
    window_size=3
)

if result['success']:
    patterns = result['patterns']  # (n_trials, n_voxels)
```

### Dimensionality Reduction

```python
from mvpa_utils import run_dimensionality_reduction

# PCA embedding
result = run_dimensionality_reduction(
    X, method='pca', n_components=10
)

if result['success']:
    embedding = result['embedding']  # (n_trials, 10)
    explained_var = result['explained_variance']
```

## Integration with Existing Pipeline

### Refactoring Example

**Before (duplicated code):**
```python
# In delay_discounting_mvpa_pipeline.py
def decode_choices(self, X, y, roi_name):
    classifier = SVC(kernel='linear', C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    # ... permutation test code ...
    return results

# In other modules - similar code repeated
```

**After (centralized):**
```python
# In delay_discounting_mvpa_pipeline.py
def decode_choices(self, X, y, roi_name):
    return run_choice_classification(
        X, y, roi_name=roi_name,
        n_permutations=self.config.N_PERMUTATIONS
    )
```

### Pipeline Integration

```python
# In main pipeline
from mvpa_utils import update_mvpa_config

# Configure MVPA utilities to match pipeline config
update_mvpa_config(
    cv_folds=config.CV_FOLDS,
    n_permutations=config.N_PERMUTATIONS,
    n_jobs=1  # Conservative for memory
)

# Use centralized functions throughout pipeline
choice_results = mvpa_analysis.decode_choices(X, y, roi_name)
continuous_results = mvpa_analysis.decode_continuous_variable(X, values, roi_name, var_name)
```

## Error Handling

### Custom Exceptions

```python
from mvpa_utils import MVPAError

try:
    result = run_classification(X, y)
except MVPAError as e:
    print(f"MVPA error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Data Validation

The module automatically validates:
- Sample/feature dimension matching
- Sufficient samples for cross-validation
- Class balance for classification
- NaN/infinite value detection
- Data variance thresholds

## Performance Considerations

### Memory Management

```python
# Configure for large datasets
update_mvpa_config(
    n_jobs=1,           # Reduce parallel jobs
    n_permutations=100  # Reduce permutation tests
)

# Use feature selection for high-dimensional data
result = run_classification(
    X, y, 
    feature_selection=True,
    n_features=1000
)
```

### Speed Optimization

```python
# Fast classification for exploratory analysis
result = run_classification(
    X, y,
    n_permutations=100,     # Reduced permutations
    cv_strategy='kfold'     # Faster than stratified
)
```

## Testing and Validation

### Demo Script

Run the comprehensive demo to test all functionality:

```bash
# Basic demos
python demo_mvpa_utils.py --demo-type basic --verbose

# Advanced features
python demo_mvpa_utils.py --demo-type advanced --plot

# Integration testing
python demo_mvpa_utils.py --demo-type integration

# All demonstrations
python demo_mvpa_utils.py --demo-type all --plot --verbose
```

### Unit Testing

The module includes extensive validation:
- Input data validation
- Algorithm parameter validation
- Cross-validation setup verification
- Result format consistency checks

## Algorithm Details

### Supported Classifiers

1. **SVM (Support Vector Machine)**
   - Linear kernel for interpretability
   - Configurable C parameter
   - Probability estimates available

2. **Logistic Regression**
   - L2 regularization
   - Multiple solvers available
   - Fast and interpretable

3. **Random Forest**
   - Feature importance available
   - Robust to overfitting
   - Handles non-linear relationships

### Supported Regressors

1. **Ridge Regression**
   - L2 regularization
   - Good for high-dimensional data
   - Stable and interpretable

2. **Lasso Regression**
   - L1 regularization
   - Automatic feature selection
   - Sparse solutions

3. **Elastic Net**
   - Combined L1/L2 regularization
   - Balances Ridge and Lasso
   - Configurable mixing parameter

4. **Support Vector Regression**
   - Non-linear kernels available
   - Robust to outliers
   - Epsilon-insensitive loss

5. **Random Forest Regression**
   - Feature importance
   - Non-linear relationships
   - Ensemble robustness

## Best Practices

### 1. Data Preparation

```python
# Ensure proper data format
assert X.ndim == 2, "X must be 2D (samples x features)"
assert len(y) == X.shape[0], "Sample count mismatch"

# Check for missing values
assert not np.any(np.isnan(X)), "Remove NaN values"
assert not np.any(np.isnan(y)), "Remove NaN labels"
```

### 2. Algorithm Selection

```python
# For interpretability: use linear models
result = run_classification(X, y, algorithm='svm')
result = run_regression(X, y, algorithm='ridge')

# For complex patterns: use ensemble methods
result = run_classification(X, y, algorithm='rf')
result = run_regression(X, y, algorithm='rf')
```

### 3. Cross-Validation Strategy

```python
# For balanced datasets
result = run_classification(X, y, cv_strategy='stratified')

# For time series or grouped data
result = run_classification(X, y, cv_strategy='group', groups=subject_ids)

# For small datasets
result = run_classification(X, y, cv_strategy='loo')
```

### 4. Statistical Testing

```python
# Standard permutation testing
result = run_classification(X, y, n_permutations=1000)

# For exploratory analysis (faster)
result = run_classification(X, y, n_permutations=100)

# For publication (more rigorous)
result = run_classification(X, y, n_permutations=5000)
```

## Migration Guide

### From Old Pipeline to MVPA Utils

1. **Replace classification functions:**
   ```python
   # Old
   def decode_choices(self, X, y, roi_name):
       # 30+ lines of repeated code
   
   # New
   def decode_choices(self, X, y, roi_name):
       return run_choice_classification(X, y, roi_name=roi_name)
   ```

2. **Replace regression functions:**
   ```python
   # Old
   def decode_continuous_variable(self, X, y, roi_name, variable_name):
       # 25+ lines of repeated code
   
   # New
   def decode_continuous_variable(self, X, y, roi_name, variable_name):
       return run_continuous_decoding(X, y, variable_name=variable_name, roi_name=roi_name)
   ```

3. **Update imports:**
   ```python
   # Add to imports
   from mvpa_utils import (
       run_classification, run_regression, extract_neural_patterns,
       run_dimensionality_reduction, update_mvpa_config
   )
   ```

4. **Configure integration:**
   ```python
   # In main pipeline initialization
   update_mvpa_config(
       cv_folds=config.CV_FOLDS,
       n_permutations=config.N_PERMUTATIONS
   )
   ```

## Troubleshooting

### Common Issues

1. **"Not enough samples for CV"**
   - Reduce `cv_folds` in configuration
   - Use `cv_strategy='loo'` for very small datasets

2. **"Insufficient choice variability"**
   - Check class balance in labels
   - Ensure binary classification has both classes

3. **Memory errors with large datasets**
   - Set `n_jobs=1` to reduce memory usage
   - Use `feature_selection=True`
   - Reduce `n_permutations`

4. **Slow performance**
   - Reduce `n_permutations` for exploratory analysis
   - Use simpler algorithms ('logistic' vs 'rf')
   - Consider data subsampling

### Debugging

```python
# Enable verbose error reporting
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data quality
from mvpa_utils import validate_input_data
try:
    validate_input_data(X, y, 'classification')
    print("Data validation passed")
except MVPAError as e:
    print(f"Data validation failed: {e}")

# Test with minimal settings
result = run_classification(
    X, y,
    n_permutations=10,  # Minimal for testing
    cv_strategy='kfold'
)
```

## Contributing

### Adding New Algorithms

1. **Extend classifier setup:**
   ```python
   def setup_classifier(algorithm, **kwargs):
       # Add new algorithm
       elif algorithm == 'new_classifier':
           return NewClassifier(**kwargs)
   ```

2. **Update documentation:**
   - Add algorithm to supported list
   - Include parameter descriptions
   - Add usage examples

3. **Add tests:**
   - Include in demo script
   - Verify with synthetic data
   - Test edge cases

### Code Style

- Follow existing function signatures
- Include comprehensive docstrings
- Use type hints where possible
- Handle errors gracefully
- Return consistent result formats

## References

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Nilearn MVPA Tutorial**: https://nilearn.github.io/
- **Cross-validation Best Practices**: Varoquaux et al. (2017) NeuroImage
- **Permutation Testing**: Nichols & Holmes (2002) Human Brain Mapping

## License

This module is part of the delay discounting analysis pipeline. Please follow the same license terms as the main project. 